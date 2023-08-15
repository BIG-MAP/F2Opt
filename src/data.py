"""Data processing.

- Parse data from the broker to a format that can be used by the optimiser.
- Prepare optimiser results in a format that can be send to the broker.
- Use schemas to validate data formats.
"""

import pandas as pd

from . import schemas
from .composition import sample_compositions_with_constraints


# Constraints


def get_constraints_from_limitations(limitations):
    """Get constraints for each task.

    Args:
        limitations (dict): A dict of tasks and limitations.
    Returns:
        Constraints dict: {task: constraints}.
    """
    constraints = {}
    for task_name, task_limitations in limitations.items():
        constraints[task_name] = get_constraints_list_from_task_limitations(task_limitations)
    return constraints


def get_constraints_list_from_task_limitations(task):
    """Get constraints from task limitations.

    Args:
        task (dict): A dict with limitations for a task.
    Returns:
        List of task constraints dicts.
    """
    # Extract limitation
    # TODO: The anyOf key should be removed from the schema in the future
    assert len(task["limitations"]["anyOf"]) == 1
    limitations = task["limitations"]["anyOf"][0]
    # Extract constraints for each formulation limitations
    constraints_list = []
    for formulation in limitations["formulation"]:
        constraints_list.append({
            "quantity": task["quantity"],
            "method": task["method"],
            "formulation": get_constraints_from_formulation_limitations(formulation),
        })
    return constraints_list


def get_constraints_from_formulation_limitations(formulation):
    """Get formulation constraints from formulation limitations.

    Args:
        formulation (list of dict): A list of chemicals with limitations.
    Returns:
        Formulation constraints dict.
    """
    chemicals, lower, upper, tolerance = [], [], [], []
    for chemical in formulation:
        chemicals.append(chemical["chemical"])
        l, u, t = get_constraints_from_fraction_limitations(chemical["fraction"])
        lower.append(l)
        upper.append(u)
        tolerance.append(t)
    assert len(formulation) == len(chemicals) == len(lower) == len(upper) == len(tolerance)
    constraints = {
        "chemicals": chemicals, "lower": lower, "upper": upper, "tolerance": tolerance
    }
    return constraints


def get_constraints_from_fraction_limitations(fraction):
    """Get fraction constraints from fraction limitations.

    Returns lower bound, upper bound and tolerance.
    - Values below tolerance should be rounded to zero.
    - Tolerance is zero by default.
    - If tolerance is defined, lower bound should be zero.

    Example input: [{"min": 0, "max": 0}, {"min": 0.1, "max": 1.0}]

    Supported input cases where 0 is the constant zero, a is min and b is max:
    - [(a, b)]:         Lower and upper bounds. Tolerance defaults zero.
    - [(0, 0), (a, b)]: Lower is zero and tolerance is a.
    - [(a, b), (0, 0)]: Lower is zero and tolerance is a.
    - [0, (a, b)]:      Lower is zero and tolerance is a.
    - [(a, b), 0]:      Lower is zero and tolerance is a.

    Args:
        fraction (list): Fraction limitations.
    Returns:
        Fraction constraints tuple of numbers(lower, upper, tolerance).
    """
    assert len(fraction) == 1 or len(fraction) == 2
    lower, upper, tolerance = None, None, 0.0
    for x in fraction:
        if isinstance(x, (int, float)):  # if it is a number, it is the lower bound
            if lower is not None:  # this is the second element in the list
                assert lower >= x, fraction
                assert x == 0.0, fraction
                tolerance = lower
            lower = x
        elif isinstance(x, dict):
            if x["min"] == x["max"]:  # if min and max are the same, it is the lower bound
                if lower is not None:  # this is the second element in the list
                    assert lower >= x["min"], fraction
                    assert x["min"] == 0.0, fraction
                    tolerance = lower
                lower = x["min"]
            else:  # min and max are different
                assert x["min"] < x["max"]
                if lower is None:
                    lower = x["min"]
                else:
                    assert lower == 0.0, fraction
                    tolerance = x["min"]
                upper = x["max"]
    assert lower is not None, (lower, upper, tolerance, fraction)
    assert upper is not None, (lower, upper, tolerance, fraction)
    assert tolerance == 0.0 or tolerance > lower, (lower, upper, tolerance, fraction)
    return lower, upper, tolerance


# Results


def get_dataframe_from_results(config, results):
    """Create dataframe from results.

    Args:
        config (dict): Optimiser configuration.
        results (dict): Collection of task results in dict format.
    Returns:
        Dataframe with results.
    """
    task_dfs = []  # List of dataframes for each task
    for task in [t for t in config["tasks"] if t["source"]]:
        task_results = results[task["name"]]
        assert len(task_results) == 1  # TODO: Only support for one quantity per task
        quantity_dfs = []  # List of dataframes for each quantity in the task
        for quantity, results_list in task_results.items():
            if len(results_list) == 0:
                continue
            else:
                rows = [get_row_from_result(result) for result in results_list]
                quantity_df = pd.DataFrame(rows)
                quantity_df["task"] = task["name"]
                # TODO: Rename columns: quantity to objective (from task config)?
                quantity_dfs.append(quantity_df)
        if len(quantity_dfs) == 0:
            continue
        else:
            # TODO: Outer join quantity dfs on request internal reference (and chemical fractions)
            assert len(quantity_dfs) == 1  # TODO: Only support for one quantity per method
            task_df = quantity_dfs[0]  # TODO: Outer join instead
            task_dfs.append(task_df)
    if len(task_dfs) == 0:
        # Return empty dataframe
        return pd.DataFrame()
    else:
        # Stack task dataframes
        df = pd.concat(task_dfs, axis=0, ignore_index=True)
        # Sort columns by name
        df = df.reindex(sorted(df.columns), axis=1)
        assert df["result_id"].is_unique  # One row per result id
        assert df["request_id"].is_unique  # One row per request id
        return df


def get_row_from_result(result):
    """Create row dict from a single result.

    Args:
        result (dict): A single result in dict format.
    Returns:
        A flat dict representing a dataframe row with the result.
    """
    assert len(result["result"]["method"]) == 1
    row = {}
    row["result_id"] = result["uuid"]
    row["ctime"] = result["ctime"]
    row["quantity"] = result["result"]["quantity"]
    row["method"] = result["result"]["method"][0]
    # TODO: request_id is not the same as the request internal reference
    row["request_id"] = result["result"]["request_uuid"]
    formulation = result["result"]["data"]["run_info"]["formulation"]
    for formulation_component in formulation:
        smiles = formulation_component["chemical"]["SMILES"]
        fraction = formulation_component["fraction"]
        row[f"x.{smiles}"] = fraction
    quantity = result["result"]["data"][result["result"]["quantity"]]
    values = quantity["values"]
    value = sum(values) / len(values)  # Compute the mean value
    row["y."+result["result"]["quantity"]] = value
    row["temperature"] = quantity["temperature"]
    return row


# Requests


def get_candidates_from_constraints(constraints_list, num_samples=1):
    """Get dataframe of candidate compositions from list of constraints.

    Args:
        constraints_list (list): A list of constraints for a single task.
        num_samples (int): Number of samples per set of constraints.
    Returns:
        Dataframe of candidates with columns for each chemical.
    """
    assert all("quantity" in c for c in constraints_list), constraints_list
    assert all("method" in c for c in constraints_list), constraints_list
    assert all("formulation" in c for c in constraints_list), constraints_list
    assert len(set(c["quantity"] for c in constraints_list)) == 1, constraints_list
    assert len(set(c["method"] for c in constraints_list)) == 1, constraints_list
    # Sample candidate compositions for each set of formulation constraints
    dfs = []  # List of candidate dataframes
    for constraints in constraints_list:
        formulation = constraints["formulation"]
        compositions = sample_compositions_with_constraints(
            lower=formulation["lower"],
            upper=formulation["upper"],
            tolerance=formulation["tolerance"],
            num_samples=num_samples,
        )
        # TODO: Perhaps the columns should be 'x.SMILES'
        columns = [chem["SMILES"] for chem in formulation["chemicals"]]
        dfs.append(pd.DataFrame(compositions, columns=columns))
    # Stack candidate samples and fill missing values with zeros
    candidates_df = pd.concat(dfs, axis=0, ignore_index=True).fillna(0)
    return candidates_df


def get_best_candidates(config, df, constraints):
    """Get best candidates for each task and quantity.

    Args:
        config (dict): Optimiser configuration.
        df (pandas.DataFrame): Dataframe of observed results.
        constraints (dict): Constraints for each task.
    Returns:
        Dict of best candidates for each task.
    """
    # TODO: Handle empty and small dataframes
    # TODO: Check df has the correct columns (chemicals and quantities)
    # TODO: Results columns are 'x.SMILES' whereas candidate columns are just 'SMILES'
    candidate_list = []
    for task in [t for t in config["tasks"] if t["request"]]:
        task_constraints_list = constraints[task["name"]]
        assert all(tc["method"] == task["method"] for tc in task_constraints_list)
        # Create dict of all chemicals included in the method constraints
        smiles_to_chemicals = {}  # {smiles: chemical}
        for task_constraints in task_constraints_list:
            for chemical in task_constraints["formulation"]["chemicals"]:
                smiles_to_chemicals[chemical["SMILES"]] = chemical
        # Get dataframe of candidates for the method
        # TODO: Use config to set number of samples
        candidates_df = get_candidates_from_constraints(task_constraints_list, num_samples=1)
        # Get random candidate for the method
        # TODO: Implement data-driven optimisation instead of random sampling
        candidate_row = candidates_df.sample(n=1).squeeze()  # Sample one row as Series
        # Extract chemicals and fractions from row
        chemicals = [smiles_to_chemicals[smiles] for smiles in candidate_row.index]
        fractions = candidate_row.values.tolist()
        assert len(chemicals) == len(fractions)
        # Prepare candidate dict
        candidate = {
            "task": task,  # Task configuration
            "chemicals": chemicals,  # List of chemical dicts
            "fractions": fractions,  # List of fractions
            # "predictions": None,  # TODO: Include the predicted quantities?
        }
        candidate_list.append(candidate)
    return candidate_list


def get_requests_from_candidate(config, candidate):
    """Prepare request from candidate.

    Args:
        config (dict): Optimiser configuration.
        candidate (dict): Candidate.
    Returns:
        List of requests.
    """
    task = candidate["task"]  # Task configuration
    formulation = []  # List of formulation components
    for chemical, fraction in zip(candidate["chemicals"], candidate["fractions"]):
        formulation_component = {
            "chemical": chemical,
            "fraction": fraction,
            "fraction_type": "molPerMol"
        }
        formulation.append(formulation_component)
    parameters = {
        "formulation": formulation,
        "temperature": task["parameters"]["temperature"],
    }
    # In mult-objective optimisation, create a request for each quantity
    # TODO: Add common reference to the requests, not yet implemented in the API
    # Use a common reference for the requests so they can be matched
    # reference = f"{config['name']}_{datetime.now().isoformat()}"
    requests = []
    for quantity in task["quantities"].keys():
        request = {
            "quantity": quantity,
            "methods": [task["method"]],
            "parameters": {task["method"]: parameters},
            "tenant_uuid": config["id"],
        }
        schemas.Request(**request)  # Validate request with schema
        requests.append(request)
    return requests
