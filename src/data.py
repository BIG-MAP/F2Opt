"""Data processing.

- Parse data from the broker to a format that can be used by the optimiser.
- Prepare optimiser results in a format that can be send to the broker.
- Use schemas to validate data formats.
"""

from datetime import datetime

import pandas as pd

from . import schemas
from .composition import sample_compositions_with_constraints


# Constraints


def get_constraints_from_method_limitations(method_limitations):
    """Get constraints for each method.

    Args:
        method_limitations (list): A list of methods with limitations.
    Returns:
        Constraints dict: {method: constraints}.
    """
    constraints = {}
    for method in method_limitations:
        constraints[method["method"]] = get_constraints_from_method(method)
    return constraints


def get_constraints_from_method(method):
    """Get constraints from method limitations.

    Args:
        method (dict): A method with limitations.
    Returns:
        List of method constraints dicts.
    """
    # Extract limitation from method
    assert len(method["limitations"]) == 1
    limitations = method["limitations"][0]
    # Extract constraints for each formulation limitations
    constraints = []
    for formulation in limitations["formulation"]:
        constraints.append({
            "quantity": method["quantity"],
            "method": method["method"],
            "formulation": get_constraints_from_formulation(formulation),
        })
    return constraints


def get_constraints_from_formulation(formulation):
    """Get formulation constraints from formulation limitations.

    Args:
        formulation (list of dict): A list of chemicals with limitations.
    Returns:
        Formulation constraints dict.
    """
    chemicals, lower, upper, tolerance = [], [], [], []
    for chemical in formulation:
        chemicals.append(chemical["chemical"])
        l, u, t = get_constraints_from_fraction(chemical["fraction"])
        lower.append(l)
        upper.append(u)
        tolerance.append(t)
    assert len(formulation) == len(chemicals) == len(lower) == len(upper) == len(tolerance)
    constraints = {
        "chemicals": chemicals, "lower": lower, "upper": upper, "tolerance": tolerance
    }
    return constraints


def get_constraints_from_fraction(fraction):
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
    """Create dataframe from results."""
    raise NotImplementedError


def get_row_from_result(result):
    """Create row from result."""
    raise NotImplementedError


# Requests


def get_candidates_from_method_constraints(constraints_list, num_samples=1):
    """Get dataframe of candidate compositions from list of constraints.

    Args:
        constraints_list (list): A list of constraints for a single method.
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
        columns = [chem["SMILES"] for chem in formulation["chemicals"]]
        dfs.append(pd.DataFrame(compositions, columns=columns))
    # Stack candidate samples and fill missing values with zeros
    candidates_df = pd.concat(dfs, axis=0, ignore_index=True).fillna(0)
    return candidates_df


def get_best_candidates(config, df, constraints):
    """Get best candidates for all target methods and quantities.

    Args:
        config (dict): Optimiser configuration.
        df (pandas.DataFrame): Dataframe of observed data.
        constraints (dict): Constraints for each method.
    Returns:
        Dict of best candidates for each target method.
    """
    # TODO: Check df has the correct columns (chemicals and quantities)
    result = {}  # {method: candidate}
    for target in config["targets"]:
        # method_name = method_config["method"]
        method_constraints = constraints[target["method"]]
        assert all(cons["method"] == target["method"] for cons in method_constraints)
        # Create dict of all chemicals included in the method constraints
        all_chemicals = {}  # {smiles: chemical}
        for cons in method_constraints:
            for chemical in cons["formulation"]["chemicals"]:
                all_chemicals[chemical["SMILES"]] = chemical
        # Get dataframe of candidates for the method
        # TODO: Use config to set number of samples
        candidates_df = get_candidates_from_method_constraints(method_constraints, num_samples=1)
        # Get random candidate for the method
        # TODO: Implement data-driven optimisation instead of random sampling
        candidate_row = candidates_df.sample(n=1)  # Sample row
        # Extract chemicals and fractions from row
        chemicals = [all_chemicals[smiles] for smiles in candidate_row.columns]
        fractions = candidate_row.values[0].tolist()
        assert len(chemicals) == len(fractions)
        # Prepare candidate
        candidate = {
            # "quantity": target["quantity"],
            # "method": target["method"],
            "target": target,
            # "method_constraints": method_constraints,
            "chemicals": chemicals,
            "fractions": fractions,
            # "objectives": None,  # TODO: Include the predicted value?
        }
        result[target["method"]] = candidate
    return result


def get_requests_from_candidate(config, candidate):
    """Prepare request from candidate.

    Args:
        config (dict): Optimiser configuration.
        candidate (dict): Candidate.
    Returns:
        List of requests.
    """
    target = candidate["target"]
    # Formulation: List[FINALES2_schemas.classes_common.FormulationComponent]
    formulation = []
    for chemical, fraction in zip(candidate["chemicals"], candidate["fractions"]):
        formulation_component = {
            "chemical": chemical,
            "fraction": fraction,
            "fraction_type": "molPerMol"
        }
        formulation.append(formulation_component)
    # FINALES2_schemas.classes_input...
    parameters = {
        "formulation": formulation,
        "temperature": target["defaults"]["temperature"],
    }
    requests = []
    identifier = f"{config['name']}_{datetime.now().isoformat()}"
    for quantity in target["quantities"].keys():
        # FINALES2.server.schemas.Request
        request = {
            "quantity": quantity,
            "methods": [target["method"]],
            "parameters": {target["method"]: parameters},
            "tenant_uuid": identifier,
        }
        schemas.Request(**request)  # Validate request with schema
        requests.append(request)
    return requests
