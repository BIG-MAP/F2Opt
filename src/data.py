"""Data processing.

Parse data from the broker to a format that can be used by the optimiser.
Prepare optimiser results in a format that can be send to the broker.
Use schemas to validate data formats.
"""

from collections import defaultdict
from datetime import datetime

from . import schemas
from .composition import sample_compositions_with_constraints


def get_dataframe_from_results(config, results):
    """Create dataframe from results."""
    raise NotImplementedError


def get_row_from_result(result):
    """Create row from result."""
    raise NotImplementedError


def get_constraints_from_limitations(config, limitations):
    """Get constraints for each target in the config."""
    # TODO: Update to current format of limitations
    constraints = defaultdict(dict)  # {method: constraints}
    for limitation in limitations:
        formulation = limitation["limitations"]["formulation"]
        chemicals = [x["chemical"] for x in formulation]  # list of available chemicals
        lower = [x["fraction"]["min"] for x in formulation]  # list of lower bounds
        upper = [x["fraction"]["max"] for x in formulation]  # list of upper bounds
        tolerance = [0.0 for x in formulation]  # list of tolerances
        assert len(formulation) == len(chemicals) == len(lower) == len(upper) == len(tolerance)
        # constraints[target["name"]]["formulation"] = {
        constraints[limitation["method"]]["formulation"] = {
            "chemicals": chemicals, "lower": lower, "upper": upper, "tolerance": tolerance
        }
    assert len(constraints) == len(config["targets"])
    assert all(target["method"] in constraints for target in config["targets"])
    return constraints


def get_best_candidates(config, df, constraints):
    """Get best candidates for all targets and quantities."""
    # TODO: Implement data-driven optimisation instead of random sampling
    candidates = {}
    for target in config["targets"]:
        method = target["method"]
        chemicals = constraints[method]["formulation"]["chemicals"]
        # Generate random candidates within constraints
        fractions = sample_compositions_with_constraints(
            lower=constraints[method]["formulation"]["lower"],
            upper=constraints[method]["formulation"]["upper"],
            tolerance=constraints[method]["formulation"]["tolerance"],
        )[0]
        assert len(chemicals) == len(fractions)
        candidate = {
            "target": target,
            "chemicals": chemicals,
            "fractions": fractions,
            # "objectives": None,  # TODO: Include the predicted value?
        }
        candidates[target["name"]] = candidate
    return candidates


def get_requests_from_candidate(config, candidate):
    """Prepare request from candidate."""
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
