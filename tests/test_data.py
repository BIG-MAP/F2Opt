import unittest

from src import data, schemas


class TestData(unittest.TestCase):

    # def test_get_dataframe_from_results(self):
    #     config = {}
    #     results = []
    #     data.get_dataframe_from_results(config, results)

    def test_get_constraints_from_limitations(self):
        config = {
            "targets": [{
                "name": "test_target",
                "quantities": {"test_quantity": "test_quantity"},
                "method": "test_method",
                "defaults": {},
                "max_queue_size": 1
            }]
        }
        limitations = [
            {
                "quantity": "test_quantity",
                "method": "test_method",
                "limitations": {
                    "formulation": [
                        {
                            "chemical": {
                                "SMILES": "test_smiles_1",
                                "InChIKey": "test_inchi_1"
                            },
                            "fraction": {"min": 0.1, "max": 0.9},
                            "fraction_type": ["molPerMol"]
                        },
                        {
                            "chemical": {
                                "SMILES": "test_smiles_2",
                                "InChIKey": "test_inchi_2"
                            },
                            "fraction": {"min": 0.1, "max": 0.9},
                            "fraction_type": ["molPerMol"]
                        },
                    ]
                }
            }
        ]
        constraints = data.get_constraints_from_limitations(config, limitations)
        self.assertIn("test_method", constraints)
        self.assertIn("formulation", constraints["test_method"])
        formula_constraints = constraints["test_method"]["formulation"]
        self.assertIn("chemicals", formula_constraints)
        self.assertIn("lower", formula_constraints)
        self.assertIn("upper", formula_constraints)
        self.assertIn("tolerance", formula_constraints)
        lengths = [len(formula_constraints[k]) for k in formula_constraints.keys()]
        self.assertEqual(len(set(lengths)), 1)  # all lists have the same length
        for lower, upper in zip(formula_constraints["lower"], formula_constraints["upper"]):
            self.assertGreater(upper, lower)

    def test_get_best_candidates(self):
        config = {
            "name": "opt",
            "targets": [{
                "name": "test_target",
                "quantities": {"test_quantity": "test_quantity"},
                "method": "test_method",
                "defaults": {"temperature": 300},
                "max_queue_size": 1,
            }]
        }
        df = None
        chemicals = [
            {
                "SMILES": "test_smiles_1",
                "InChIKey": "test_inchi_1",
            },
            {
                "SMILES": "test_smiles_2",
                "InChIKey": "test_inchi_2",
            },
        ]
        constraints = {"test_method": {"formulation": {
            "chemicals": chemicals,
            "lower": [0.1, 0.1],
            "upper": [0.9, 0.9],
            "tolerance": [0.0, 0.0],
        }}}
        candidates = data.get_best_candidates(config, df, constraints)
        self.assertIn("test_target", candidates)
        candidate = candidates["test_target"]
        self.assertEqual(candidate["target"]["name"], "test_target")

    def test_prepare_request(self):
        config = {
            "name": "opt",
            "targets": [{
                "name": "test_target",
                "quantities": {"test_quantity": "test_quantity"},
                "method": "test_method",
                "defaults": {"temperature": 300},
                "max_queue_size": 1,
            }]
        }
        chemicals = [
            {
                "SMILES": "test_smiles_1",
                "InChIKey": "test_inchi_1",
            },
            {
                "SMILES": "test_smiles_2",
                "InChIKey": "test_inchi_2",
            },
        ]
        candidate = {
            "target": config["targets"][0],
            "chemicals": chemicals,
            "fractions": [0.5, 0.5],
        }
        requests = data.get_requests_from_candidate(config, candidate)
        self.assertEqual(len(requests), 1)
        schemas.Request(**requests[0])  # Validate request with schema
