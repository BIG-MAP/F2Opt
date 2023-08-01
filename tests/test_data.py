import unittest

from src import data, schemas


class TestData(unittest.TestCase):

    # def test_get_dataframe_from_results(self):
    #     config = {}
    #     results = []
    #     data.get_dataframe_from_results(config, results)

    def test_get_constraints_from_method_limitations(self):
        limitations = [
            {
                "quantity": "test_quantity",
                "method": "test_method",
                "limitations": [
                    {
                        "formulation": [[
                            {
                                "chemical": {"SMILES": "test_smiles_1", "InChIKey": "test_inchi_1"},
                                # "fraction": [{"min": 0.0, "max": 1.0}],
                                "fraction": [{"min": 0.1, "max": 1.0}, {"min": 0.0, "max": 0.0}],
                                "fraction_type": "molPerMol"
                            },
                            {
                                "chemical": {"SMILES": "test_smiles_2", "InChIKey": "test_inchi_2"},
                                # "fraction": [{"min": 0.0, "max": 1.0}],
                                "fraction": [{"min": 0.1, "max": 1.0}, 0.0],
                                "fraction_type": "molPerMol"
                            },
                        ]],
                        "temperature": [{"min": 243, "max": 333}]
                    }
                ]
            }
        ]
        constraints = data.get_constraints_from_method_limitations(limitations)
        self.assertIn("test_method", constraints)
        self.assertEqual(len(constraints["test_method"]), 1)
        self.assertEqual(constraints["test_method"][0]["method"], "test_method")
        self.assertEqual(constraints["test_method"][0]["quantity"], "test_quantity")
        self.assertIn("formulation", constraints["test_method"][0])
        formula_constraints = constraints["test_method"][0]["formulation"]
        self.assertIn("chemicals", formula_constraints)
        self.assertIn("lower", formula_constraints)
        self.assertIn("upper", formula_constraints)
        self.assertIn("tolerance", formula_constraints)
        lengths = [len(formula_constraints[k]) for k in formula_constraints.keys()]
        self.assertEqual(len(set(lengths)), 1)  # all lists have the same length
        for lower, upper in zip(formula_constraints["lower"], formula_constraints["upper"]):
            self.assertGreater(upper, lower)

    def test_get_constraints_from_formulation(self):
        formulation = [
            {
                "chemical": {"SMILES": "test_smiles_1", "InChIKey": "test_inchi_1"},
                # "fraction": [{"min": 0.0, "max": 1.0}],
                "fraction": [{"min": 0.1, "max": 1.0}, {"min": 0.0, "max": 0.0}],
                "fraction_type": "molPerMol"
            },
            {
                "chemical": {"SMILES": "test_smiles_2", "InChIKey": "test_inchi_2"},
                # "fraction": [{"min": 0.0, "max": 1.0}],
                "fraction": [{"min": 0.1, "max": 1.0}, 0.0],
                "fraction_type": "molPerMol"
            },
        ]
        formula_constraints = data.get_constraints_from_formulation(formulation)
        self.assertIn("chemicals", formula_constraints)
        self.assertIn("lower", formula_constraints)
        self.assertIn("upper", formula_constraints)
        self.assertIn("tolerance", formula_constraints)
        lengths = [len(formula_constraints[k]) for k in formula_constraints.keys()]
        self.assertEqual(len(set(lengths)), 1)  # all lists have the same length
        for lower, upper in zip(formula_constraints["lower"], formula_constraints["upper"]):
            self.assertGreater(upper, lower)

    def test_get_constraints_from_fraction(self):
        fraction = [{"min": 0.0, "max": 1.0}]
        lower, upper, tolerance = data.get_constraints_from_fraction(fraction)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 1.0)
        self.assertEqual(tolerance, 0.0)

        fraction = [{"min": 0.1, "max": 0.9}]
        lower, upper, tolerance = data.get_constraints_from_fraction(fraction)
        self.assertEqual(lower, 0.1)
        self.assertEqual(upper, 0.9)
        self.assertEqual(tolerance, 0.0)

        fraction = [{"min": 0.0, "max": 0.0}, {"min": 0.0, "max": 1.0}]
        lower, upper, tolerance = data.get_constraints_from_fraction(fraction)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 1.0)
        self.assertEqual(tolerance, 0.0)

        fraction = [{"min": 0.0, "max": 0.0}, {"min": 0.1, "max": 0.9}]
        lower, upper, tolerance = data.get_constraints_from_fraction(fraction)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 0.9)
        self.assertEqual(tolerance, 0.1)

        fraction = [{"min": 0.0, "max": 1.0}, {"min": 0.0, "max": 0.0}]
        lower, upper, tolerance = data.get_constraints_from_fraction(fraction)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 1.0)
        self.assertEqual(tolerance, 0.0)

        fraction = [{"min": 0.1, "max": 0.9}, {"min": 0.0, "max": 0.0}]
        lower, upper, tolerance = data.get_constraints_from_fraction(fraction)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 0.9)
        self.assertEqual(tolerance, 0.1)

        fraction = [0.0, {"min": 0.1, "max": 0.9}]
        lower, upper, tolerance = data.get_constraints_from_fraction(fraction)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 0.9)
        self.assertEqual(tolerance, 0.1)

        fraction = [{"min": 0.0, "max": 1.0}, 0.0]
        lower, upper, tolerance = data.get_constraints_from_fraction(fraction)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 1.0)
        self.assertEqual(tolerance, 0.0)

        fraction = [{"min": 0.1, "max": 0.9}, 0.0]
        lower, upper, tolerance = data.get_constraints_from_fraction(fraction)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 0.9)
        self.assertEqual(tolerance, 0.1)

        # should fail
        # fraction = [{"min": 0.1, "max": 0.9}, {"min": 0.1, "max": 0.9}]
        # lower, upper, tolerance = data.get_constraints_from_fraction(fraction)

        # should fail
        # fraction = [{"min": 0.1, "max": 0.1}, {"min": 0.2, "max": 0.9}]
        # lower, upper, tolerance = data.get_constraints_from_fraction(fraction)

        # should fail
        # fraction = [{"min": 0.2, "max": 0.9}, {"min": 0.1, "max": 0.1}]
        # lower, upper, tolerance = data.get_constraints_from_fraction(fraction)

        # should fail
        # fraction = [{"min": 0.1, "max": 0.7}, {"min": 0.3, "max": 0.9}]
        # lower, upper, tolerance = data.get_constraints_from_fraction(fraction)

    def test_get_candidates_from_method_constraints(self):
        constraints = [
            {
                "quantity": "test_quantity",
                "method": "test_method",
                "formulation": {
                    "chemicals": [
                        {"SMILES": "test_smiles_1", "InChIKey": "test_inchi_1"},
                        {"SMILES": "test_smiles_2", "InChIKey": "test_inchi_2"}
                    ],
                    "lower": [0.1, 0.1],
                    "upper": [0.9, 0.9],
                    "tolerance": [0.0, 0.0],
                }
            },
            {
                "quantity": "test_quantity",
                "method": "test_method",
                "formulation": {
                    "chemicals": [
                        {"SMILES": "test_smiles_2", "InChIKey": "test_inchi_2"},
                        {"SMILES": "test_smiles_3", "InChIKey": "test_inchi_3"}
                    ],
                    "lower": [0.0, 0.0],
                    "upper": [1.0, 1.0],
                    "tolerance": [0.1, 0.1]
                }
            },
        ]
        num_samples = 1
        candidates = data.get_candidates_from_method_constraints(constraints, num_samples)
        self.assertEqual(len(candidates.columns), 3)
        self.assertEqual(len(candidates), len(constraints) * num_samples)
        self.assertIn("test_smiles_1", candidates.columns)
        self.assertIn("test_smiles_2", candidates.columns)
        self.assertIn("test_smiles_3", candidates.columns)

    def test_get_best_candidates(self):
        config = {
            "name": "test_optimiser",
            "targets": [{
                "name": "test_target",
                "quantities": {"test_quantity": "test_quantity"},
                "method": "test_method",
                "defaults": {"temperature": 300},
                "max_queue_size": 1,
            }]
        }
        constraints = {"test_method": [
            {
                "quantity": "test_quantity",
                "method": "test_method",
                "formulation": {
                    "chemicals": [
                        {"SMILES": "test_smiles_1", "InChIKey": "test_inchi_1"},
                        {"SMILES": "test_smiles_2", "InChIKey": "test_inchi_2"},
                    ],
                    "lower": [0.1, 0.1],
                    "upper": [0.9, 0.9],
                    "tolerance": [0.0, 0.0],
                }
            },
            {
                "quantity": "test_quantity",
                "method": "test_method",
                "formulation": {
                    "chemicals": [
                        {"SMILES": "test_smiles_2", "InChIKey": "test_inchi_2"},
                        {"SMILES": "test_smiles_3", "InChIKey": "test_inchi_3"},
                    ],
                    "lower": [0.0, 0.0],
                    "upper": [1.0, 1.0],
                    "tolerance": [0.1, 0.1]
                }
            }
        ]}
        df = None  # TODO: Add dataframe of observed data for data based approach
        candidates = data.get_best_candidates(config, df, constraints)
        self.assertIn("test_method", candidates.keys())
        candidate = candidates["test_method"]
        self.assertEqual(candidate["target"]["name"], "test_target")

    def test_get_requests_from_candidate(self):
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
            {"SMILES": "test_smiles_1", "InChIKey": "test_inchi_1"},
            {"SMILES": "test_smiles_2", "InChIKey": "test_inchi_2"},
        ]
        candidate = {
            "target": config["targets"][0],
            "chemicals": chemicals,
            "fractions": [0.5, 0.5],
        }
        requests = data.get_requests_from_candidate(config, candidate)
        self.assertEqual(len(requests), 1)
        schemas.Request(**requests[0])  # Validate request with schema

    def test_integration(self):
        config = {
            "name": "test_optimiser",
            "targets": [{
                "name": "test_target",
                "quantities": {"test_quantity": "test_quantity"},
                "method": "test_method",
                "defaults": {"temperature": 300},
                "max_queue_size": 1,
            }]
        }
        limitations = [
            {
                "quantity": "test_quantity",
                "method": "test_method",
                "limitations": [
                    {
                        "formulation": [[
                            {
                                "chemical": {"SMILES": "test_smiles_1", "InChIKey": "test_inchi_1"},
                                # "fraction": [{"min": 0.0, "max": 1.0}],
                                "fraction": [{"min": 0.1, "max": 1.0}, {"min": 0.0, "max": 0.0}],
                                "fraction_type": "molPerMol"
                            },
                            {
                                "chemical": {"SMILES": "test_smiles_2", "InChIKey": "test_inchi_2"},
                                # "fraction": [{"min": 0.0, "max": 1.0}],
                                "fraction": [{"min": 0.1, "max": 1.0}, 0.0],
                                "fraction_type": "molPerMol"
                            },
                        ]],
                        "temperature": [{"min": 243, "max": 333}]
                    }
                ]
            }
        ]
        constraints = data.get_constraints_from_method_limitations(limitations)
        self.assertIn("test_method", constraints)
        self.assertEqual(len(constraints["test_method"]), 1)
        df = None  # TODO: Add dataframe of observed data for data based approach
        candidates = data.get_best_candidates(config, df, constraints)
        self.assertIn("test_method", candidates.keys())
        candidate = candidates["test_method"]
        self.assertEqual(candidate["target"]["name"], "test_target")
        requests = data.get_requests_from_candidate(config, candidate)
        self.assertEqual(len(requests), 1)
        schemas.Request(**requests[0])  # Validate request with schema
