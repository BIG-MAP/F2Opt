import unittest

import pandas as pd

from src import data, schemas
from src.broker import MockBroker


class TestData(unittest.TestCase):

    def test_get_dataframe_from_results(self):
        config = {
            "name": "otest",
            "id": "test_id",
            "objectives": [],
            "tasks": [
                {
                    "name": "test_task_1",
                    "quantities": {
                        "test_quantity_1": "test_objective_1",
                        "test_quantity_2": "test_objective_2"
                    },
                    "method": "test_method_1",
                    "source": True,
                    "request": False,
                },
                {
                    "name": "test_task_2",
                    "quantities": {
                        "test_quantity_1": "test_objective_1",
                        "test_quantity_2": "test_objective_2"
                    },
                    "method": "test_method_2",
                    "source": True,
                    "request": False,
                }
            ]
        }
        results = {
            "test_task_1": {
                "test_quantity_1": [
                    {
                        "uuid": "test_result_id_1",
                        "ctime": "2023-08-01T12:01:00",
                        "status": "original",
                        "result": {
                            "data": {
                                "run_info": {
                                    "formulation": [
                                        {
                                            "chemical": {
                                                "SMILES": "test_smiles_1",
                                                "InChIKey": "test_inchi_1"
                                            },
                                            "fraction": 0.1,
                                            "fraction_type": "molPerMol"
                                        },
                                        {
                                            "chemical": {
                                                "SMILES": "test_smiles_2",
                                                "InChIKey": "test_inchi_2"
                                            },
                                            "fraction": 0.9,
                                            "fraction_type": "molPerMol"
                                        }
                                    ],
                                    "internal_reference": "test_internal_reference_1"
                                },
                                "test_quantity_1": {
                                    "values": [0.090, 0.091],
                                    "temperature": 298,
                                    "meta": {"success": True, "rating": 5}
                                }
                            },
                            "quantity": "test_quantity_1",
                            "method": ["test_method_1"],
                            "parameters": {},  # left out
                            "tenant_uuid": "test_tenant_id",
                            "request_uuid": "test_request_id_1",
                        }
                    },
                    {
                        "uuid": "test_result_id_2",
                        "ctime": "2023-08-01T12:02:00",
                        "status": "original",
                        "result": {
                            "data": {
                                "run_info": {
                                    "formulation": [
                                        {
                                            "chemical": {
                                                "SMILES": "test_smiles_1",
                                                "InChIKey": "test_inchi_1"
                                            },
                                            "fraction": 0.2,
                                            "fraction_type": "molPerMol"
                                        },
                                        {
                                            "chemical": {
                                                "SMILES": "test_smiles_2",
                                                "InChIKey": "test_inchi_2"
                                            },
                                            "fraction": 0.8,
                                            "fraction_type": "molPerMol"
                                        }
                                    ],
                                    "internal_reference": "test_internal_reference_1"
                                },
                                "test_quantity_1": {
                                    "values": [0.16],
                                    "temperature": 298,
                                    "meta": {"success": True, "rating": 5}
                                }
                            },
                            "quantity": "test_quantity_1",
                            "method": ["test_method_1"],
                            "parameters": {},  # left out
                            "tenant_uuid": "test_tenant_id",
                            "request_uuid": "test_request_id_2",
                        }
                    }
                ],
                # "test_quantity_2": [
                #     {
                #         "uuid": "test_result_id_3",
                #         "ctime": "2023-08-01T12:03:00",
                #         "status": "original",
                #         "result": {
                #             "data": {
                #                 "run_info": {
                #                     "formulation": [
                #                         {
                #                             "chemical": {
                #                                 "SMILES": "test_smiles_1",
                #                                 "InChIKey": "test_inchi_1"
                #                             },
                #                             "fraction": 0.1,
                #                             "fraction_type": "molPerMol"
                #                         },
                #                         {
                #                             "chemical": {
                #                                 "SMILES": "test_smiles_2",
                #                                 "InChIKey": "test_inchi_2"
                #                             },
                #                             "fraction": 0.9,
                #                             "fraction_type": "molPerMol"
                #                         }
                #                     ],
                #                     "internal_reference": "test_internal_reference_1"
                #                 },
                #                 "test_quantity_2": {
                #                     "values": [0.09],
                #                     "temperature": 298,
                #                     "meta": {"success": True, "rating": 5}
                #                 }
                #             },
                #             "quantity": "test_quantity_2",
                #             "method": ["test_method_1"],
                #             "parameters": {},  # left out
                #             "tenant_uuid": "test_tenant_id",
                #             "request_uuid": "test_request_id_1",
                #         }
                #     },
                #     {
                #         "uuid": "test_result_id_4",
                #         "ctime": "2023-08-01T12:04:00",
                #         "status": "original",
                #         "result": {
                #             "data": {
                #                 "run_info": {
                #                     "formulation": [
                #                         {
                #                             "chemical": {
                #                                 "SMILES": "test_smiles_1",
                #                                 "InChIKey": "test_inchi_1"
                #                             },
                #                             "fraction": 0.2,
                #                             "fraction_type": "molPerMol"
                #                         },
                #                         {
                #                             "chemical": {
                #                                 "SMILES": "test_smiles_2",
                #                                 "InChIKey": "test_inchi_2"
                #                             },
                #                             "fraction": 0.8,
                #                             "fraction_type": "molPerMol"
                #                         }
                #                     ],
                #                     "internal_reference": "test_internal_reference_1"
                #                 },
                #                 "test_quantity_2": {
                #                     "values": [0.16],
                #                     "temperature": 298,
                #                     "meta": {"success": True, "rating": 5}
                #                 }
                #             },
                #             "quantity": "test_quantity_2",
                #             "method": ["test_method_1"],
                #             "parameters": {},  # left out
                #             "tenant_uuid": "test_tenant_id",
                #             "request_uuid": "test_request_id_2",
                #         }
                #     }
                # ]
            },
            "test_task_2": {
                "test_quantity_1": [
                    {
                        "uuid": "test_result_id_5",
                        "ctime": "2023-08-01T12:05:00",
                        "status": "original",
                        "result": {
                            "data": {
                                "run_info": {
                                    "formulation": [
                                        {
                                            "chemical": {
                                                "SMILES": "test_smiles_1",
                                                "InChIKey": "test_inchi_1"
                                            },
                                            "fraction": 0.1,
                                            "fraction_type": "molPerMol"
                                        },
                                        {
                                            "chemical": {
                                                "SMILES": "test_smiles_2",
                                                "InChIKey": "test_inchi_2"
                                            },
                                            "fraction": 0.9,
                                            "fraction_type": "molPerMol"
                                        }
                                    ],
                                    "internal_reference": "test_internal_reference_1"
                                },
                                "test_quantity_1": {
                                    "values": [0.09],
                                    "temperature": 298,
                                    "meta": {"success": True, "rating": 5}
                                }
                            },
                            "quantity": "test_quantity_1",
                            "method": ["test_method_2"],
                            "parameters": {},  # left out
                            "tenant_uuid": "test_tenant_id",
                            "request_uuid": "test_request_id_5",
                        }
                    },
                    {
                        "uuid": "test_result_id_6",
                        "ctime": "2023-08-01T12:06:00",
                        "status": "original",
                        "result": {
                            "data": {
                                "run_info": {
                                    "formulation": [
                                        {
                                            "chemical": {
                                                "SMILES": "test_smiles_1",
                                                "InChIKey": "test_inchi_1"
                                            },
                                            "fraction": 0.2,
                                            "fraction_type": "molPerMol"
                                        },
                                        {
                                            "chemical": {
                                                "SMILES": "test_smiles_2",
                                                "InChIKey": "test_inchi_2"
                                            },
                                            "fraction": 0.8,
                                            "fraction_type": "molPerMol"
                                        }
                                    ],
                                    "internal_reference": "test_internal_reference_1"
                                },
                                "test_quantity_1": {
                                    "values": [0.16],
                                    "temperature": 298,
                                    "meta": {"success": True, "rating": 5}
                                }
                            },
                            "quantity": "test_quantity_1",
                            "method": ["test_method_2"],
                            "parameters": {},  # left out
                            "tenant_uuid": "test_tenant_id",
                            "request_uuid": "test_request_id_6",
                        }
                    }
                ],
                # "test_quantity_2": [...]
            }
        }
        df = data.get_dataframe_from_results(config, results)
        self.assertEqual(len(df), 5)
        self.assertIn("test_smiles_1", df.columns)
        self.assertIn("test_smiles_2", df.columns)
        self.assertIn("test_quantity_1", df.columns)
        self.assertIn("test_objective_1", df.columns)
        # import pandas as pd
        # pd.set_option('display.max_columns', None)
        # assert False, f"\n{str(df)}"

    def test_get_dataframe_from_results_eol(self):
        config = {
            "name": "otest",
            "id": "test_id",
            "objectives": [],
            "tasks": [
                {
                    "name": "test_task_1",
                    "quantities": {
                        "degradationEOL": "EOL"
                    },
                    "method": "degradation_workflow",
                    "source": True,
                    "request": True,
                }
            ]
        }
        results = {
            "test_task_1": {
                "degradationEOL": [
                    {
                        "uuid": "test_result_id_1",
                        "ctime": "2023-08-01T12:01:00",
                        "status": "original",
                        "result": {
                            "data": {
                                "run_info": {
                                    "formulation": [
                                        {
                                            "chemical": {
                                                "SMILES": "test_smiles_1",
                                                "InChIKey": "test_inchi_1"
                                            },
                                            "fraction": 1.0,
                                            "fraction_type": "molPerMol"
                                        }
                                    ],
                                    "internal_reference": "An ID assigned by the tenant",
                                    "formulation_info": {},  # left out
                                    "chemicals_info": {},  # left out
                                },
                                "degradationEOL": [
                                    {
                                        "end_of_life": 400,
                                        "end_of_life_uncertainty": 5.9,
                                        "capacity_trajectory": [3.2, 2.9, 2.6, 2.4, 2.0],
                                        "capacity_trajectory_uncertainty": [
                                            0.04, 0.009, 0.02, 0.0005, 0.01
                                        ],
                                        "meta": {"success": True, "rating": 5},
                                        "cell_info": {}  # left out
                                    },
                                    {
                                        "end_of_life": 200,
                                        "end_of_life_uncertainty": 2.9,
                                        "capacity_trajectory": [2.2, 2.9, 2.6, 2.4, 2.0],
                                        "capacity_trajectory_uncertainty": [
                                            0.02, 0.004, 0.04, 0.0002, 0.04
                                        ],
                                        "meta": {"success": True, "rating": 5},
                                        "cell_info": {}  # left out
                                    }
                                ]
                            },
                            "quantity": "degradationEOL",
                            "method": ["degradation_workflow"],
                            "parameters": {},
                            "tenant_uuid": "7337771802bd49d7ae8437d6bff2bfc2",
                            "request_uuid": "e20ef4f7-179a-4715-acaa-e700386cd58e"
                        }
                    }
                ]
            }
        }
        df = data.get_dataframe_from_results(config, results)
        self.assertEqual(len(df), 2)
        self.assertIn("test_smiles_1", df.columns)
        self.assertIn("degradationEOL", df.columns)
        self.assertIn("EOL", df.columns)
        # import pandas as pd
        # pd.set_option('display.max_columns', None)
        # assert False, f"\n{str(df)}"

    def test_get_rows_from_result_default_format(self):
        result = {
            "uuid": "test_result_id_1",
            "ctime": "2023-08-01T12:01:00",
            "status": "original",
            "result": {
                "data": {
                    "run_info": {
                        "formulation": [
                            {
                                "chemical": {
                                    "SMILES": "test_smiles_1",
                                    "InChIKey": "test_inchi_1"
                                },
                                "fraction": 0.1,
                                "fraction_type": "molPerMol"
                            },
                            {
                                "chemical": {
                                    "SMILES": "test_smiles_2",
                                    "InChIKey": "test_inchi_2"
                                },
                                "fraction": 0.9,
                                "fraction_type": "molPerMol"
                            }
                        ],
                        "internal_reference": "test_internal_reference_1"
                    },
                    "test_quantity_1": {
                        "values": [0.090, 0.091],
                        "temperature": 298,
                        "meta": {"success": True, "rating": 5}
                    }
                },
                "quantity": "test_quantity_1",
                "method": ["test_method_1"],
                "parameters": {},  # left out
                "tenant_uuid": "test_tenant_id",
                "request_uuid": "test_request_id_1",
            }
        }
        rows = data.get_rows_from_result(result)
        self.assertEqual(len(rows), 2)
        # import json
        # assert False, json.dumps(rows, indent=2)

    def test_get_rows_from_result_eol_format(self):
        result = {
            "uuid": "test_result_id_1",
            "ctime": "2023-08-01T12:01:00",
            "status": "original",
            "result": {
                "data": {
                    "run_info": {
                        "formulation": [
                            {
                                "chemical": {
                                    "SMILES": "CCOC(=O)OC",
                                    "InChIKey": "JBTWLSYIZRCDFO-UHFFFAOYSA-N"
                                },
                                "fraction": 1.0,
                                "fraction_type": "molPerMol"
                            }
                        ],
                        "internal_reference": "An ID assigned by the tenant",
                        "formulation_info": {},  # left out
                        "chemicals_info": {},  # left out
                    },
                    "degradationEOL": [
                        {
                            "end_of_life": 400,
                            "end_of_life_uncertainty": 5.9,
                            "capacity_trajectory": [3.2, 2.9, 2.6, 2.4, 2.0],
                            "capacity_trajectory_uncertainty": [0.04, 0.009, 0.02, 0.0005, 0.01],
                            "meta": {"success": True, "rating": 5},
                            "cell_info": {}  # left out
                        },
                        {
                            "end_of_life": 200,
                            "end_of_life_uncertainty": 2.9,
                            "capacity_trajectory": [2.2, 2.9, 2.6, 2.4, 2.0],
                            "capacity_trajectory_uncertainty": [0.02, 0.004, 0.04, 0.0002, 0.04],
                            "meta": {"success": True, "rating": 5},
                            "cell_info": {}  # left out
                        }
                    ]
                },
                "quantity": "degradationEOL",
                "method": ["degradation_workflow"],
                "parameters": {},
                "tenant_uuid": "7337771802bd49d7ae8437d6bff2bfc2",
                "request_uuid": "e20ef4f7-179a-4715-acaa-e700386cd58e"
            }
        }
        rows = data.get_rows_from_result(result)
        self.assertEqual(len(rows), 2)
        # import json
        # assert False, json.dumps(rows, indent=2)

    def test_get_constraints_from_limitations(self):
        limitations = {
            "test_task": {
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
        }
        constraints = data.get_constraints_from_limitations(limitations)
        self.assertIn("test_task", constraints)
        self.assertEqual(len(constraints["test_task"]), 1)
        self.assertEqual(constraints["test_task"][0]["method"], "test_method")
        self.assertEqual(constraints["test_task"][0]["quantity"], "test_quantity")
        self.assertIn("formulation", constraints["test_task"][0])
        formula_constraints = constraints["test_task"][0]["formulation"]
        self.assertIn("chemicals", formula_constraints)
        self.assertIn("lower", formula_constraints)
        self.assertIn("upper", formula_constraints)
        self.assertIn("tolerance", formula_constraints)
        lengths = [len(formula_constraints[k]) for k in formula_constraints.keys()]
        self.assertEqual(len(set(lengths)), 1)  # all lists have the same length
        for lower, upper in zip(formula_constraints["lower"], formula_constraints["upper"]):
            self.assertGreater(upper, lower)

    def test_get_constraints_from_limitations_eol(self):
        limitations = {
            "test_task": {
                "quantity": "test_quantity",
                "method": "test_method",
                "limitations": [{"battery_chemistry": {
                    "electrolyte": [[
                        {
                            "chemical": {"SMILES": "test_smiles_1", "InChIKey": "test_inchi_1"},
                            "fraction": [{"min": 0.0, "max": 1.0}],
                            "fraction_type": "molPerMol"
                        },
                        {
                            "chemical": {"SMILES": "test_smiles_2", "InChIKey": "test_inchi_2"},
                            "fraction": [{"min": 0.0, "max": 1.0}],
                            "fraction_type": "molPerMol"
                        },
                    ]],
                }}]
            }
        }
        constraints = data.get_constraints_from_limitations(limitations)
        self.assertIn("test_task", constraints)
        self.assertEqual(len(constraints["test_task"]), 1)
        self.assertEqual(constraints["test_task"][0]["method"], "test_method")
        self.assertEqual(constraints["test_task"][0]["quantity"], "test_quantity")
        self.assertIn("formulation", constraints["test_task"][0])
        formula_constraints = constraints["test_task"][0]["formulation"]
        self.assertIn("chemicals", formula_constraints)
        self.assertIn("lower", formula_constraints)
        self.assertIn("upper", formula_constraints)
        self.assertIn("tolerance", formula_constraints)
        lengths = [len(formula_constraints[k]) for k in formula_constraints.keys()]
        self.assertEqual(len(set(lengths)), 1)  # all lists have the same length
        for lower, upper in zip(formula_constraints["lower"], formula_constraints["upper"]):
            self.assertGreater(upper, lower)

    def test_get_constraints_from_formulation_limitations(self):
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
        formula_constraints = data.get_constraints_from_formulation_limitations(formulation)
        self.assertIn("chemicals", formula_constraints)
        self.assertIn("lower", formula_constraints)
        self.assertIn("upper", formula_constraints)
        self.assertIn("tolerance", formula_constraints)
        lengths = [len(formula_constraints[k]) for k in formula_constraints.keys()]
        self.assertEqual(len(set(lengths)), 1)  # all lists have the same length
        for lower, upper in zip(formula_constraints["lower"], formula_constraints["upper"]):
            self.assertGreater(upper, lower)

    def test_get_constraints_from_fraction_limitations(self):
        fraction = [{"min": 0.0, "max": 1.0}]
        lower, upper, tolerance = data.get_constraints_from_fraction_limitations(fraction)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 1.0)
        self.assertEqual(tolerance, 0.0)

        fraction = [{"min": 0.1, "max": 0.9}]
        lower, upper, tolerance = data.get_constraints_from_fraction_limitations(fraction)
        self.assertEqual(lower, 0.1)
        self.assertEqual(upper, 0.9)
        self.assertEqual(tolerance, 0.0)

        fraction = [{"min": 0.0, "max": 0.0}, {"min": 0.0, "max": 1.0}]
        lower, upper, tolerance = data.get_constraints_from_fraction_limitations(fraction)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 1.0)
        self.assertEqual(tolerance, 0.0)

        fraction = [{"min": 0.0, "max": 0.0}, {"min": 0.1, "max": 0.9}]
        lower, upper, tolerance = data.get_constraints_from_fraction_limitations(fraction)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 0.9)
        self.assertEqual(tolerance, 0.1)

        fraction = [{"min": 0.0, "max": 1.0}, {"min": 0.0, "max": 0.0}]
        lower, upper, tolerance = data.get_constraints_from_fraction_limitations(fraction)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 1.0)
        self.assertEqual(tolerance, 0.0)

        fraction = [{"min": 0.1, "max": 0.9}, {"min": 0.0, "max": 0.0}]
        lower, upper, tolerance = data.get_constraints_from_fraction_limitations(fraction)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 0.9)
        self.assertEqual(tolerance, 0.1)

        fraction = [0.0, {"min": 0.1, "max": 0.9}]
        lower, upper, tolerance = data.get_constraints_from_fraction_limitations(fraction)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 0.9)
        self.assertEqual(tolerance, 0.1)

        fraction = [{"min": 0.0, "max": 1.0}, 0.0]
        lower, upper, tolerance = data.get_constraints_from_fraction_limitations(fraction)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 1.0)
        self.assertEqual(tolerance, 0.0)

        fraction = [{"min": 0.1, "max": 0.9}, 0.0]
        lower, upper, tolerance = data.get_constraints_from_fraction_limitations(fraction)
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

    def test_get_candidates_from_constraints(self):
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
        candidates = data.get_candidates_from_constraints(constraints, num_samples)
        self.assertEqual(len(candidates.columns), 3)
        self.assertEqual(len(candidates), len(constraints) * num_samples)
        self.assertIn("test_smiles_1", candidates.columns)
        self.assertIn("test_smiles_2", candidates.columns)
        self.assertIn("test_smiles_3", candidates.columns)

    def test_get_best_candidates_wo_data(self):
        config = {
            "name": "otest",
            "id": "test_id",
            "objectives": [{
                "quantity": "test_objective",
                "maximize": True
            }],
            "tasks": [{
                "name": "test_task",
                "quantities": {"test_quantity": "test_objective"},
                "method": "test_method",
                "source": True,
                "request": True,
                "parameters": {},
                "max_queue_size": 1,
                "num_candidates": 100,
                "min_results_for_ml": 1,
            }]
        }
        constraints = {"test_task": [
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
        df = pd.DataFrame()  # Empty dataframe
        candidates = data.get_best_candidates(config, df, constraints)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate["task"]["name"], "test_task")

    def test_get_best_candidates_w_data(self):
        config = {
            "name": "otest",
            "id": "test_id",
            "objectives": [{
                "quantity": "test_objective",
                "maximize": True
            }],
            "tasks": [{
                "name": "test_task",
                "quantities": {"test_quantity": "test_objective"},
                "method": "test_method",
                "source": True,
                "request": True,
                "parameters": {},
                "max_queue_size": 1,
                "num_candidates": 100,
                "min_results_for_ml": 1,
                "num_training_steps": 100,
            }]
        }
        constraints = {"test_task": [
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
        df = pd.DataFrame([
            {
                "result_id": "test_result_id_1",
                "request_id": "test_request_id_1",
                "task_name": "test_task",
                "quantity": "test_quantity",
                "method": "test_method",
                "test_smiles_1": 0.1,
                "test_smiles_2": 0.2,
                "test_smiles_3": 0.7,
                "test_objective": 0.014,
            },
            {
                "result_id": "test_result_id_2",
                "request_id": "test_request_id_2",
                "task_name": "test_task",
                "quantity": "test_quantity",
                "method": "test_method",
                "test_smiles_1": 0.2,
                "test_smiles_2": 0.3,
                "test_smiles_3": 0.5,
                "test_objective": 0.03,
            },
        ])
        candidates = data.get_best_candidates(config, df, constraints)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate["task"]["name"], "test_task")

    def test_get_requests_from_candidate(self):
        config = {
            "name": "otest",
            "id": "test_id",
            "objectives": [],
            "tasks": [{
                "name": "test_task",
                "quantities": {"test_quantity": "test_objective"},
                "method": "test_method",
                "source": True,
                "request": True,
                "parameters": {},
                "max_queue_size": 1
            }]
        }
        chemicals = [
            {"SMILES": "test_smiles_1", "InChIKey": "test_inchi_1"},
            {"SMILES": "test_smiles_2", "InChIKey": "test_inchi_2"},
        ]
        candidate = {
            "task": config["tasks"][0],
            "chemicals": chemicals,
            "fractions": [0.5, 0.5],
        }
        requests = data.get_requests_from_candidate(config, candidate)
        self.assertEqual(len(requests), 1)
        schemas.Request(**requests[0])  # Validate request with schema

    def test_integration(self):
        config = {
            "name": "otest",
            "id": "test_id",
            "objectives": [{
                "quantity": "test_objective",
                "maximize": True
            }],
            "tasks": [{
                "name": "test_task",
                "quantities": {"test_quantity": "test_objective"},
                "method": "test_method",
                "source": True,
                "request": True,
                "parameters": {},
                "max_queue_size": 1,
                "num_candidates": 100,
                "min_results_for_ml": 1,
            }]
        }
        limitations = {
            "test_task": {
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
        }
        constraints = data.get_constraints_from_limitations(limitations)
        self.assertIn("test_task", constraints)
        self.assertEqual(len(constraints["test_task"]), 1)
        df = pd.DataFrame()  # TODO: Add dataframe of observed data for data based approach
        candidates = data.get_best_candidates(config, df, constraints)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate["task"]["name"], "test_task")
        requests = data.get_requests_from_candidate(config, candidate)
        self.assertEqual(len(requests), 1)
        schemas.Request(**requests[0])  # Validate request with schema

    def test_integration_with_mock_broker(self):
        config = {
            "name": "omock",
            "id": "mock_id",
            "objectives": [{
                "quantity": "mock_objective",
                "maximize": True
            }],
            "tasks": [{
                "name": "mock_task",
                "quantities": {"mock_quantity": "mock_objective"},
                "method": "mock_method_1",
                "source": True,
                "request": True,
                "parameters": {},
                "max_queue_size": 1,
                "num_candidates": 100,
                "min_results_for_ml": 1,
                "num_training_steps": 100,
            }]
        }
        broker = MockBroker(authenticated=True, compute_results=False)
        raw_limitations = broker.get_limitations()
        self.assertEqual(len(raw_limitations), 2)
        self.assertEqual(len(raw_limitations[0]["limitations"]), 1)
        self.assertEqual(len(raw_limitations[1]["limitations"]), 1)
        self.assertEqual(raw_limitations[0]["quantity"], "mock_quantity")
        self.assertEqual(raw_limitations[0]["method"], "mock_method_1")
        self.assertEqual(raw_limitations[1]["quantity"], "mock_quantity")
        self.assertEqual(raw_limitations[1]["method"], "mock_method_2")
        limitations = {config["tasks"][0]["name"]: raw_limitations[0]}
        constraints = data.get_constraints_from_limitations(limitations)
        self.assertIn("mock_task", constraints)
        self.assertEqual(len(constraints["mock_task"]), 1)
        results = broker.get_results(quantity="mock_quantity", method="mock_method_1")
        results = {"mock_task": {"mock_quantity": results}}
        df = data.get_dataframe_from_results(config, results)
        self.assertFalse(df.empty)
        candidates = data.get_best_candidates(config, df, constraints)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate["task"]["name"], "mock_task")
        requests = data.get_requests_from_candidate(config, candidate)
        self.assertEqual(len(requests), 1)
        schemas.Request(**requests[0])  # Validate request with schema
