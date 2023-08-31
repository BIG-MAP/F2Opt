"""Mock Broker client."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path


logger = logging.getLogger("mock broker")


class MockBroker:
    """Mock broker class for imitating interactions with the broker server."""

    def __init__(self, authenticated=False, compute_results=False, save_results=False):
        logger.info("Init mock broker")
        self.auth_header = "authenticated" if authenticated else None
        self.compute_results = compute_results
        self.save_results = save_results
        self.output_path = Path(__file__).parent / "../../tmp/"  # path for output files
        self.timestamp = datetime.now().isoformat()  # timestamp for output files
        self.capabilities = [
            {
                "quantity": "mock_quantity",
                "method": "mock_method",
                # "json_schema_specifications": {},  # schema left out
                # "json_schema_result_output": {}  # schema left out
            }
        ]
        self.limitations = [
            {
                "quantity": "mock_quantity",
                "method": "mock_method",
                "limitations": [
                    {
                        "formulation": [[
                            {
                                "chemical": {"SMILES": "mock_smiles_1", "InChIKey": "mock_inchi_1"},
                                "fraction": [{"min": 0.0, "max": 1.0}, {"min": 0.0, "max": 0.0}],
                                "fraction_type": "molPerMol"
                            },
                            {
                                "chemical": {"SMILES": "mock_smiles_2", "InChIKey": "mock_inchi_2"},
                                "fraction": [{"min": 0.0, "max": 1.0}, 0.0],
                                "fraction_type": "molPerMol"
                            },
                        ]],
                        "temperature": [{"min": 243, "max": 333}]
                    }
                ]
            }
        ]
        if compute_results:
            self.results = []
            self.requests = []
        else:
            self.results = [
                {
                    "uuid": "mock_result_id_1",
                    "ctime": "2023-08-01T12:00:00",
                    "status": "original",
                    "result": {
                        "data": {
                            "run_info": {
                                "formulation": [
                                    {
                                        "chemical": {
                                            "SMILES": "mock_smiles_1", "InChIKey": "mock_inchi_1"
                                        },
                                        "fraction": 0.1,
                                        "fraction_type": "molPerMol"
                                    },
                                    {
                                        "chemical": {
                                            "SMILES": "mock_smiles_2", "InChIKey": "mock_inchi_2"
                                        },
                                        "fraction": 0.9,
                                        "fraction_type": "molPerMol"
                                    }
                                ],
                                "internal_reference": "mock_internal_reference_1"
                            },
                            "mock_quantity": {
                                "values": [0.09],
                                "temperature": 298,
                                "meta": {"success": True, "rating": 1}
                            }
                        },
                        "quantity": "mock_quantity",
                        "method": ["mock_method"],
                        "parameters": {
                            "mock_method": {
                                "formulation": [
                                    {
                                        "chemical": {
                                            "SMILES": "mock_smiles_1", "InChIKey": "mock_inchi_1"
                                        },
                                        "fraction": 0.1,
                                        "fraction_type": "molPerMol"
                                    },
                                    {
                                        "chemical": {
                                            "SMILES": "mock_smiles_2", "InChIKey": "mock_inchi_2"
                                        },
                                        "fraction": 0.9,
                                        "fraction_type": "molPerMol"
                                    }
                                ],
                                "temperature": 298
                            }
                        },
                        "tenant_uuid": "mock_tenant_id",
                        "request_uuid": "mock_request_id_1",
                    }
                },
                {
                    "uuid": "mock_result_id_2",
                    "ctime": "2023-08-01T12:00:00",
                    "status": "original",
                    "result": {
                        "data": {
                            "run_info": {
                                "formulation": [
                                    {
                                        "chemical": {
                                            "SMILES": "mock_smiles_1", "InChIKey": "mock_inchi_1"
                                        },
                                        "fraction": 0.2,
                                        "fraction_type": "molPerMol"
                                    },
                                    {
                                        "chemical": {
                                            "SMILES": "mock_smiles_2", "InChIKey": "mock_inchi_2"
                                        },
                                        "fraction": 0.8,
                                        "fraction_type": "molPerMol"
                                    }
                                ],
                                "internal_reference": "mock_internal_reference_1"
                            },
                            "mock_quantity": {
                                "values": [0.16],
                                "temperature": 298,
                                "meta": {"success": True, "rating": 1}
                            }
                        },
                        "quantity": "mock_quantity",
                        "method": ["mock_method"],
                        "parameters": {
                            "mock_method": {
                                "formulation": [
                                    {
                                        "chemical": {
                                            "SMILES": "mock_smiles_1", "InChIKey": "mock_inchi_1"
                                        },
                                        "fraction": 0.2,
                                        "fraction_type": "molPerMol"
                                    },
                                    {
                                        "chemical": {
                                            "SMILES": "mock_smiles_2", "InChIKey": "mock_inchi_2"
                                        },
                                        "fraction": 0.8,
                                        "fraction_type": "molPerMol"
                                    }
                                ],
                                "temperature": 298
                            }
                        },
                        "tenant_uuid": "mock_tenant_id",
                        "request_uuid": "mock_request_id_2",
                    }
                }
            ]
        self.requests = [
                {
                    "quantity": "mock_quantity",
                    "methods": ["mock_method"],
                    "parameters": {
                        "mock_method": {
                            "formulation": [
                                {
                                    "chemical": {
                                        "SMILES": "mock_smiles_1", "InChIKey": "mock_inchi_1"
                                    },
                                    "fraction": 0.1,
                                    "fraction_type": "molPerMol"
                                },
                                {
                                    "chemical": {
                                        "SMILES": "mock_smiles_2", "InChIKey": "mock_inchi_2"
                                    },
                                    "fraction": 0.9,
                                    "fraction_type": "molPerMol"
                                },
                            ],
                        }
                    },
                    "tenant_uuid": "mock_requester_id",
                    "uuid": "mock_request_id_3"
                }
            ]

    def ping(self):
        """Ping the broker server."""
        logger.info("Ping")
        return True

    def authenticate(self):
        """Authenticate."""
        logger.info("Authenticate")
        self.auth_header = "authenticated"
        return True

    def get_capabilities(self, available=True):
        """Get capabilities."""
        assert self.auth_header is not None, "Not authenticated."
        logger.info("Get capabilities")
        return self.capabilities

    def get_limitations(self, available=True):
        """Get limitations."""
        assert self.auth_header is not None, "Not authenticated."
        logger.info("Get limitations")
        return self.limitations

    def get_results(self, quantity=None, method=None):
        """Get results."""
        assert self.auth_header is not None, "Not authenticated."
        logger.info("Get results")
        # Filter results by quantity and method
        filtered_results = []
        for result in self.results:
            if result["result"]["quantity"] == quantity and method in result["result"]["method"]:
                filtered_results.append(result)
        return filtered_results

    def post_result(self, result, verbose=True):
        """Post result."""
        assert self.auth_header is not None, "Not authenticated."
        if verbose:
            logger.info("Post result")
        # Add result uuid
        result["uuid"] = str(uuid.uuid4())
        # Remove request with matching request id
        self.requests = [r for r in self.requests if r["uuid"] != result["result"]["request_uuid"]]
        # Add result to results
        self.results.append(result)
        # Save results to file
        if self.save_results:
            with open(self.output_path / f"mock_results_{self.timestamp}.json", "w") as f:
                json.dump(self.results, f, indent=4)
        return result["uuid"]

    def get_pending_requests(self, quantity=None, method=None):
        """Get pending requests."""
        assert self.auth_header is not None, "Not authenticated."
        logger.info("Get pending requests")
        # Filter requests by quantity and method
        filtered_requests = []
        for request in self.requests:
            if request["quantity"] == quantity and method in request["methods"]:
                filtered_requests.append(request)
        return filtered_requests

    def post_request(self, request):
        """Post request."""
        assert self.auth_header is not None, "Not authenticated."
        logger.info("Post request")
        # Add request_uuid to request
        request["uuid"] = str(uuid.uuid4())
        self.requests.append(request)
        if self.compute_results:
            result = self._compute_result_from_request(request)
            self.post_result(result, verbose=False)
        return request["uuid"]

    def _compute_result_from_request(self, request):
        """Compute result."""
        # Compute result value as product of chemical fractions
        result_value = 1.0
        for chemical in request["parameters"]["mock_method"]["formulation"]:
            result_value *= chemical["fraction"]
        # Prepare and return result
        return {
            "ctime": datetime.now().isoformat(),
            "status": "original",
            "result": {
                "data": {
                    "run_info": {
                        "formulation": request["parameters"]["mock_method"]["formulation"],
                        "internal_reference": "mock_internal_reference_1"
                    },
                    "mock_quantity": {
                        "values": [result_value],
                        "temperature": 298,
                        "meta": {"success": True, "rating": 1}
                    }
                },
                "quantity": "mock_quantity",
                "method": ["mock_method"],
                "parameters": {
                    "mock_method": {
                        "formulation": request["parameters"]["mock_method"]["formulation"],
                        "temperature": 298
                    }
                },
                "tenant_uuid": "mock_tenant_id",
                "request_uuid": request["uuid"],
            }
        }
