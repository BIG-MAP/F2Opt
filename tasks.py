"""Invoke tasks."""

import logging
import json
from collections import defaultdict

from invoke import task


# Setup logging
logging.basicConfig(level=logging.INFO)


# Broker server

BROKER_URL = "http://localhost:13371"
BROKER_USERNAME = ""
BROKER_PASSWORD = ""
BROKER_SHADOW_MODE = False


def get_broker():
    from src.broker import Broker
    return Broker(
        url=BROKER_URL,
        username=BROKER_USERNAME,
        password=BROKER_PASSWORD,
        shadow_mode=BROKER_SHADOW_MODE,
    )


@task
def ping(c):
    broker = get_broker()
    broker.ping()


@task
def authenticate(c):
    broker = get_broker()
    broker.authenticate()


@task
def get_capabilities(c):
    broker = get_broker()
    broker.authenticate()
    capabilities = broker.get_capabilities()
    print(json.dumps(capabilities, indent=2))


@task
def get_limitations(c):
    broker = get_broker()
    broker.authenticate()
    limitations = broker.get_limitations()
    print(json.dumps(limitations, indent=2))


@task
def get_results(c, quantity=None, method=None):
    broker = get_broker()
    broker.authenticate()
    results = broker.get_results(quantity=quantity, method=method)
    print(json.dumps(results, indent=2))


@task
def get_dataframe(c, quantity=None, method=None, file_name=None, filter=False):
    broker = get_broker()
    broker.authenticate()
    results = broker.get_results(quantity=quantity, method=method)

    results_dict = defaultdict(dict)
    for result in results:
        quantity = result["result"]["quantity"]
        method = result["result"]["method"][0]
        task_name = f"{quantity}-{method}"
        if quantity not in results_dict[task_name]:
            results_dict[task_name][quantity] = []
        results_dict[task_name][quantity].append(result)

    config = {
        "tasks": [{
            "name": task_name,
            "quantities": {q: "y" for q in quantity_dict.keys()},
            "source": True,
        } for task_name, quantity_dict in results_dict.items()]
    }

    from src.data import get_dataframe_from_results
    df = get_dataframe_from_results(config, results_dict, filter=filter)
    import pandas as pd
    pd.set_option('display.max_columns', None)
    print(df)
    if file_name:
        df.to_csv(file_name)


@task
def get_result(c, result_id):
    broker = get_broker()
    broker.authenticate()
    result = broker.get_result(result_id)
    print(json.dumps(result, indent=2))


@task
def post_result(c, request_id):
    broker = get_broker()
    broker.authenticate()
    result = {}
    # result = {
    #     "data": {"density(method1)": {"type": "number", "value": 33}},
    #     "quantity": "DummyQuantity",
    #     "method": ["DummyMethod"],
    #     "parameters": {
    #         "DummyMethod": {
    #             "internal_temperature": {"value": 42, "type": "number", "description": ""},
    #             "voltage_setting": {"value": 2, "type": "number", "description": ""}
    #         }
    #     },
    #     "tenant_uuid": "DummyOptimizer",
    #     "request_uuid": request_id  # Only works with an valid request id
    # }
    result_id = broker.post_result(result)
    print(result_id)


@task
def get_pending_requests(c, quantity=None, method=None):
    broker = get_broker()
    broker.authenticate()
    requests = broker.get_pending_requests(quantity=quantity, method=method)
    print(json.dumps(requests, indent=2))


@task
def get_request(c, request_id):
    broker = get_broker()
    broker.authenticate()
    request = broker.get_request(request_id)
    print(json.dumps(request, indent=2))


@task
def post_request(c):
    broker = get_broker()
    broker.authenticate()
    request = {}
    # request = {
    #     "quantity": "DummyQuantity",
    #     "methods": ["DummyMethod"],
    #     "parameters": {"DummyMethod": {"internal_temperature": 1}},
    #     "tenant_uuid": "DummyOptimizer"
    # }
    # request = {
    #     'quantity': 'conductivity',
    #     'methods': ['two_electrode'],
    #     'parameters': {'two_electrode': {
    #         'formulation': [
    #             {'chemical': {'SMILES': '[Li+].F[P-](F)(F)(F)(F)F', 'InChIKey': 'AXPLOJNSKRXQPA-UHFFFAOYSA-N'}, 'fraction': 0.0128, 'fraction_type': 'molPerMol'},
    #             {'chemical': {'SMILES': 'COC(=O)OC', 'InChIKey': 'IEJIGPNLZYLLBP-UHFFFAOYSA-N'}, 'fraction': 0.8526, 'fraction_type': 'molPerMol'},
    #             {'chemical': {'SMILES': 'CCOC(=O)OC', 'InChIKey': 'JBTWLSYIZRCDFO-UHFFFAOYSA-N'}, 'fraction': 0.0895, 'fraction_type': 'molPerMol'},
    #             {'chemical': {'SMILES': 'C1COC(=O)O1', 'InChIKey': 'KMTRUDSVKNLOMY-UHFFFAOYSA-N'}, 'fraction': 0.0451, 'fraction_type': 'molPerMol'},
    #         ],
    #         'temperature': 298.15,
    #     }},
    #     'tenant_uuid': 'f3f7d376-3b58-4d25-adb4-0f3994f215ce',
    # }
    request_id = broker.post_request(request)
    print(request_id)
