"""Main file for the conductivity optimizer."""

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from requests.exceptions import ConnectionError

from .broker import Broker, MockBroker
from .data import (get_best_candidates, get_constraints_from_limitations,
                   get_dataframe_from_results, get_requests_from_candidate)


def load_configuration_from_file(config_path):
    """Load configuration from file."""
    with open(config_path) as config_file:
        config = json.load(config_file)
    return config


def setup_broker(broker_url, broker_username, broker_password, mock, shadow_mode):
    """Setup broker."""
    if mock:
        assert not shadow_mode, "Shadow mode not supported with mock broker."
        broker = MockBroker(authenticated=False, compute_results=True, save_results=True)
    else:
        assert broker_url and broker_username and broker_password, \
            "Broker credentials not provided."
        broker = Broker(broker_url, broker_username, broker_password, shadow_mode)
    broker.authenticate()
    try:
        broker.ping()
    except ConnectionError:
        logger.error("Broker ping failed. Exiting.")
        raise SystemExit
    return broker


def check_capabilities(config, broker):
    """Check tasks are in capabilities."""
    capabilities = broker.get_capabilities()
    for task in config["tasks"]:
        for quantity in task["quantities"].keys():
            for capability in capabilities:
                if capability["quantity"] == quantity and capability["method"] == task["method"]:
                    break
            else:
                assert False, f"Capability not found for task {task} in {capabilities}."
    return True


def get_queue(config, broker):
    """Get queue of pending requests for each task from broker."""
    queue = defaultdict(dict)  # {task_name: {quantity: pending_request}}
    for task in [t for t in config["tasks"] if t["request"]]:
        for quantity in task["quantities"].keys():
            queue[task["name"]][quantity] = broker.get_pending_requests(
                quantity=quantity, method=task["method"]
            )
    return queue


def get_results(config, broker):
    """Get results for each task from broker."""
    results_dict = defaultdict(dict)  # {task_name: {quantity: results}}
    for task in [t for t in config["tasks"] if t["source"]]:
        for quantity in task["quantities"].keys():
            results = broker.get_results(
                quantity=quantity,
                method=task["method"],
            )
            results_dict[task["name"]][quantity] = results
    return results_dict


def get_limitations(config, broker):
    """Get limitations for each task from the broker."""
    limitations = broker.get_limitations()

    def limitations_generator(config, limitations):
        for task in [t for t in config["tasks"] if t["request"]]:
            method = task["method"]
            for quantity in task["quantities"].keys():
                # Find matching limitation for task
                for limitation in limitations:
                    if limitation["quantity"] == quantity and limitation["method"] == method:
                        break  # match found
                else:
                    raise ValueError(f"Limitation not found for task {task['name']}.")
                yield task["name"], limitation

    return {task_name: limitation
            for task_name, limitation in limitations_generator(config, limitations)}


def optimisation_step(config, broker):
    """Run one optimisation step."""
    logger.info("Start optimisation step.")
    # Check tasks are in capabilities
    assert check_capabilities(config, broker)
    # Check queue status
    queue = get_queue(config, broker)
    # If at least one task has room in the queue
    # TODO: If there is no room in queue, no reason to continue
    # Get results from broker
    results = get_results(config, broker)
    # Build dataframe from results
    df = get_dataframe_from_results(config, results)
    # Get limitations from broker
    limitations = get_limitations(config, broker)
    # Get constraints from limitations
    constraints = get_constraints_from_limitations(limitations)
    assert len(constraints) == len([t for t in config["tasks"] if t["request"]])
    assert all(task["name"] in constraints for task in [t for t in config["tasks"] if t["request"]])
    # Get candidates for each task
    logger.info("Get best candidates.")
    candidates = get_best_candidates(config, df, constraints)
    # Prepare requests
    for candidate in candidates:
        task = candidate["task"]  # task configuation
        # Check queue size. If there are multiple quantities, select max.
        queue_size = max(len(queue[task["name"]][q]) for q in task["quantities"].keys())
        if queue_size < task["max_queue_size"]:
            logger.info(f"Post request to task: {task['name']}")
            requests = get_requests_from_candidate(config, candidate)
            # Post requests
            for request in requests:
                broker.post_request(request)
        else:
            logger.info(f"Queue is full for task: {task['name']}")
    logger.info("End optimisation step.")


def main(config_path, broker_url, broker_username, broker_password,
         loop_delay=-1, mock=False, shadow_mode=False):
    """Main entry point."""
    # Setup
    logger.info("Setup.")
    config = load_configuration_from_file(config_path)
    broker = setup_broker(broker_url, broker_username, broker_password, mock, shadow_mode)
    # Main loop
    logger.info("Main loop.")
    while True:
        optimisation_step(config, broker)
        if loop_delay > 0:
            time.sleep(loop_delay)
        else:
            break
    logger.info("Job's done!")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Task configuration file.")
    parser.add_argument("--broker_url", default="", help="Broker URL.")
    parser.add_argument("--broker_username", default="", help="Broker username.")
    parser.add_argument("--broker_password", default="", help="Broker password.")
    parser.add_argument("--loop", type=int, default=-1,
                        help="Loop delay in seconds. If negative, run only once.")
    parser.add_argument("--mock", action='store_true',
                        help="Use mock broker for testing.")
    parser.add_argument("--shadow", action='store_true',
                        help="Disable post reqests to the broker server.")
    parser.add_argument("--log_dir", default=Path(__file__).parent.parent.resolve() / "logs",
                        help="Logging directory.")
    parser.add_argument("--log_level", default="info", help="Logging level.")

    args = parser.parse_args()
    # Setup logging
    log_file_name = datetime.now().strftime("%Y-%m-%d") + ".log"
    log_file_handler = logging.FileHandler(Path(args.log_dir) / log_file_name, mode="a")
    logging.basicConfig(
        level=logging.getLevelName(args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            log_file_handler,
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("main")
    # Run the main function
    main(
        config_path=args.config,
        broker_url=args.broker_url,
        broker_username=args.broker_username,
        broker_password=args.broker_password,
        loop_delay=args.loop,
        mock=args.mock,
        shadow_mode=args.shadow,
    )
