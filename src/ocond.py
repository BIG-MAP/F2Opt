"""Main file for the conductivity optimizer."""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import requests

from broker import Broker


def load_configuration_from_file(config_path):
    """Load configuration from file."""
    with open(config_path) as config_file:
        config = json.load(config_file)
    return config


def setup_broker(broker_url, broker_username, broker_password, shadow_mode):
    """Setup broker."""
    broker = Broker(broker_url, broker_username, broker_password, shadow_mode)
    # broker.authenticate()  # TODO
    try:
        broker.ping()
    except requests.exceptions.ConnectionError:
        logger.error("Broker ping failed. Exiting.")
        raise SystemExit


def get_queue(config, broker):
    """Get queued requests from broker."""
    raise NotImplementedError


def get_capabilities(config, broker):
    """Get capabilities from broker."""
    raise NotImplementedError


def get_results(config, broker):
    """Get results from broker."""
    raise NotImplementedError


def get_dataframe(results):
    """Create dataframe from results."""
    raise NotImplementedError


def get_best_candidates(config, capabilities, df):
    """Get best candidates for all targets and quantities."""
    # TODO: Produce random candidates within capabilities for now
    raise NotImplementedError


def optimisation_step(config, broker):
    """Run one optimisation step."""
    logger.info("Optimisation step.")
    # TODO: Implement main functionality
    # Check queue status
    # queue = get_queue(config, broker)
    # If at least one request target has room in the queue
    # Get capabilities
    # Get results from broker
    # results = get_results(config, broker)
    # Build dataframe
    # df = get_dataframe(results)
    # Get candidates
    # candidates = get_best_candidates(config, capabilities, df)
    # Prepare requests
    # Send requests


def main(config_path, broker_url, broker_username, broker_password,
         loop_delay=-1, shadow_mode=False):
    """Main entry point."""
    # Setup
    logger.info("Setup.")
    config = load_configuration_from_file(config_path)
    broker = setup_broker(broker_url, broker_username, broker_password, shadow_mode)
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
    parser.add_argument("--broker_url", required=True, help="Broker URL.")
    parser.add_argument("--broker_username", required=True, help="Broker username.")
    parser.add_argument("--broker_password", required=True, help="Broker password.")
    parser.add_argument("--loop", type=int, default=-1,
                        help="Loop delay in seconds. If negative, run only once.")
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
        shadow_mode=args.shadow,
    )
