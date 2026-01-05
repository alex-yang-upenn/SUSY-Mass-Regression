"""Unified configuration loader for YAML config files.

This module provides functionality to load YAML configuration files and return
configuration dictionaries with computed absolute paths. It supports command-line
arguments to specify different config files.
"""

import os
import sys

import yaml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config():
    """Load configuration from YAML file specified via command line.

    Parses command-line arguments for --config flag to determine which
    configuration file to load. Defaults to 'config.yaml' if not specified.
    Computes absolute paths for all directory configurations.

    Args:
        None. Reads from sys.argv for --config flag.

    Returns:
        dict: Configuration dictionary with the following structure:
            - All original YAML key-value pairs
            - ROOT_DIR: Absolute path to project root
            - RAW_DATA_DIRECTORY: Absolute path to raw data directory
            - PROCESSED_DATA_DIRECTORY: Absolute path to processed data directory
            - RAW_DATA_BACKGROUND_DIRECTORY: Absolute path to raw background data (if exists)
            - PROCESSED_DATA_BACKGROUND_DIRECTORY: Absolute path to processed background data (if exists)

    Raises:
        FileNotFoundError: If specified config file doesn't exist.
        yaml.YAMLError: If config file is not valid YAML.

    Example:
        >>> config = load_config()  # Loads config.yaml
        >>> config = load_config()  # With --config set2, loads config_set2.yaml
    """
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_file = sys.argv[idx + 1]
    else:
        config_file = "config.yaml"

    config_path = os.path.join(ROOT_DIR, config_file)

    # Load YAML config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Add computed absolute paths
    config["ROOT_DIR"] = ROOT_DIR
    config["RAW_DATA_DIRECTORY"] = os.path.join(ROOT_DIR, config["RAW_DATA_DIRECTORY"])
    config["PROCESSED_DATA_DIRECTORY"] = os.path.join(
        ROOT_DIR, config["PROCESSED_DATA_DIRECTORY"]
    )

    # Add background directories if they exist in config
    if "RAW_DATA_BACKGROUND_DIRECTORY" in config:
        config["RAW_DATA_BACKGROUND_DIRECTORY"] = os.path.join(
            ROOT_DIR, config["RAW_DATA_BACKGROUND_DIRECTORY"]
        )
    if "PROCESSED_DATA_BACKGROUND_DIRECTORY" in config:
        config["PROCESSED_DATA_BACKGROUND_DIRECTORY"] = os.path.join(
            ROOT_DIR, config["PROCESSED_DATA_BACKGROUND_DIRECTORY"]
        )

    return config
