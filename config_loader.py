"""
Module Name: config_loader

Description:
    Unified configuration loader that loads YAML config files and returns
    configuration dictionaries with computed paths. Expects command line arguments
    that describe the config filename relative to the project root.

Usage:
"""
import os
import sys
import yaml


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config():
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_file = sys.argv[idx + 1]
    else:
        config_file = "config.yaml"
    
    config_path = os.path.join(ROOT_DIR, config_file)
    
    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add computed absolute paths
    config['ROOT_DIR'] = ROOT_DIR
    config['RAW_DATA_DIRECTORY'] = os.path.join(ROOT_DIR, config['RAW_DATA_DIRECTORY'])
    config['PROCESSED_DATA_DIRECTORY'] = os.path.join(ROOT_DIR, config['PROCESSED_DATA_DIRECTORY'])
    
    # Add background directories if they exist in config
    if 'RAW_DATA_BACKGROUND_DIRECTORY' in config:
        config['RAW_DATA_BACKGROUND_DIRECTORY'] = os.path.join(ROOT_DIR, config['RAW_DATA_BACKGROUND_DIRECTORY'])
    if 'PROCESSED_DATA_BACKGROUND_DIRECTORY' in config:
        config['PROCESSED_DATA_BACKGROUND_DIRECTORY'] = os.path.join(ROOT_DIR, config['PROCESSED_DATA_BACKGROUND_DIRECTORY'])
    
    return config
