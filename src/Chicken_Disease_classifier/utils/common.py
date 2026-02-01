import os
import box.exceptions
import yaml
import Chicken_Disease_classifier
import json
import joblib
from ensure import ensure_annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.Chicken_Disease_classifier.constants import * 

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> Dict[str, Any]:
    """
    Reads a YAML file and returns its content as a dictionary.
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return content
    except Exception as e:
        raise e

@ensure_annotations
def save_json(path_to_json: Path, content: Dict[str, Any]) -> None:
    """
    Saves a dictionary to a JSON file.
    """
    try:
        with open(path_to_json, 'w') as json_file:
            json.dump(content, json_file, indent=4)
            logging.info(f"json file: {path_to_json} saved successfully")
    except Exception as e:
        raise e


@ensure_annotations
def load_bin(path_to_bin: Path) -> Any:
    """
    Loads a binary file and returns its content.
    """
    try:
        with open(path_to_bin, 'rb') as bin_file:
            content = joblib.load(bin_file)
            logging.info(f"binary file: {path_to_bin} loaded successfully")
            return content
    except Exception as e:
        raise e

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns the size of a file in a human-readable format.
    """
    try:
        size_in_bytes = path.stat().st_size
        logging.info(f"file size: {size_in_bytes} bytes")
        return size_in_bytes
    except Exception as e:
        raise e
