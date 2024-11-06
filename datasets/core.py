import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("projectLog")


def get_dataset_root() -> Path:
    """
    Get the dataset root directory from the local env.json file.
    Handles the case where the file is not found or the key is missing.
    """
    dataset_root = os.getenv("DATASET_ROOT")  # first try to load from environment variables
    if dataset_root is None:  # try to load from env.json instead
        logger.debug("No DATASET_ROOT environment variable found. Looking for env.json instead.")
        env_file = Path.cwd() / "env.json"
        try:
            env_settings = json.load(env_file.open("r"))
            dataset_root = env_settings["dataset_root"]
        except FileNotFoundError:
            raise FileNotFoundError(f"Please create an env.json file (looking for {env_file.resolve()}).")
        except KeyError:
            raise KeyError(f"Please ensure that {env_file.resolve()} contains a `dataset_root` key.")
    else:
        logger.debug("DATASET_ROOT environment variable found.")
    path = Path(dataset_root)
    if path.is_dir():
        logger.debug("Dataset root variable set and is an existing valid directory. Returning Path object.")
        return Path(dataset_root)
    else:
        raise FileNotFoundError(f"The dataset root was set successfully but doesn't actually exist: {path.resolve()} is not a directory.")