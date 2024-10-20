from pathlib import Path
import json

def get_dataset_root() -> Path:
    """
    Get the dataset root directory from the local env.json file.
    Handles the case where the file is not found or the key is missing.
    """
    env_file = Path.cwd() / "env.json"
    try:
        env_settings = json.load(env_file.open("r"))
        dataset_root = Path(env_settings["dataset_root"])
    except FileNotFoundError:
        raise FileNotFoundError(f"Please create an env.json file (looking for {env_file.resolve()}).")
    except KeyError:
        raise KeyError(f"Please ensure that {env_file.resolve()} contains a `dataset_root` key.")

    return dataset_root