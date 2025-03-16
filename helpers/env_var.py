import os
import platform
from pathlib import Path


def get_project_root() -> Path:
    """
    Get the project root directory
    """
    project_root = os.getenv("SAT_PROJECT_ROOT", "~/l3_project")
    path = Path(project_root).expanduser().resolve()
    if path.is_dir():
        # logger.debug(f"Project root variable set and is an existing valid directory: {path}.")
        return path
    else:
        raise FileNotFoundError(f"The expected project root, {path}, doesn't exist.")


def get_xai_output_root() -> Path:
    """
    Return the output directory to use for XAI results since there appears to
    be a difference in pixel values depending on the platform.

    Loading the windows images on Linux/macOS gives a difference of 0.0111 on
    eurosat for instance for exactly the same input eurosat image loaded
    from disk.
    """

    if platform.system() == "Windows":
        return get_project_root() / "xai_output_windows"
    else:
        return get_project_root() / "xai_output"


def get_dataset_root() -> Path:
    """
    Get the dataset root directory from the environment variable "DATASET_ROOT".
    Uses ~/datasets as default if not set.
    """

    dataset_root = os.getenv("DATASET_ROOT", "~/datasets")
    path = Path(dataset_root).expanduser().resolve()
    if path.is_dir():
        # logger.debug(f"Dataset root variable set and is an existing valid directory: {path}")
        return path
    else:
        raise FileNotFoundError(f"The dataset root was set successfully but doesn't actually exist: {path}")
