import os
import platform
from pathlib import Path


def get_project_root() -> Path:
    """
    Get the project root directory from the environment variable SAT_PROJECT_ROOT.
    Defaults to ~/l3_project if not set.
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
    Return the output directory to use for XAI results (explanations, metric evaluations, etc.).

    Different directories are used for Windows and Unix systems since Resize leads to very slightly different
    output images/files on Windows compared to Unix systems, so equality comparisons used in xai Explainers fail.
    See _platform_behaviour_investigation/resize_eurosat.py and resize_by_platform.py for more details.
    """

    if platform.system() == "Windows":
        return get_project_root() / "xai_output_windows"
    else:
        return get_project_root() / "xai_output_unix"


def get_dataset_root() -> Path:
    """
    Get the dataset root directory from the environment variable DATASET_ROOT.
    Defaults to ~/datasets if not set.
    """

    dataset_root = os.getenv("DATASET_ROOT", "~/datasets")
    path = Path(dataset_root).expanduser().resolve()
    if path.is_dir():
        # logger.debug(f"Dataset root variable set and is an existing valid directory: {path}")
        return path
    else:
        raise FileNotFoundError(f"The dataset root was set successfully but doesn't actually exist: {path}")
