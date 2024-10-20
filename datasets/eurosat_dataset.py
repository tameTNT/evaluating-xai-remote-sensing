import torchgeo.datasets
import datasets.dataset_core

DATASET_ROOT = datasets.dataset_core.get_dataset_root()

def get_eurosat_dataset(download=False) -> torchgeo.datasets.EuroSAT:
    """
    Get the EuroSAT dataset.
    """
    return torchgeo.datasets.EuroSAT(
        root=str(DATASET_ROOT / "eurosat"), split="train", transforms=None, download=download
    )