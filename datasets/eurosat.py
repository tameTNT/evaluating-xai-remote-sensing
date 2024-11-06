import logging
import typing as t

import torch
import torchgeo.datasets

import datasets.core

DATASET_ROOT = datasets.core.get_dataset_root()
logger = logging.getLogger("projectLog")


def get_dataset(
        split: t.Literal["train", "test", "val"],
        bands: t.Sequence[str] = ('B04', 'B03', 'B02'),  # defaults to RGB bands
        transforms: t.Optional[t.Callable[[t.Dict[str, torch.Tensor]], t.Dict[str, torch.Tensor]]] = None,
        download=False,
) -> torchgeo.datasets.EuroSAT:
    """
    Get the EuroSAT dataset using torchgeo (can download if necessary).

    Defaults to returning only the RGB bands ('B04', 'B03', 'B02'). But supports all 13 bands.
    """
    logger.debug("Using `torchgeo.datasets` to load EuroSAT dataset.")
    return torchgeo.datasets.EuroSAT(
        root=str(DATASET_ROOT / "eurosat"),
        split=split,
        bands=bands,
        transforms=transforms,
        download=download
    )
