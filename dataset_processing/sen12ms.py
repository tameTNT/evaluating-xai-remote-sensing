import typing as t

import torch
import torchgeo.datasets

import helpers

DATASET_ROOT = helpers.env_var.get_dataset_root()
logger = helpers.log.get_logger("main")


def get_dataset(
        split: t.Literal["train", "test", "val"],
        bands: t.Sequence[str] = ('B04', 'B03', 'B02'),  # defaults to RGB bands
        transforms: t.Optional[t.Callable[[t.Dict[str, torch.Tensor]], t.Dict[str, torch.Tensor]]] = None,
) -> torchgeo.datasets.SEN12MS:
    """
    Get the SEN12MS dataset using torchgeo.

    Needs to be downloaded separately first (you can use bash script in `/dataset_processing/scripts/download_sen12ms.sh`).
    """

    logger.debug("Using `torchgeo.datasets` to load EuroSAT dataset.")
    return torchgeo.datasets.SEN12MS(
        root=str(DATASET_ROOT / "sen12ms"),
        split=split,
        bands=bands,
        transforms=None,
    )
