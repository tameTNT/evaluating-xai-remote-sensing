import logging
import typing as t

import torch
import torchgeo.datasets
from jaxtyping import Float

import datasets.core

DATASET_ROOT = datasets.core.get_dataset_root()
logger = logging.getLogger("projectLog")


# Overwrite the `__getitem__` method to cast the image to 0-1 floats
# instead of integer floats (e.g. 125., 9.) for .plot() to work
class CustomUCMerced(torchgeo.datasets.UCMerced):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> t.Dict[str, Float[torch.Tensor, "c h w"]]:
        logger.debug(f"Loading UC Merced dataset sample at index {index} using {self.__class__} class.")
        sample = super().__getitem__(index)
        sample["image"] = sample["image"] / 255
        return sample


def get_dataset(
        split: t.Literal["train", "test", "val"],
        transforms: t.Optional[t.Callable[[t.Dict[str, torch.Tensor]], t.Dict[str, torch.Tensor]]] = None,
        download=False
) -> CustomUCMerced:
    """
    Get the UC Merced dataset using torchgeo (can download if necessary).
    """
    logger.info("Using `torchgeo.datasets` to load UC Merced dataset dataset.")
    return CustomUCMerced(
        root=str(DATASET_ROOT / "ucmerced"),
        split=split,
        transforms=transforms,
        download=download
    )
