import typing as t

import torch

from helpers import log

from . import core
from .eurosat import EuroSATRGB, EuroSATMS
from .ucmerced import UCMerced
from .reben import ReBEN
from .patternnet import PatternNet

logger = log.main_logger

DATASET_NAMES = t.Literal["EuroSATRGB", "EuroSATMS", "UCMerced", "ReBEN", "PatternNet"]


# noinspection PyIncorrectDocstring
def get_dataset_object(
        name: DATASET_NAMES,
        split: t.Literal["train", "val", "test"],
        image_size: int,
        batch_size: int,
        num_workers: int,
        device: torch.device,
        use_augmentations: bool = True,
        use_resize: bool = True,
        download: bool = False,
        **kwargs
) -> t.Union[EuroSATRGB, EuroSATMS, UCMerced, ReBEN, PatternNet]:
    """
    todo: write this docstring
    :param name: The name of the dataset to load. One of the dataset names in DATASET_NAMES.
    :param kwargs: Passed to the dataset class. See dataset_processing.core.RSDatasetMixin
      for undocumented parameters and the specific dataset class for the kwargs unique to that dataset.
    :return: A dataset object of the specified type.
    """

    standard_kwargs = {
        "split": split,
        "image_size": image_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": device,
        "use_augmentations": use_augmentations,
        "use_resize": use_resize,
        "download": download,
    }

    logger.debug(f"Attempting to load {name} dataset...")
    assert name in t.get_args(DATASET_NAMES), (f"Invalid dataset name ({name}) provided to get_dataset_object. "
                                               f"Must be one of {t.get_args(DATASET_NAMES)}.")
    ds = globals()[name](**standard_kwargs, **kwargs)
    logger.info(f"Dataset {ds.logging_name} loaded with {len(ds)} samples.")
    return ds
