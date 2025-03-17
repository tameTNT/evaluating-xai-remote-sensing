import typing as t

import torch

from helpers import log
from . import core
from . import eurosat, ucmerced, reBEN

logger = log.get_logger("main")

DATASET_NAMES = t.Literal["EuroSATRGB", "EuroSATMS", "UCMerced", "reBEN"]


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
) -> t.Union[eurosat.EuroSATBase, ucmerced.UCMerced]:
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

    if name == "EuroSATRGB":
        logger.debug("Loading EuroSATRGB dataset...")
        ds = eurosat.EuroSATRGB(**standard_kwargs, **kwargs)
    elif name == "EuroSATMS":
        logger.debug("Loading EuroSATMS dataset...")
        ds = eurosat.EuroSATMS(**standard_kwargs, **kwargs)
    elif name == "UCMerced":
        logger.debug("Loading UCMerced dataset...")
        ds = ucmerced.UCMerced(**standard_kwargs, **kwargs)
    elif name == "reBEN":
        logger.debug("Loading reBEN dataset...")
        ds = reBEN.BigEarthNetV2(**standard_kwargs, **kwargs)
    else:
        logger.error(f"Invalid dataset name ({name}) provided to get_dataset_object. "
                     f"Must be one of {t.get_args(DATASET_NAMES)}.")
        raise ValueError(f"Dataset {name} does not exist.")

    logger.info(f"Dataset {ds.logging_name} loaded with {len(ds)} samples.")
    return ds
