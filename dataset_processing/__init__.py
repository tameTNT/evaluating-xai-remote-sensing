import typing as t

from helpers import logging
from . import core
from . import eurosat

lg = logging.get_logger("main")

DATASET_NAMES = ["EuroSATRGB", "EuroSATMS"]


def get_dataset_object(
        name: t.Literal["EuroSATRGB", "EuroSATMS"],
        split: t.Literal["train", "val", "test"],
        image_size: int,
        **kwargs
) -> t.Union[eurosat.EuroSATBase]:
    """

    :param name: The name of the dataset to load. One of the dataset names in DATASET_NAMES.
    :param split: One of "train", "val" or "test" indicating the dataset split.
    :param image_size: The size images of the dataset should be resized to for the model being used.
    :param kwargs: Passed to the dataset class.
    :return: A dataset object of the specified type.
    """
    standard_kwargs = {
        "split": split,
        "image_size": image_size,
    }

    if name == "EuroSATRGB":
        lg.debug("Loading EuroSATRGB dataset...")
        ds = eurosat.EuroSATRGB(**standard_kwargs, **kwargs)
    elif name == "EuroSATMS":
        lg.debug("Loading EuroSATMS dataset...")
        ds = eurosat.EuroSATMS(**standard_kwargs, **kwargs)
    else:
        lg.error(f"Invalid dataset name ({name}) provided to get_dataset_object. "
                 f"Must be one of {DATASET_NAMES}.")
        raise ValueError(f"Dataset {name} does not exist.")

    lg.info(f"Dataset {name} ({split}) loaded with {len(ds)} samples.")
    return ds
