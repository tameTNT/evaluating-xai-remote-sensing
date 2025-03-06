import json
import logging
import os
import typing as t
from pathlib import Path

import torch
from torch import Tensor

DATASET_NAMES = ["EuroSATRGB", "EuroSATMS"]


logger = logging.getLogger("main")


def get_dataset_root() -> Path:
    """
    Get the dataset root directory from the local env.json file.
    Handles the case where the file is not found or the key is missing.
    """
    dataset_root = os.getenv("DATASET_ROOT")  # first try to load from environment variables
    if dataset_root is None:  # try to load from env.json instead
        logger.debug("No DATASET_ROOT environment variable found. Looking for env.json instead.")
        env_file = Path.cwd() / "env.json"
        try:
            env_settings = json.load(env_file.open("r"))
            dataset_root = env_settings["dataset_root"]
        except FileNotFoundError:
            raise FileNotFoundError(f"Please create an env.json file (looking for {env_file.resolve()}).")
        except KeyError:
            raise KeyError(f"Please ensure that {env_file.resolve()} contains a `dataset_root` key.")
    else:
        logger.debug("DATASET_ROOT environment variable found.")
    path = Path(dataset_root)
    if path.is_dir():
        logger.info("Dataset root variable set and is an existing valid directory. Returning Path object.")
        return Path(dataset_root)
    else:
        raise FileNotFoundError(f"The dataset root was set successfully but doesn't actually exist: "
                                f"{path.resolve()} is not a directory.")


class RSNormaliseTransform:
    """
    Transformation the input tensor to the float range [0, 1] using the provided min and max values.
    If None, the min and max values are calculated from the input tensor's min/max directly,
    optionally using provided percentiles rather than the pure min/max.
    """

    def __init__(
            self,
            input_min: t.Optional[float] = None,
            # For EuroSAT, authors claim range is 0-2750. Other torchgeo datasets use 3000.
            input_max: t.Optional[float] = None,
            percentiles: t.Optional[t.Tuple[float, float]] = (.02, .98),
            channel_wise: bool = False,
            clamp: bool = False,
    ):
        self.min = input_min
        self.max = input_max
        self.percentiles = percentiles
        self.channel_wise = channel_wise
        self.clamp = clamp

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        c, h, w = image.shape

        # flatten final dimensions to allow for quantile/max/min per channel
        image = image.reshape(c, -1)

        scale_kwargs = {"dim": 1, "keepdim": True} if self.channel_wise else {}
        if self.max is None:
            # See article for further reasoning behind percentile normalization:
            # https://medium.com/sentinel-hub/how-to-normalize-satellite-images-for-deep-learning-d5b668c885af
            if self.percentiles:
                self.max = image.quantile(self.percentiles[1], **scale_kwargs)  # use (98th) percentile as max
            else:
                self.max = image.amax(**scale_kwargs)  # a variant is consistent in return type
        if self.min is None:
            if self.percentiles:
                self.min = image.quantile(self.percentiles[0], **scale_kwargs)  # use (2nd) percentile as min
            else:
                self.min = image.amin(**scale_kwargs)

        image = (image - self.min) / (self.max - self.min)
        if self.clamp:
            image = image.clamp(0, 1)
        return image.reshape(c, h, w)  # reshape back to original dimensions

    def __repr__(self):
        return f"{__name__}.{self.__class__.__name__}(min={self.min}, max={self.max})"


def tensor_dict_transform_wrapper(
        transform: t.Callable[[torch.Tensor], torch.Tensor]
) -> t.Callable[[dict[str, Tensor]], dict[str, Tensor]]:
    """
    Applies the provided transform function (e.g. torchvision.transforms.Compose) to the "image" key of a dictionary.
    The "label" and any other key-value pairs in the dictionary are left untouched.

    :param transform: A function that takes a torch.Tensor and returns a transformed torch.Tensor.
    :return: A function that takes a dictionary with an "image" key and
        returns a dictionary with the "image" key transformed.
    """

    def wrapper(dataset_dict: t.Dict[str, torch.Tensor]) -> t.Dict[str, torch.Tensor]:
        """
        Apply the transform function provided to outer wrapper function to the "image" key of dataset_dict.
        :param dataset_dict: A dictionary which includes an "image" key with a torch.Tensor value.
        :return: The original dictionary with the "image" key transformed.
        """

        # logger.debug(f"Applying transform to 'image' key of `dataset_dict` "
        #              f"via `{tensor_dict_transform_wrapper.__name__}`'s `wrapper`.")
        transformed_image = transform(dataset_dict["image"])
        return_dict = dataset_dict.copy()
        return_dict["image"] = transformed_image
        return return_dict

    return wrapper


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
