import json
import logging
import os
import typing as t
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import v2 as vision_transforms
from tqdm.autonotebook import tqdm

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


class RSScalingTransform:
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
        return f"{self.__class__.__name__}(channel_wise={self.channel_wise}, clamp={self.clamp})"


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


class RSDatasetMixin:
    def __init__(
            self,
            split: t.Literal["train", "val", "test"],
            image_size: int,
            batch_size: int,
            num_workers: int,
            device: torch.device,
    ):
        self.root = ""

        self.split = split

        self.image_size = image_size
        self.N_BANDS = 0
        self.bands: list[str] = []

        self.classes: list[str] = []
        self.N_CLASSES = 0

        self.composed_transforms = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        self.mean = torch.zeros(0).to(self.device)
        self.var = torch.zeros(0).to(self.device)
        self.std = torch.zeros(0).to(self.device)

    @property
    def mean_std_path(self) -> Path:
        band_string = "_".join(self.bands)
        dir_path = get_dataset_root() / "mean_stds"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{self.__class__.__name__}_{band_string}.npz"

    @property
    def repr_name(self) -> str:
        return f"{self.__class__.__name__}({self.split})"

    def get_original_train_dataloader(self) -> torch.utils.data.DataLoader[dict[str, Tensor]]:
        pass

    def get_mean_std(self) -> t.Tuple[Tensor, Tensor]:
        if self.mean is None or self.var is None or self.std is None:
            logger.info(f"Mean and std not yet stored in {self.__class__.__name__}.")
            try:
                with np.load(self.mean_std_path) as data:  # type: dict[str, np.ndarray]
                    logger.debug(f"Loading mean, var, and std from {self.mean_std_path}.")
                    self.mean = torch.from_numpy(data["mean"])
                    self.var = torch.from_numpy(data["var"])
                    self.std = torch.from_numpy(data["std"])
            except FileNotFoundError:
                logger.warning(f"Mean and std not found in {self.mean_std_path}. "
                               f"Calculating from dataloader instead.")
                self.calculate_channel_wise_mean_std(self.get_original_train_dataloader())

        return self.mean, self.std

    def calculate_channel_wise_mean_std(
            self,
            dataloader: torch.utils.data.DataLoader[dict[str, Tensor]],
    ):
        # Adapted from https://stackoverflow.com/a/60803379/7253717
        n_images = 0
        mean = torch.zeros(self.N_BANDS)
        var = torch.zeros(self.N_BANDS)
        with tqdm(total=len(dataloader), desc=f"Mean/std of {self.__class__.__name__}",
                  unit="batch", ncols=110, leave=False) as pbar:
            for _, batch in enumerate(dataloader):  # type: _, dict[str, torch.Tensor]
                images = batch["image"]
                b, c, h, w = images.shape
                # Rearrange batch to be the shape of [B, C, H*W]
                images = images.view(b, c, -1)
                # Update total number of images
                n_images += b
                # Compute mean and std here (over H*W and then sum over b)
                mean += images.mean(2).sum(0)
                var += images.var(2).sum(0)

                pbar.update()
                logger.debug(str(pbar))

        self.mean /= n_images
        self.var /= n_images
        self.std = torch.sqrt(var)
        logger.info(f"Calculated mean and std for {self.repr_name}) "
                    f"with {len(self.bands)} bands: mean={self.mean}, std={self.std}.")

    def get_transforms(
            self,
            scaling_transform: RSScalingTransform = None,
            normalisation: vision_transforms.Normalize = None,
            augmentations: list[vision_transforms.Transform] = None,
            use_resize: bool = True,
    ):

        # scaling handled by normalise below
        transform_list = [
            vision_transforms.ToImage(),
            vision_transforms.ToDtype(torch.float32, scale=False),
        ]

        if scaling_transform is not None:
            transform_list.append(scaling_transform)
            logger.debug(f"Using {scaling_transform} for {self.repr_name}")

        if normalisation is not None:
            transform_list.append(normalisation)
            logger.debug(f"Using {normalisation} for {self.repr_name}")

        if augmentations is not None:
            transform_list += augmentations
            logger.debug(f"Applying additional augmentations={augmentations} for {self.repr_name}")

        # Resize to image size required by input layer of model
        if use_resize:  # rescale the image to the required size via interpolation
            scaling_transform = vision_transforms.Resize(
                self.image_size, interpolation=vision_transforms.InterpolationMode.BILINEAR
            )
            logger.debug(f"Upsizing {self.repr_name} images via Resize.")
        else:  # just put the image in the middle and pad around it
            scaling_transform = vision_transforms.CenterCrop(self.image_size)
            logger.debug(f"Upsizing {self.repr_name} images via CenterCrop.")
        transform_list.append(scaling_transform)

        self.composed_transforms = vision_transforms.Compose(transform_list)  # todo: move transforms to cuda?
        logger.info(f"Built transforms for {self.repr_name}.")
        return tensor_dict_transform_wrapper(self.composed_transforms)
