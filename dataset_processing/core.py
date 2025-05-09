import typing as t
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import v2 as vision_transforms
from tqdm.autonotebook import tqdm
from jaxtyping import Float

import helpers

logger = helpers.log.main_logger


class RSScalingTransform:
    """
    Transformation the input tensor to the float range [0, 1] using the provided min and max values.
    If None (the default), the min and max values are calculated from the input tensor's min/max directly,
    using provided percentiles (defaulting to 2% and 98%) rather than the pure min/max.

    Can be applied channel-wise (default False). Outputs are *not* clamped to [0, 1] by default.
    """

    def __init__(
            self,
            input_min: t.Optional[float] = None,
            # For EuroSAT, authors claim range is 0-2750. Other torchgeo datasets use 3000.
            input_max: t.Optional[float] = None,
            percentiles: t.Optional[tuple[float, float]] = (.02, .98),
            channel_wise: bool = False,
            clamp: bool = False,
    ):
        self.min = input_min
        self.max = input_max
        self.percentiles = percentiles
        self.channel_wise = channel_wise
        self.clamp = clamp

    def __call__(self, image: Tensor) -> Tensor:
        c, h, w = image.shape

        # flatten final dimensions to allow for quantile/max/min per channel
        image = image.reshape(c, -1)

        scale_kwargs = {"dim": 1, "keepdim": True} if self.channel_wise else {}
        if self.max is None:
            # See the following article for further reasoning behind percentile normalisation:
            # https://medium.com/sentinel-hub/how-to-normalize-satellite-images-for-deep-learning-d5b668c885af
            if self.percentiles:
                self.max = image.quantile(self.percentiles[1], **scale_kwargs)  # use (98th) percentile as max
            else:
                self.max = image.amax(**scale_kwargs)  # amax is consistent in return type regardless of kwargs
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


class ChoiceRotationTransform:
    """Rotate input randomly by one of the given θ in angles."""

    def __init__(self, angles: list[float]):
        self.angles = angles

    def __call__(self, image: Tensor) -> Tensor:
        choice_idx = torch.randint(0, len(self.angles), (1,)).item()
        angle = self.angles[choice_idx]

        image = vision_transforms.functional.rotate(image, angle)

        return image

    def __repr__(self):
        return f"{self.__class__.__name__}(angles={self.angles})"


class TensorDictTransformWrapper:
    """
    Applies the transform function/module provided to the "image" key (with
    Tensor value) of the dictionary passed to the class when called
    (see __call__ method).
    """

    def __init__(self, transform: t.Callable[[Tensor], Tensor]) -> None:
        self.transform = transform

    def __call__(
            self,
            dataset_dict: dict[t.Literal["image", "label"], Tensor]
    ) -> dict[t.Literal["image", "label"], Tensor]:
        dataset_dict["image"] = self.transform(dataset_dict["image"])
        return dataset_dict


def cycle(iterable: t.Iterable) -> t.Generator:
    """Yield from the iterable indefinitely."""
    while True:
        for x in iterable:
            yield x


BASIC_ROTATION_AUGMENTATIONS = [
    vision_transforms.RandomApply([  # Transpose of img
        ChoiceRotationTransform([90]),
        vision_transforms.RandomHorizontalFlip(p=1)
    ], p=0.5),
    ChoiceRotationTransform([0, 90, 180, 270]),
    vision_transforms.RandomRotation(7),
]


class RSDatasetMixin:
    # Band 4 = Red, Band 3 = Green, Band 2 = Blue
    rgb_bands = ("B04", "B03", "B02")

    def __init__(
            self,
            split: t.Literal["train", "val", "test"],
            image_size: int,
            use_augmentations: bool,
            use_resize: bool,
            batch_size: int,
            num_workers: int,
            device: torch.device,
            download: bool,
            **kwargs,
    ):
        """
        :param split: Which dataset split to use. One of "train", "val", or "test".
        :param image_size: The size images should be scaled to before being passed to the model.
        :param batch_size: The batch size to use for the dataloader should any dataset metrics need to be calculated.
        :param num_workers: The num_workers to use for the dataloader should any dataset metrics need to be calculated.
        :param device: The torch device to use for the dataloader should any dataset metrics need to be calculated.
        :param use_augmentations: Whether to apply random augmentations to the data. N/A if split != "train".
        :param use_resize: Whether to use torchvision.transforms.Resize to scale images.
            If False, torchvision.transforms.CenterCrop is used instead, placing images in the centre with padding.
        :param download: Whether to download the dataset if it is not already present in DATASET_ROOT.

        :param kwargs: Any spare keyword arguments at this point are caught and a warning is logged. They are not used.
        """

        assert split in ("train", "val", "test"), f"Invalid split '{split}'. Must be one of 'train', 'val', or 'test'."
        self.split = split

        self.N_BANDS = 0
        self.bands: list[str] = []  # *ordered* sequence of bands corresponding to sample channels
        self.classes: list[str] = []
        self.N_CLASSES = 0

        self.image_size = image_size
        self.use_augmentations = use_augmentations
        self.use_resize = use_resize
        self.composed_transforms = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.download = download

        self.mean = torch.zeros(0).to(self.device)
        self.var = torch.zeros(0).to(self.device)
        self.std = torch.zeros(0).to(self.device)

        if len(kwargs) > 0:
            logger.warning(f"Unused kwargs passed to {self.__class__.__name__}: {kwargs}")

    @property
    def mean_std_path(self) -> Path:
        """The path to the npz file containing the pre-computed mean and std (per channel) values for the dataset."""
        band_string = "_".join(self.bands)
        dir_path = helpers.env_var.get_dataset_root() / "mean_stds"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{self.__class__.__name__}_{band_string}.npz"

    @property
    def logging_name(self) -> str:
        return f"{self.__class__.__name__}({self.split})"

    def get_original_train_dataloader(self, shuffle=False) -> torch.utils.data.DataLoader[dict[str, Tensor]]:
        raise NotImplementedError(f"get_original_train_dataloader not implemented "
                                  f"in base class {self.__class__.__name__}.")

    @property
    def rgb_indices(self) -> list[int]:
        """The indices of the RGB bands in self.bands."""
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError(f"{self.__class__.__name__} doesn't contain some of the RGB bands.")

        return rgb_indices

    def inverse_transform(
            self,
            x: Float[torch.Tensor, "n_samples channels height width"]
    ) -> Float[np.ndarray, "n_samples height width channels"]:
        """
        Takes a set of normalised (x ~ N(0, 1)) samples from the dataset and
        returns an RGB image in numpy format (n, h, w, c).
        """

        images: np.ndarray = np.take(x.numpy(force=True), indices=self.rgb_indices, axis=1)

        # normalised images are in the range [-1, 1] originally
        if self.mean.nonzero().numel() > 0:  # either via explicit mean/std
            # futuretodo: this doesn't work for multichannel images properly - at least not EuroSATMS
            b, c, h, w = images.shape
            assert c == 3, f"Expected 3 channels only when inverting for display, got {c} instead."

            # was originally normalised via imges = (images - mean) / std
            mean = self.mean.cpu()[self.rgb_indices].view(1, c, 1, 1)  # make broadcastable
            std = self.std.cpu()[self.rgb_indices].view(1, c, 1, 1)

            # turn to range [0, max]
            images: torch.Tensor = torch.from_numpy(images) * std + mean
            images: torch.Tensor = images.reshape(b, c, -1)  # flatten for normalisation per channel
            upper_quantile = images.quantile(0.98, dim=-1, keepdim=True)  # use the 98th percentile as maximum
            upper_quantile = torch.where(upper_quantile == 0., 1., upper_quantile)  # avoid division by 0
            images /= upper_quantile  # scale to ~[0, 1]
            images = images.reshape(b, c, h, w).clip(0, 1).cpu().numpy()
        else:  # or via naive assumption of originally [0, 1] images
            images: np.ndarray = (images + 1) / 2  # de-normalise to [0, 1]

        return images.clip(0, 1).transpose(0, 2, 3, 1)  # clip for display and convert to channels last format

    def get_mean_std(self) -> tuple[Float[Tensor, "channels"], Float[Tensor, "channels"]]:
        """
        Retrieve the mean and std (per channel) values for the dataset. If not already calculated, calculate them.

        :return: A tuple of Tensors, one each for the mean and std values (per channel) for the dataset.
        """
        if self.mean.numel() == 0 or self.var.numel() == 0 or self.std.numel() == 0:
            logger.info(f"Mean and std not yet stored for {self.__class__.__name__}.")
            try:
                with np.load(self.mean_std_path) as data:  # type: dict[str, np.ndarray]
                    logger.debug(f"Loading mean, var, and std from {self.mean_std_path}.")
                    self.mean = torch.from_numpy(data["mean"]).to(self.device)
                    self.var = torch.from_numpy(data["var"]).to(self.device)
                    self.std = torch.from_numpy(data["std"]).to(self.device)
            except FileNotFoundError:
                logger.warning(f"Mean and std not found in {self.mean_std_path}. "
                               f"Calculating from dataloader instead.")
                self.calculate_channel_wise_mean_std(self.get_original_train_dataloader())

        return self.mean.cpu(), self.std.cpu()

    def calculate_channel_wise_mean_std(
            self,
            dataloader: torch.utils.data.DataLoader[dict[str, Tensor]],
    ):
        # Adapted from https://stackoverflow.com/a/60803379/7253717
        n_images = 0
        self.mean = torch.zeros(self.N_BANDS).to(self.device)
        self.var = torch.zeros(self.N_BANDS).to(self.device)
        with tqdm(total=len(dataloader)*2, desc=f"Mean/std of {self.__class__.__name__}",
                  unit="batch", ncols=110, leave=False) as pbar:
            for i, batch in enumerate(dataloader):  # type: int, dict[str, Tensor]
                images = batch["image"].to(self.device)
                b, c, h, w = images.shape
                # Rearrange batch to be the shape of [B, C, H*W]
                images = images.view(b, c, -1)
                # Update total number of images
                n_images += b
                # Compute mean and std here (over H*W and then sum over b)
                self.mean += images.mean(2).sum(0)

                pbar.update()
                logger.debug(str(pbar)) if i % 100 == 0 else None
            self.mean /= n_images

            # Variance calculation requires whole dataset's mean
            for i, batch in enumerate(dataloader):  # type: int, dict[str, Tensor]
                images = batch["image"].to(self.device)
                b, c, h, w = images.shape
                images = images.view(b, c, -1)
                self.var += ((images - self.mean.view(1, c, 1)) ** 2).mean(2).sum(0)

                pbar.update()
                logger.debug(str(pbar)) if i % 100 == 0 else None
            self.var /= n_images
            self.std = torch.sqrt(self.var)

        logger.info(f"Calculated mean and std for {self.logging_name} "
                    f"with {len(self.bands)} bands: mean={self.mean}, std={self.std}.")

        np.savez_compressed(self.mean_std_path,
                            mean=self.mean.cpu().numpy(),
                            var=self.var.cpu().numpy(),
                            std=self.std.cpu().numpy())
        logger.info(f"Saved mean and std for {self.logging_name} to {self.mean_std_path}.")

    def build_transforms(
            self,
            scaling_transform: RSScalingTransform = None,
            normalisation: vision_transforms.Normalize = None,
            augmentations: list[vision_transforms.Transform] = None,
            use_resize: bool = True,
    ) -> t.Callable[[dict[t.Literal["image", "label"], Tensor]], dict[t.Literal["image", "label"], Tensor]]:
        """
        Builds and composes a series of image transformation operations to preprocess input
        images for a model. The function incorporates scaling, normalisation, optional
        augmentations, and resize operations. It outputs a callable composed of the
        specified transformations applicable to a dictionary of Tensors (the convention for torchgeo).
        All parameters are optional. If they are not provided, that transform is not applied.

        :param scaling_transform: An optional value scaling transformation to be applied to the images.
            Usually used to scale input reflectance values to a specific range (e.g. [0, 2750] -> [0, 1]).
        :param normalisation: An optional normalisation transformation to standardise the
            image data values to have approximately zero mean and unit variance.
        :param augmentations: A list of optional data augmentation transformations
            (torchvision.transforms.Transform) that can be applied to batches of images.
        :param use_resize: Indicates whether images should be resized to a specific size
            using interpolation (`torchvision.transforms.Resize`) (True, the default)
            or padded and centered (`torchvision.transforms.CenterCrop`).

        :return: A callable transformation wrapping all the specified operations.
            Should be applied to dictionaries of tensors with the "image" key.
        """

        transform_list: list[t.Union[vision_transforms.Transform, RSScalingTransform]] = [
            vision_transforms.ToImage(),
            vision_transforms.ToDtype(torch.float32, scale=False),  # scaling handled by normalise below
        ]

        if scaling_transform is not None:
            transform_list.append(scaling_transform)
            logger.debug(f"Using {scaling_transform} for {self.logging_name}.")

        if normalisation is not None:
            transform_list.append(normalisation)
            logger.debug(f"Using {normalisation} for {self.logging_name}.")

        if augmentations is not None:
            transform_list += augmentations
            logger.debug(f"Applying additional augmentations={augmentations} for {self.logging_name}.")

        # Resize to image_size required by the input layer of the model
        if use_resize:  # Rescale the image to the required size via interpolation
            # Warning! Behaves inconsistently across platforms (Windows vs Unix)!!!
            scaling_transform = vision_transforms.Resize(
                self.image_size, interpolation=vision_transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
            logger.debug(f"Upsizing {self.logging_name} images via Resize.")
        else:  # Just put the image in the middle and pad around it
            scaling_transform = vision_transforms.CenterCrop(self.image_size)
            logger.debug(f"Upsizing {self.logging_name} images via CenterCrop.")
        transform_list.append(scaling_transform)

        self.composed_transforms = vision_transforms.Compose(transform_list)

        logger.info(f"Built transforms for {self.logging_name}.")
        wrapped_transforms = TensorDictTransformWrapper(self.composed_transforms)
        return wrapped_transforms
