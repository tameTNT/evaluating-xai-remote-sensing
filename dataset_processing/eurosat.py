import typing as t

import torch
import torchvision.transforms.v2 as tv_transforms
from torchgeo.datasets import EuroSAT

import dataset_processing.core
import helpers.logging

DATASET_ROOT = dataset_processing.core.get_dataset_root()
logger = helpers.logging.get_logger("main")


class EuroSATBase(EuroSAT):
    N_CLASSES = 10

    def __init__(
            self,
            split: t.Literal["train", "val", "test"],
            image_size: int,
            variant: t.Literal["rgb", "ms"] = "rgb",
            download: bool = False,
            use_normalisation: t.Union[bool, list[list[float], list[float]]] = True,
            use_augmentations: bool = True,
            use_resize: bool = True,
    ):
        """
        :param split: Which dataset split to use. One of "train", "val", or "test".
        :param image_size: The size images should be scaled to before being passed to the model.
        :param variant: Which variant of EuroSAT to use. One of "rgb" or "ms".
        :param download: Whether to download the dataset if it is not already present in DATASET_ROOT.
        :param use_normalisation: Whether to apply normalisation to the images to attempt scaling to [-1, 1] range.
        :param use_augmentations: Whether to apply random augmentations to the data. N/A if split != "train".
        :param use_resize: Whether to use torchvision.transforms.Resize to scale images.
            If False, torchvision.transforms.CenterCrop is used instead, placing images in the centre with padding.
        """

        self.split = split
        self.image_size = image_size
        self.variant = variant

        if self.variant == "rgb":
            bands = self.rgb_bands  # ("B04", "B03", "B02")
        elif self.variant == "ms":
            # bands = self.all_band_names
            bands = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B12")

            # bands 1, 9, 10 only for atmospheric correction? (https://doi.org/10.1109/IGARSS47720.2021.9553337)
            # bands 1, 9, 10, 11 not used:
            # https://ceurspt.wikidata.dbis.rwth-aachen.de/Vol-2771/AICS2020_paper_50.pdf
        else:
            raise NotImplementedError(f"Unsupported EuroSAT version: {self.variant}")
            # todo: new combination of bands e.g. NDVI, NDWI, etc.
            #   see indicies in https://doi.org/10.5194/isprs-archives-XLIII-B3-2021-369-2021
            #   see also https://torchgeo.readthedocs.io/en/stable/tutorials/transforms.html

        self.N_BANDS = len(bands)

        transforms = self.get_transforms(use_normalisation, use_augmentations, use_resize)

        super().__init__(
            root=str(DATASET_ROOT / "eurosat"),
            split=split,
            bands=bands,
            transforms=transforms,
            download=download
        )

    def get_transforms(
            self,
            use_normalisation: t.Union[bool, list[list[float], list[float]]],
            use_augmentations: bool,
            use_resize: bool
    ):  # todo: customise for RGB/MS and add more?
        # todo: generalise this to core dataset type/class
        transform_list = [tv_transforms.ToImage()]

        if self.variant == "rgb":
            # scaling handled by normalise below
            transform_list.append(tv_transforms.ToDtype(torch.float32, scale=False))
            if use_normalisation and isinstance(use_normalisation, bool):
                transform_list.append(
                    dataset_processing.core.RSNormaliseTransform(input_min=0, input_max=2750, clamp=True)
                )
                logger.debug(f"Using 0-2750 initial transforms for {self.__class__.__name__}")
        else:
            transform_list.append(tv_transforms.ToDtype(torch.float32, scale=False))
            if use_normalisation and isinstance(use_normalisation, bool):
                transform_list.append(
                    dataset_processing.core.RSNormaliseTransform(channel_wise=True)
                )
                logger.debug(f"Using channel_wise initial transforms for {self.__class__.__name__}")

        if use_normalisation:  # Shift to mean 0 and std 1, [-1, 1]
            if isinstance(use_normalisation, bool):  # assume input is uniform [0, 1]. Same as 2x - 1 =(x - 0.5)/0.5
                transform_list.append(
                    tv_transforms.Normalize(mean=[0.5]*self.N_BANDS, std=[0.5]*self.N_BANDS, inplace=True),

                    # Scale as expected by ResNet (see torchvision docs)
                    # tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                )
            else:
                transform_list.append(
                    tv_transforms.Normalize(mean=use_normalisation[0], std=use_normalisation[1], inplace=True),
                )

        # Add randomised transforms
        if self.split == "train" and use_augmentations:
            transform_list += [
                tv_transforms.RandomHorizontalFlip(p=0.5),
            ]
            if self.variant != "rgb":
                transform_list += [
                    tv_transforms.RandomVerticalFlip(p=0.5),
                    tv_transforms.RandomAffine(
                        degrees=2, translate=(0.1, 0.1), shear=0.2,
                        scale=(0.95, 1.05), fill=0
                    ),
                ]
                logger.debug(f"Applying additional random transforms for {self.__class__.__name__}")

        # Resize to image size required by input layer of model
        if use_resize:  # rescale the image to the required size via interpolation
            scaling_transform = tv_transforms.Resize(
                self.image_size, interpolation=tv_transforms.InterpolationMode.BILINEAR
            )
            logger.debug(f"Upsizing {self.__class__.__name__} images via Resize.")
        else:  # just put the image in the middle and pad around it
            scaling_transform = tv_transforms.CenterCrop(self.image_size)
            logger.debug(f"Upsizing {self.__class__.__name__} images via CenterCrop.")
        transform_list.append(scaling_transform)

        transforms = tv_transforms.Compose(transform_list)  # todo: move transforms to cuda?
        return dataset_processing.core.tensor_dict_transform_wrapper(transforms)


class EuroSATRGB(EuroSATBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, variant="rgb")


class EuroSATMS(EuroSATBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, variant="ms")
