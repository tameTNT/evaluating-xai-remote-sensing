import typing as t

import torch
from torchgeo.datasets import EuroSAT
from torchvision.transforms import v2 as vision_transforms

import dataset_processing.core
import helpers

DATASET_ROOT = helpers.env_var.get_dataset_root()
logger = helpers.log.main_logger


class EuroSATBase(EuroSAT, dataset_processing.core.RSDatasetMixin):
    def __init__(
            self,
            variant: t.Literal["rgb", "ms"] = "rgb",
            normalisation_type: t.Literal["scaling", "mean_std", "none"] = "scaling",
            **kwargs,
    ):
        """
        :param variant: Which variant of EuroSAT to use. One of "rgb" or "ms".
        :param normalisation_type: Which type of normalisation to apply to the dataset.
            One of "scaling" (uses min/max or percentile scaling depending on variant),
            "mean_std" (which computes the mean/std for each channel across the whole train dataset),
            or "none" (which applies no normalisation).
        """

        dataset_processing.core.RSDatasetMixin.__init__(self, **kwargs)

        self.variant = variant

        if self.variant == "rgb":
            self.bands = self.rgb_bands  # ("B04", "B03", "B02")
        elif self.variant == "ms":
            # bands = self.all_band_names
            self.bands = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B12")

            # bands 1, 9, 10 only for atmospheric correction? (https://doi.org/10.1109/IGARSS47720.2021.9553337)
            # bands 1, 9, 10, 11 not used:
            # https://ceurspt.wikidata.dbis.rwth-aachen.de/Vol-2771/AICS2020_paper_50.pdf
        else:
            raise NotImplementedError(f"Unsupported EuroSAT version: {self.variant}")
            # todo: new combination of bands e.g. NDVI, NDWI, etc.
            #   see indices in https://doi.org/10.5194/isprs-archives-XLIII-B3-2021-369-2021
            #   see also https://torchgeo.readthedocs.io/en/stable/tutorials/transforms.html

        self.N_BANDS = len(self.bands)

        # Build transforms
        scaling_transform = None
        normalisation = None
        if normalisation_type == "scaling":
            if self.variant == "rgb":
                scaling_transform = dataset_processing.core.RSScalingTransform(input_min=0, input_max=2750, clamp=True)
            else:
                scaling_transform = dataset_processing.core.RSScalingTransform(channel_wise=True)

            # Shift to mean 0 and std 1, [-1, 1] assuming input is uniform [0, 1]. Same as 2x - 1 =(x - 0.5)/0.5
            normalisation = vision_transforms.Normalize(mean=[0.5] * self.N_BANDS,
                                                        std=[0.5] * self.N_BANDS, inplace=True)
            logger.debug(f"Normalising {self.logging_name} assuming mean and std of 0.5.")
            # Scale as expected by ResNet (see torchvision docs)
            # vision_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        elif normalisation_type == "mean_std":
            mean, std = self.get_mean_std()
            normalisation = vision_transforms.Normalize(mean=mean, std=std, inplace=True)
            logger.debug(f"Normalising {self.logging_name} using calculated mean and std.")

        elif normalisation_type != "none":
            raise ValueError(f"Unsupported normalisation type: {normalisation_type}")

        # Add randomised transforms
        augmentations = None
        if self.split == "train" and self.use_augmentations:
            augmentations = [vision_transforms.RandomHorizontalFlip(p=0.5)]
            if self.variant != "rgb":
                augmentations += [
                    vision_transforms.RandomVerticalFlip(p=0.5),
                    vision_transforms.RandomAffine(
                        degrees=2, translate=(0.1, 0.1), shear=0.2,
                        scale=(0.95, 1.05), fill=0
                    ),
                ]

        self.transforms = self.build_transforms(scaling_transform, normalisation, augmentations, self.use_resize)

        super().__init__(
            root=str(DATASET_ROOT / "eurosat"),
            split=self.split,
            bands=self.bands,
            transforms=self.transforms,
            download=self.download
        )

        self.N_CLASSES = len(self.classes)

    def get_original_train_dataloader(self, shuffle=False):
        return torch.utils.data.DataLoader(EuroSAT(
            root=str(DATASET_ROOT / "eurosat"),
            split="train",
            bands=self.bands,
            transforms=None,
            download=self.download,
        ), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)


class EuroSATRGB(EuroSATBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, variant="rgb")


class EuroSATMS(EuroSATBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, variant="ms")
