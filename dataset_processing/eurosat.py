import typing as t

import torch
from torchgeo.datasets import EuroSAT
from torchvision.transforms import v2 as vision_transforms

import dataset_processing.core
import helpers

DATASET_ROOT = helpers.env_var.get_dataset_root()
logger = helpers.log.get_logger("main")


class EuroSATBase(EuroSAT, dataset_processing.core.RSDatasetMixin):
    def __init__(
            self,
            split: t.Literal["train", "val", "test"],
            image_size: int,
            variant: t.Literal["rgb", "ms"] = "rgb",
            download: bool = False,
            normalisation_type: t.Literal["scaling", "mean_std", "none"] = "scaling",
            use_augmentations: bool = True,
            use_resize: bool = True,
            batch_size: int = 32,
            num_workers: int = 4,
            device: torch.device = "cpu",
    ):
        """
        :param split: Which dataset split to use. One of "train", "val", or "test".
        :param image_size: The size images should be scaled to before being passed to the model.
        :param variant: Which variant of EuroSAT to use. One of "rgb" or "ms".
        :param download: Whether to download the dataset if it is not already present in DATASET_ROOT.
        :param normalisation_type: Which type of normalisation to apply to the dataset.
            One of "scaling" (uses min/max or percentile scaling depending on variant),
            "mean_std" (which computes the mean/std for each channel across the whole train dataset),
            or "none" (which applies no normalisation).
        :param use_augmentations: Whether to apply random augmentations to the data. N/A if split != "train".
        :param use_resize: Whether to use torchvision.transforms.Resize to scale images.
            If False, torchvision.transforms.CenterCrop is used instead, placing images in the centre with padding.
        :param batch_size: The batch size to use for the dataloader should mean_std be specified.
        """
        dataset_processing.core.RSDatasetMixin.__init__(
            self, split=split, image_size=image_size, batch_size=batch_size, num_workers=num_workers, device=device,
        )

        self.variant = variant

        if self.variant == "rgb":
            self.bands = self.rgb_bands  # ("B04", "B03", "B02")
        elif self.variant == "ms":
            # bands = self.all_band_names
            self.bands = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B12")

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
        if self.split == "train" and use_augmentations:
            augmentations = [vision_transforms.RandomHorizontalFlip(p=0.5)]
            if self.variant != "rgb":
                augmentations += [
                    vision_transforms.RandomVerticalFlip(p=0.5),
                    vision_transforms.RandomAffine(
                        degrees=2, translate=(0.1, 0.1), shear=0.2,
                        scale=(0.95, 1.05), fill=0
                    ),
                ]

        self.transforms = self.build_transforms(scaling_transform, normalisation, augmentations, use_resize)

        super().__init__(
            root=str(DATASET_ROOT / "eurosat"),
            split=split,
            bands=self.bands,
            transforms=self.transforms,
            download=download
        )

        self.N_CLASSES = len(self.classes)

    def get_original_train_dataloader(self):
        return torch.utils.data.DataLoader(EuroSAT(
            root=str(DATASET_ROOT / "eurosat"),
            split="train",
            bands=self.bands,
            transforms=None,
            download=False
        ), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class EuroSATRGB(EuroSATBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, variant="rgb")


class EuroSATMS(EuroSATBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, variant="ms")
