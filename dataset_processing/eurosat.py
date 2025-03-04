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
            rgb_only: bool = True,
            download: bool = False,
            do_transforms: bool = True,
    ):
        self.split = split
        self.image_size = image_size
        self.rgb_only = rgb_only

        if self.rgb_only:
            bands = self.rgb_bands  # ("B04", "B03", "B02")
        # todo: new combination of bands e.g. NDVI, NDWI, etc.
        #   see indicies in https://doi.org/10.5194/isprs-archives-XLIII-B3-2021-369-2021
        else:
            bands = self.all_band_names  # todo: evaluation paper excludes B10 band?
            # bands 1, 9, 10 are for atmospheric correction? (https://doi.org/10.1109/IGARSS47720.2021.9553337)
            # bands 1, 9, 10, 11 not used:
            # https://ceurspt.wikidata.dbis.rwth-aachen.de/Vol-2771/AICS2020_paper_50.pdf

        self.N_BANDS = len(bands)

        if do_transforms:
            transforms = self.get_transforms()
        else:
            transforms = None

        super().__init__(
            root=str(DATASET_ROOT / "eurosat"),
            split=split,
            bands=bands,
            transforms=transforms,
            download=download
        )

    def get_transforms(self):  # todo: customise for RGB/MS and add more?
        transform_list = [
            tv_transforms.ToImage(),
        ]

        if self.rgb_only:
            transform_list += [
                tv_transforms.ToDtype(torch.float32, scale=False),  # scaling handles by normalise below
                dataset_processing.core.RSNormaliseTransform(0, 2750),
            ]
        else:
            transform_list += [
                tv_transforms.ToDtype(torch.float32, scale=False),
                dataset_processing.core.RSNormaliseTransform(channel_wise=True),
            ]

        if self.split == "train":
            # Add randomised transforms
            transform_list += [
                tv_transforms.RandomHorizontalFlip(p=0.5),
            ]
            if not self.rgb_only:
                transform_list += [
                    tv_transforms.RandomVerticalFlip(p=0.5),
                    tv_transforms.RandomAffine(
                        degrees=2, translate=(0.1, 0.1), shear=0.2,
                        scale=(0.95, 1.05), fill=0
                    ),
                ]

        transform_list += [
            # Shift to mean 0 and std 1 (i.e. [-1, 1])
            tv_transforms.Normalize(mean=[0.5]*self.N_BANDS, std=[0.5]*self.N_BANDS, inplace=True),

            # Scale as expected by ResNet (see torchvision docs)
            # tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

            # Resize to image size required by input layer of model
            tv_transforms.Resize(self.image_size, interpolation=tv_transforms.InterpolationMode.BILINEAR),
        ]

        transforms = tv_transforms.Compose(transform_list)

        return dataset_processing.core.tensor_dict_transform_wrapper(transforms)


class EuroSATRGB(EuroSATBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, rgb_only=True)


class EuroSATMS(EuroSATBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, rgb_only=False)
