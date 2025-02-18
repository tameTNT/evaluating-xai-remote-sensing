import logging
import typing as t

import torch
import torchgeo.datasets
import torchvision.transforms.v2 as transforms

import dataset_processing.core

DATASET_ROOT = dataset_processing.core.get_dataset_root()
logger = logging.getLogger("projectLog")


def get_base_dataset(
        split: t.Literal["train", "test", "val"],
        bands: t.Sequence[str] = ('B04', 'B03', 'B02'),  # defaults to RGB bands
        transforms: t.Optional[t.Callable[[t.Dict[str, torch.Tensor]], t.Dict[str, torch.Tensor]]] = None,
        download=False,
) -> torchgeo.datasets.EuroSAT:
    """
    Get the EuroSAT dataset using torchgeo (can download if necessary).

    Defaults to returning only the RGB bands ('B04', 'B03', 'B02'). But supports all 13 bands.
    """
    logger.debug("Using `torchgeo.datasets` to load EuroSAT dataset.")
    return torchgeo.datasets.EuroSAT(
        root=str(DATASET_ROOT / "eurosat"),
        split=split,
        bands=bands,
        transforms=transforms,
        download=download
    )


def get_standard_rgb(split: t.Literal["test", "val"]) -> torchgeo.datasets.EuroSAT:
    base_transforms = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=False),  # scaling handles by normalise below
        dataset_processing.core.RSNormaliseTransform(0, 2750),
        # normalise to [0, 1] (based on maximums used in original paper)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # shift to mean 0 and std 1

        # scale as expected by ResNet (see https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights)
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    wrapped_base_transforms = dataset_processing.core.tensor_dict_transform_wrapper(base_transforms)

    return get_base_dataset(
        split, bands=("B04", "B03", "B02"),  # RGB bands
        transforms=wrapped_base_transforms, download=False
    )
