import typing as t

import torch
import torchgeo.datasets
import torchvision.datasets.utils
from torchgeo.datasets import BigEarthNet

import helpers

DATASET_ROOT = helpers.env_var.get_dataset_root()
logger = helpers.log.get_logger("main")


def get_dataset(
        split: t.Literal["train", "test", "val"],
        num_classes: t.Literal[19, 43] = 19,
        transforms: t.Optional[t.Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]] = None,
) -> torchgeo.datasets.BigEarthNet:
    """
    Get the Refined Big Earth Net dataset.
    See https://github.com/microsoft/torchgeo/pull/2371 for v2 dataloader implementation.

    Needs to be downloaded separately first
    (you can use bash script in `/dataset_processing/scripts/download_sen12ms.sh`).
    """
    reben_directory = DATASET_ROOT / "reBEN"
    dataset_directory = reben_directory / "ds"
    if not dataset_directory.exists():
        # extract the dataset
        zst_path = reben_directory / "BigEarthNet-S2.tar.zst"
        if not zst_path.exists():
            raise RuntimeError(
                f"Dataset not found in {reben_directory} and `download=False`, "
                "either specify a different DATASET_ROOT directory or download the dataset."
                "You can use `wget https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst` for example."
            )
        else:
            # extract the dataset
            torchvision.datasets.utils.extract_archive(str(zst_path), str(dataset_directory))

    return BigEarthNet(
        root=str(dataset_directory.resolve()),
        split=split,
        num_classes=num_classes,
        transforms=transforms,
        download=False,
        checksum=False
    )


if __name__ == "__main__":
    get_dataset(split="train")
