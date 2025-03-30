import torch
from torchgeo.datasets import BigEarthNetV2 as BigEarthNetV2Base
from torchvision.transforms import v2 as vision_transforms

import dataset_processing.core
import helpers

DATASET_ROOT = helpers.env_var.get_dataset_root()


class ReBEN(BigEarthNetV2Base, dataset_processing.core.RSDatasetMixin):
    def __init__(self, **kwargs):
        dataset_processing.core.RSDatasetMixin.__init__(self, **kwargs)

        scaling_transform = None
        normalisation = None

        # Add randomised transforms
        augmentations = None
        if self.split == "train" and self.use_augmentations:
            augmentations = []

        self.transforms = self.build_transforms(scaling_transform, normalisation, augmentations, self.use_resize)

        super().__init__(
            root=str(DATASET_ROOT / "reBEN"),
            split=self.split,
            bands="s2",
            transforms=self.transforms,
            download=self.download,
        )

        self.N_CLASSES = self._load_image(0, "s2").shape[0]
        self.N_BANDS = len(self.bands)

    def get_original_train_dataloader(self, shuffle=False):
        return torch.utils.data.DataLoader(BigEarthNetV2Base(
            root=str(DATASET_ROOT / "reBEN"),
            split="train",
            transforms=None,
            download=self.download,
        ), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample: dict[str, torch.Tensor] = {
            "image": self._load_image(index, "s2"),
            "mask": self._load_map(index),
            "label": self._load_target(index)
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
