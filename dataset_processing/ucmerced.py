import torch
from torchgeo.datasets import UCMerced as UCMercedBase
from torchvision.transforms import v2 as vision_transforms

import dataset_processing.core
import helpers

DATASET_ROOT = helpers.env_var.get_dataset_root()
logger = helpers.log.get_logger("main")


class UCMerced(UCMercedBase, dataset_processing.core.RSDatasetMixin):
    def __init__(self, **kwargs):
        dataset_processing.core.RSDatasetMixin.__init__(self, **kwargs)

        # only uses RGB bands
        self.bands = ("B04", "B03", "B02")
        self.N_BANDS = len(self.bands)

        # Scale the image to 0-1 floats instead of integer floats (e.g. 125., 9.)
        scaling_transform = dataset_processing.core.RSScalingTransform(
            input_min=0., input_max=255., clamp=False
        )
        normalisation = vision_transforms.Normalize(mean=[0.5] * self.N_BANDS,
                                                    std=[0.5] * self.N_BANDS, inplace=True)

        # Add randomised transforms
        augmentations = None
        if self.split == "train" and self.use_augmentations:
            augmentations = dataset_processing.core.BASIC_ROTATION_AUGMENTATIONS

        self.transforms = self.build_transforms(scaling_transform, normalisation, augmentations, self.use_resize)

        super().__init__(
            root=str(DATASET_ROOT / "ucmerced"),
            split=self.split,
            transforms=self.transforms,
            download=self.download,
        )

        self.N_CLASSES = len(self.classes)

    def get_original_train_dataloader(self, shuffle=False):
        return torch.utils.data.DataLoader(UCMercedBase(
            root=str(DATASET_ROOT / "ucmerced"),
            split="train",
            transforms=None,
            download=self.download,
        ), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)
