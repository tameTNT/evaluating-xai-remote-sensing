import torch
from torch.utils.data import Subset
from torchgeo.datasets import PatternNet as PatternNetBase, NonGeoClassificationDataset
from torchvision.transforms import v2 as vision_transforms
from sklearn.model_selection import train_test_split
import numpy as np

import dataset_processing.core
import helpers

DATASET_ROOT = helpers.env_var.get_dataset_root()


class PatternNet(PatternNetBase, dataset_processing.core.RSDatasetMixin):
    def __init__(
            self,
            random_split_seed: int = 42,
            **kwargs
    ):
        dataset_processing.core.RSDatasetMixin.__init__(self, **kwargs)

        # only uses RGB bands
        self.bands = ("B04", "B03", "B02")
        self.N_BANDS = len(self.bands)

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
            root=str(DATASET_ROOT / "patternnet"),
            transforms=self.transforms,
            download=self.download,
        )

        self.N_CLASSES = len(self.classes)

        # Manually add dataset splits (not provided by torchgeo)
        all_class_labels = np.array([l for _, l in self.imgs])

        train_idxs, split_idxs = train_test_split(
            np.arange(len(all_class_labels)), test_size=0.4,
            stratify=all_class_labels, random_state=random_split_seed
        )

        val_idxs, test_idxs = train_test_split(
            split_idxs, test_size=0.5,
            stratify=all_class_labels[split_idxs], random_state=random_split_seed
        )
        self.split_idxs = {  # 60%, 20%, 20% split
            "train": train_idxs,
            "val": val_idxs,
            "test": test_idxs
        }
        # # replace self.imgs with only the imgs in the split
        # self.imgs = self.imgs[self.split_idxs[self.split]]

    def __getitem__(self, idx):
        split_idx = self.split_idxs[self.split][idx]
        return super().__getitem__(split_idx)

    def __len__(self):
        return len(self.split_idxs[self.split])

    def get_original_train_dataloader(self, shuffle=False):
        return torch.utils.data.DataLoader(
            Subset(
                PatternNetBase(
                    root=str(DATASET_ROOT / "patternnet"),
                    transforms=None,
                    download=self.download,
                ), self.split_indexes["train"]
            ), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle
        )
