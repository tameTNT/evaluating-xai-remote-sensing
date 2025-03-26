import os
import sys
from pathlib import Path

import torch

CUR_DIR = Path(__file__).parent
sys.path.append(str(CUR_DIR.parent))

import dataset_processing
import helpers

import numpy as np
import platform


print("Executing...")
RESIZE_SIZE = 96

torch.manual_seed(42)
np_rng = np.random.default_rng(42)

eurosat = dataset_processing.get_dataset_object(
    "EuroSATRGB", "val", RESIZE_SIZE,
    normalisation_type="scaling", use_resize=True,
    batch_size=32, num_workers=4, device=helpers.utils.get_torch_device(),
    download=False,
)
classes = np.array([class_ for _, class_ in eurosat.imgs])
class_idxs = np.where(classes == 0)[0]
class_idxs_sampled = np_rng.choice(class_idxs, 32, replace=False)
batch = torch.stack([eurosat[j]["image"] for j in class_idxs_sampled])

torch.set_printoptions(precision=10)
print(batch[0])

np.savez_compressed(CUR_DIR / f"eurosat_rgb_{RESIZE_SIZE:03}_{platform.system()}.npz", batch=batch.numpy())

with np.load(CUR_DIR / f"eurosat_rgb_{RESIZE_SIZE:03}_Windows.npz") as data:
    saved_version_win = torch.from_numpy(data["batch"])
with np.load(CUR_DIR / f"eurosat_rgb_{RESIZE_SIZE:03}_Darwin.npz") as data:
    saved_version_mac = torch.from_numpy(data["batch"])
with np.load(CUR_DIR / f"eurosat_rgb_{RESIZE_SIZE:03}_Linux.npz") as data:
    saved_version_linux = torch.from_numpy(data["batch"])

print("Difference between Win/Mac version:", (saved_version_win.float() - saved_version_mac.float()).abs().sum())
print("Difference between Linux/Mac version:", (saved_version_linux.float() - saved_version_mac.float()).abs().sum())
print("Difference between Linux/Win version:", (saved_version_linux.float() - saved_version_win.float()).abs().sum())
