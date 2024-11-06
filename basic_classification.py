import random

import einops.einops as einops
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

import datasets.core
import datasets.eurosat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

resize_transform = datasets.core.tensor_dict_transform_wrapper(transforms.Compose([
    transforms.Resize((224, 224), antialias=False),  # Authors rescale images to 224x224
    datasets.core.ClampTransform(input_min=0., input_max=2750.)  # Authors clamp from 0.-2750 but torchgeo uses 3000
]))

eurosat_train_ds = datasets.eurosat.get_dataset("train", transforms=resize_transform)
# eurosat_test_ds = datasets.eurosat.get_dataset("test", transforms=resize_transform)
print(f"There are {len(eurosat_train_ds)} training samples")  # and {len(eurosat_test_ds)} test samples.")
print(eurosat_train_ds[0]["image"].size())

# Display 25 random images from the dataset without border
random.seed(42)
plt.figure(figsize=(10, 10), tight_layout=True)
for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.subplots_adjust(wspace=0, hspace=0)
    random_index = random.randint(0, len(eurosat_train_ds) - 1)
    plt.imshow(einops.rearrange(eurosat_train_ds[random_index]["image"], "c h w -> h w c"))
    plt.axis("off")

plt.show()
