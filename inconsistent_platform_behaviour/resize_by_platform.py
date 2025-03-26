import os
import platform

import torchvision
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import torch

print(torch.__version__, torchvision.__version__)
torch.manual_seed(42)

rand_tensor = torch.stack(
    torch.meshgrid(
        torch.linspace(-1, 1, 65),
        torch.linspace(-1, 1, 65),
        indexing="ij",
    )
).cpu().to(torch.float32)
# torch.set_printoptions(precision=10)
# print(rand_tensor[0])

torch.save(rand_tensor, f"rand_tensor_{platform.system()}.pt")

resized_tensor = resize(
    rand_tensor, [66, 66],
    interpolation=InterpolationMode.BILINEAR, antialias=True
)
# print(resized_tensor[0])

torch.save(resized_tensor, f"resized_tensor_{platform.system()}.pt")

if (os.path.exists("resized_tensor_Windows.pt")
        and os.path.exists("resized_tensor_Darwin.pt")
        and os.path.exists("resized_tensor_Linux.pt")):
    win_tensor = torch.load("rand_tensor_Windows.pt", weights_only=True)
    mac_tensor = torch.load("rand_tensor_Darwin.pt", weights_only=True)
    linux_tensor = torch.load("rand_tensor_Linux.pt", weights_only=True)
    print("Difference between original tensors:")
    print("    Win/Mac:", (win_tensor.float() - mac_tensor.float()).abs().sum())
    print("  Win/Linux:", (win_tensor.float() - linux_tensor.float()).abs().sum())
    print("  Linux/Mac:", (mac_tensor.float() - linux_tensor.float()).abs().sum())

    win_resized_tensor = torch.load("resized_tensor_Windows.pt", weights_only=True)
    mac_resized_tensor = torch.load("resized_tensor_Darwin.pt", weights_only=True)
    linux_resized_tensor = torch.load("resized_tensor_Linux.pt", weights_only=True)
    print("Difference between resized tensors:")
    print("    Win/Mac:", (win_resized_tensor.float() - mac_resized_tensor.float()).abs().sum())
    print("  Win/Linux:", (win_resized_tensor.float() - linux_resized_tensor.float()).abs().sum())
    print("  Linux/Mac:", (mac_resized_tensor.float() - linux_resized_tensor.float()).abs().sum())
else:
    print("One or more tensor files do not exist. Please check the file paths and try again.")
