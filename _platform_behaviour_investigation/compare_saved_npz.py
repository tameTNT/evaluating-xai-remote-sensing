import numpy as np
import torch


print("Numpy version:", np.__version__)

with np.load("xai_output_windows/shap/ResNet50.npz") as data:  # type: dict[str, np.ndarray]
    a = torch.from_numpy(data["explanation_input"])
    exp_a = torch.from_numpy(data["explanation"])

with np.load("xai_output/shap/ResNet50.npz") as data:  # type: dict[str, np.ndarray]
    b = torch.from_numpy(data["explanation_input"])
    exp_b = torch.from_numpy(data["explanation"])

diff = (a - b).abs()
print(diff.sum(), diff.mean(), diff.max())

exp_diff = (exp_a - exp_b).abs()
print(exp_diff.sum(), exp_diff.mean(), exp_diff.max())
