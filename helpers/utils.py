import platform
from typing import Generator

import numpy as np
import torch
from jaxtyping import Float, Int

from helpers import log  # direct import to avoid circular import

logger = log.main_logger


# This function is left here in utils to allow for use of logger
# (imported after log in __init__.py) without causing circular import issues.
def get_torch_device(force_mps: bool = False) -> torch.device:
    """
    Try to get the best PyTorch device available: cuda > mps > cpu.
    Note that on macOS, will only return mps if torch.version >= 2.6 as mps is
    still missing some required features (see comment below).
    Use `force_mps=True` if you want to get mps as the device anyway.

    Also updates the number of threads for torch on non-Linux systems to 1.
    """

    if torch.cuda.is_available():
        torch_device = torch.device("cuda")
        logger.debug(f"Found {torch.cuda.get_device_name()} to use as a cuda device.")

    elif platform.system() == "Darwin":
        # noinspection PyTypeChecker
        if torch.torch_version.TorchVersion(torch.__version__) >= (2, 6) or force_mps:
            torch_device = torch.device("mps")
        else:
            # See https://github.com/pytorch/pytorch/issues/142344

            # Example error when running GradCAM explainer using MPS device:
            #   File ".../site-packages/torch/autograd/graph.py", line 825, in _engine_run_backward
            #     return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
            #   RuntimeError: view size is not compatible with input tensor's size and stride
            #   (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

            logger.warning("MPS with backward graph computations is only fixed in PyTorch 2.6.0 and later. "
                           "Falling back to cpu.")
            torch_device = torch.device("cpu")
    else:
        torch_device = torch.device("cpu")
    logger.info(f"Using {torch_device} as torch device.")

    if platform.system() != "Linux":
        # significantly speeds up data loading processes with less loading overhead
        # see https://discuss.pytorch.org/t/pytorch-v2-high-cpu-consumption/205990 and
        # https://discuss.pytorch.org/t/cpu-usage-far-too-high-and-training-inefficient/57228
        torch.set_num_threads(1)
        logger.debug("Set number of threads to 1 as using a non-Linux machine.")

    return torch_device


def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Returns the torch device of a torch Module (model).
    """

    return next(model.parameters()).device


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """
    Returns the torch dtype used by a torch Module (model).
    """

    return next(model.parameters()).dtype


def make_device_batches(
        x: torch.Tensor,
        max_batch_size: int,
        target_device: torch.device
) -> Generator[torch.Tensor, None, None]:
    """
    Splits a tensor `x` into mini batches and moves them to the target device.
    Yields these mini batches (of size up to max_batch_size) one by one.
    """

    for i in range(0, x.shape[0], max_batch_size):
        yield x[i:i + max_batch_size].to(target_device)


def rank_pixel_importance(
        x: Float[np.ndarray, "n_samples height width"]
) -> Int[np.ndarray, "n_samples height width"]:
    """
    Convert pixel importance to *rank* pixel importance (0 = most important)
    for each image in `x`.
    """

    n, h, w = x.shape
    # per_pixel_contribution = np.sum(x, axis=-1)  # sum over colour channels
    # flatten over each image
    flat_imgs = x.reshape(n, -1)
    pixel_ranks = np.argsort(np.argsort(flat_imgs))
    # invert to rank from 0 to h*w-1, with 0 being the most important pixel
    pixel_ranks_0_top = np.abs(pixel_ranks - (h * w) + 1).reshape(n, h, w)

    return pixel_ranks_0_top
