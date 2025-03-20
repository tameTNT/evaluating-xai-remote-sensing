import platform
from typing import Generator

import numpy as np
import torch
from jaxtyping import Float, Int

from helpers import log  # direct import to avoid circular import

logger = log.main_logger


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        torch_device = torch.device('cuda')
        logger.debug(f'Found {torch.cuda.get_device_name()} to use as a cuda device.')

    elif platform.system() == 'Darwin':
        torch_device = torch.device('mps')

    else:
        torch_device = torch.device('cpu')
    logger.info(f'Using {torch_device} as torch device.')

    if platform.system() != 'Linux':
        torch.set_num_threads(1)
        logger.debug('Set number of threads to 1 as using a non-Linux machine.')

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
    Convert pixel importance to rank pixel importance (0 = most important)
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
