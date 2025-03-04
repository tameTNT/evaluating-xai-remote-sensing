import numpy as np
import torch
from jaxtyping import Float, Int


def make_device_batches(
        x: torch.Tensor,
        max_batch_size: int,
        target_device: torch.device
) -> torch.Tensor:
    """
    Splits a tensor `x` into mini batches and moves them to the target device.
    Yields these mini batches (of size up to max_batch_size) one by one.
    """

    for i in range(0, x.shape[0], max_batch_size):
        yield x[i:i + max_batch_size].to(target_device)


def rank_pixel_importance(
        x: Float[np.ndarray, "batch_size height width"]
) -> Int[np.ndarray, "batch_size height width"]:
    """
    Convert pixel importance to rank pixel importance (0 = most important)
    for each image in `x`.
    """

    bs, h, w = x.shape
    # per_pixel_contribution = np.sum(x, axis=-1)  # sum over colour channels
    # flatten over each image
    flat_imgs = x.reshape(bs, -1)
    pixel_ranks = np.argsort(np.argsort(flat_imgs))
    # invert to rank from 0 to h*w-1, with 0 being the most important pixel
    pixel_ranks_0_top = np.abs(pixel_ranks - (h * w) + 1).reshape(bs, h, w)

    return pixel_ranks_0_top
