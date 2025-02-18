import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import skimage
import torch
from jaxtyping import Float, Int


# for SHAP values this is better evaluated with L2 (hamming doesn't make sense
# (not the same)); SSIM is for heatmaps
def pixel_l2_distance_per_label(
        x1: Float[np.ndarray, "batch_size labels height width channels"],
        x2: Float[np.ndarray, "batch_size labels height width channels"]
) -> Float[torch.Tensor, "labels"]:
    """
    Calculate mean L2 distance between two sets of images (`x1`, `x2`) per label
    over the batch size.
    Each image (H, W, C) is flattened (i.e. to H*W*C) and the L2 distance is
    calculated per element.
    """

    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    # flatten image dimensions (HxWxC)
    # then sum squared differences and take average over num samples (first dim)
    # This is the L2 (Euclidian) distance per label
    return (x1 - x2).flatten(-3, -1).pow(2).sum(-1).mean(dim=0)


def rank_pixel_importance(
        x: Float[np.ndarray, "batch_size height width channels"]
) -> Int[np.ndarray, "batch_size height width"]:
    """
    Convert pixel importance to rank pixel importance (0 = most important)
    for each image in `x`.
    """

    *ld, h, w, c = x.shape
    per_pixel_contribution = np.sum(x, axis=-1)  # sum over colour channels
    # flatten over each image
    flat_imgs = per_pixel_contribution.reshape(*ld, -1)
    pixel_ranks = np.argsort(np.argsort(flat_imgs))
    # invert to rank from 0 to h*w-1, with 0 being the most important pixel
    pixel_ranks_0_top = np.abs(pixel_ranks - (h * w) + 1).reshape(*ld, h, w)

    return pixel_ranks_0_top


DELETION_METHODS = t.Union[float, int, np.random.Generator,
t.Literal["blur", "inpaint", "nn"]]


def delete_top_k_important(
        x: Float[np.ndarray, "batch_size channels height width"],
        importance_rank: Int[np.ndarray, "batch_size height width"],
        k: int,
        method: DELETION_METHODS,
) -> Float[np.ndarray, "batch_size channels height width"]:
    """
    'Delete' the top k most important pixels (as specified by `importance_rank`)
    in `x` (values in [-1,1]) using one of several methods
    determined by the `method` parameter:

    - If method is a float or int, set the top k pixels to this value.
    - If method is 'blur', apply a Gaussian blur to the top k pixels.
    - If method is a numpy random generator (`np.random.Generator`), add
        Gaussian noise to the top k pixels.
    - If method is 'inpaint', replace/inpaint the top k pixels using
        `skimage.restoration.inpaint_biharmonic`.
    - If method is 'nn', replace the top k pixels using nearest neighbour
        interpolation.
    """

    masked_img = x.copy()
    top_k_mask = importance_rank < k
    target_region = masked_img[:, top_k_mask]  # mask across all colour channels

    # we need at least one 'known' pixel (where the top_k_mask is False)
    # for the inpaint and nn methods
    if method in ["inpaint", "nn"]:
        if np.sum(~top_k_mask) < 1:
            top_k_mask[top_k_mask.shape[0] // 2,
                       top_k_mask.shape[1] // 2] = False

    if isinstance(method, (float, int)):
        masked_img[:, top_k_mask] = np.clip(method, -1, 1)

    elif method == "blur":
        masked_img[:, top_k_mask] = scipy.ndimage.gaussian_filter(
            target_region, sigma=5
        )

    elif isinstance(method, np.random.Generator):
        noise = method.normal(size=target_region.shape) / 5
        masked_img[:, top_k_mask] = np.clip(target_region + noise, -1, 1)

    elif method == "inpaint":
        mask = np.zeros_like(importance_rank)
        mask[top_k_mask] = 1  # indicate unknown pixels
        masked_img = skimage.restoration.inpaint_biharmonic(
            masked_img, mask, channel_axis=0
        )

    elif method == "nn":
        # Adapted from https://stackoverflow.com/a/27745627/7253717
        masked_img[:, top_k_mask] = np.nan
        filled_ind = scipy.ndimage.distance_transform_edt(
            np.isnan(masked_img), return_distances=False, return_indices=True)
        masked_img = masked_img[tuple(filled_ind)]
    return masked_img


def incrementally_delete(
        # todo: make channels position consistent across functions
        x: Float[np.ndarray, "channels height width"],
        importance_rank: t.Union[t.Tuple[int, np.random.Generator],
        Int[np.ndarray, "height width"]],
        num_iterations: int,
        method: DELETION_METHODS,
        show_random_grid: bool = False
) -> t.Tuple[
    Float[np.ndarray, "num_iterations channels height width"],
    Int[np.ndarray, "num_iterations"]
]:
    """
    Iteratively delete the top k most important pixels (as specified by
    `importance_rank`).
    If `importance_rank` is a tuple (and not a numpy array),
    a random rank grid of edge length given is used instead, shuffled using the
    random generator.
    This grid is shown using `plt.matshow` if `show_random_grid` is True.

    k is increased linearly from 0 to the number of pixels in the image over
    `num_iterations`.
    An ndarray of shape (num_iterations, *x.shape) is returned alongside the
    values of k used.
    """

    num_pixels = x.shape[1] * x.shape[2]

    if isinstance(importance_rank, tuple):
        random_res, random_gen = importance_rank
        random_importance = random_gen.permuted(
            np.floor(np.linspace(0, num_pixels, random_res ** 2))
        ).reshape(random_res, random_res)
        importance_rank = skimage.transform.resize(
            random_importance, output_shape=x.shape[1:],
            order=0, clip=False, preserve_range=True
        )
        if show_random_grid:
            plt.matshow(importance_rank)

    k_values = np.floor(
        np.linspace(0, num_pixels, num_iterations)
    ).astype(int)

    incrementally_deleted = np.zeros((num_iterations, *x.shape),
                                     dtype=x.dtype)
    for i, k in enumerate(k_values):
        x = delete_top_k_important(x, importance_rank, k, method)
        incrementally_deleted[i] = x

    return incrementally_deleted, k_values


def make_preds_df(
        model: torch.nn.Module,
        x: Float[np.ndarray, "batch_size channels height width"],
        index: t.Optional[t.List[float]] = None,
        columns: t.Optional[t.List[str]] = None,
):
    model.eval()
    x = torch.from_numpy(x).to(next(model.parameters()).device)
    preds = model(x).softmax(dim=-1)

    if index is None:
        index = range(preds.shape[0])
    if columns is None:
        columns = [f"cls_{i}" for i in range(preds.shape[1])]

    df = pd.DataFrame(preds.numpy(force=True), index=index, columns=columns)
    return df
