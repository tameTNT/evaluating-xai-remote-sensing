import typing as t

import numpy as np
import pandas as pd
import scipy
import skimage
import torch
from jaxtyping import Float, Int
from tqdm.autonotebook import tqdm


def reset_child_params(model: torch.nn.Module):
    """
    Reset all parameters of the model to their defaults **inplace**.
    Adapted from https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        reset_child_params(layer)


# for SHAP values this is better evaluated with L2 (hamming doesn't make sense
# (not the same)); SSIM is for heatmaps
def pixel_l2_distance_per_label(
        x1: Float[np.ndarray, "batch_size num_labels height width channels"],
        x2: Float[np.ndarray, "batch_size num_labels height width channels"],
        normalise: bool = True,
) -> Float[torch.Tensor, "num_labels"]:
    """
    Calculate normalised mean L2 distance between two sets of images
    (`x1`, `x2`) per label over the batch size.
    Each image (H, W) is flattened (i.e. to H*W) and the L2 distance is
    calculated per element (normalised by default, after summing channels, to
    the range [0, 1]).
    """
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    batch_size, num_labels, h, w, _ = x1.shape

    # sum over colour channels before normalising
    x1 = torch.from_numpy(x1).sum(-1)
    x2 = torch.from_numpy(x2).sum(-1)

    if normalise:
        x1_range = x1.max() - x1.min()
        x2_range = x2.max() - x2.min()

        if x1_range != 0:
            x1 = (x1 - x1.min()) / x1_range
        elif x1.max() > 0:  # Array is all same constant. If all 0s leave.
            # If greater than 0, set to 1
            x1 = torch.ones_like(x1)
        else:
            x1 = torch.zeros_like(x1)

        if x2_range != 0:
            x2 = (x2 - x2.min()) / x2_range
        elif x2.max() > 0:
            x2 = torch.ones_like(x2)
        else:
            x2 = torch.zeros_like(x2)

    # flatten image dimensions (HxW)
    # then sum squared differences and take average over batch_size (first dim)
    # This is the L2 (Euclidian) distance per label.
    return (x1 - x2).reshape(batch_size, num_labels, -1).pow(2).sum(-1).mean(0) / (h * w)


def make_l2_distance_per_label_df(
        class_names: list[str], *args, **kwargs
) -> pd.DataFrame:
    """
    Returns the result of `pixel_l2_distance_per_label` function as a DataFrame.
    *args and **kwargs are passed to `pixel_l2_distance_per_label`.
    """
    l2_dist = pixel_l2_distance_per_label(*args, **kwargs)
    df = pd.DataFrame(l2_dist.numpy(force=True), index=class_names, columns=["L2 distance"])
    df.index.name = "Class"
    return df


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


DELETION_METHODS = t.Union[float, int, np.random.Generator, t.Literal["blur", "inpaint", "nn"]]


def delete_top_k_important(
        x: Float[np.ndarray, "channels height width"],
        importance_rank: Int[np.ndarray, "height width"],
        k: int,
        method: DELETION_METHODS,
) -> Float[np.ndarray, "channels height width"]:
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
        importance_rank: t.Union[
            t.Tuple[int, t.Union[np.random.Generator, None], int], Int[np.ndarray, "height width"]],
        num_iterations: int,
        method: DELETION_METHODS,
) -> t.Tuple[
    Float[np.ndarray, "num_iterations trials channels height width"],
    Int[np.ndarray, "num_iterations"]
]:
    """
    Iteratively delete the top k most important pixels (as specified by
    `importance_rank`).

    If `importance_rank` is a tuple (and not a numpy array),
    a random rank grid of edge length given is used instead, shuffled using the
    random generator provided (a consistent one per call if generator is None).
    Only if this random ranking is used, is the `trials` dim not 1 and is
    instead the third tuple argument of `importance_rank`.

    k is increased linearly from 0 (inclusive) to the total number pixels in the
    image over `num_iterations`.
    An ndarray of shape (num_iterations, *x.shape) is returned alongside the
    values of k used.
    """

    num_pixels = x.shape[1] * x.shape[2]

    is_random = False
    random_res, random_gen, num_trials = 0, None, 1
    if isinstance(importance_rank, tuple):
        is_random = True
        random_res, random_gen, num_trials = importance_rank

    k_values = np.floor(
        np.linspace(0, num_pixels, num_iterations + 1)  # +1 because of 0 step
    ).astype(int)

    incrementally_deleted = np.zeros((num_iterations + 1, num_trials, *x.shape),
                                     dtype=x.dtype)

    for i, k in tqdm(enumerate(k_values), total=len(k_values),
                     desc="Incrementally deleting important pixels"):
        if is_random:
            for j in range(num_trials):
                if random_gen is None:
                    random_gen = np.random.default_rng(j)

                random_importance = random_gen.permuted(
                    np.floor(np.linspace(0, num_pixels, random_res ** 2))
                ).reshape(random_res, random_res)
                importance_rank = skimage.transform.resize(
                    random_importance, output_shape=x.shape[1:],
                    order=0, clip=False, preserve_range=True
                )
                x_j = delete_top_k_important(x, importance_rank, k, method)
                x_j = x_j[np.newaxis, :]
                output = x_j if j == 0 else np.concatenate((output, x_j), axis=0)
        else:
            output = delete_top_k_important(x, importance_rank, k, method)
            # add repeats dim at front (just 1 for non-random)
            output = output[np.newaxis, :]

        incrementally_deleted[i] = output

    return incrementally_deleted, k_values


def make_preds_df(
        model: torch.nn.Module,
        x: Float[np.ndarray, "batch_size channels height width"],
        index: t.Optional[t.List[float]] = None,
        columns: t.Optional[t.List[str]] = None,
        max_batch_size: int = 32,
) -> pd.DataFrame:
    model.eval()
    model_device = next(model.parameters()).device

    preds = []
    # split into smaller batches to avoid memory issues
    for i in range(0, x.shape[0], max_batch_size):
        x_batch = torch.from_numpy(x[i:i + max_batch_size]).to(model_device)
        batch_preds = model(x_batch).softmax(dim=-1).detach().cpu()
        preds.append(batch_preds)
    preds = torch.cat(preds, dim=0)

    if index is None:
        index = range(preds.shape[0])
    if columns is None:
        columns = [f"cls_{i}" for i in range(preds.shape[1])]

    df = pd.DataFrame(preds.numpy(force=True), index=index, columns=columns)
    return df
