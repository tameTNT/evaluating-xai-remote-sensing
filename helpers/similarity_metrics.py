import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim
from jaxtyping import Float
import torch
import einops
import pandas as pd


# for SHAP values this is better evaluated with L2 (hamming doesn't make sense
# (not the same))
def pixel_l2_distance_per_label(
        x1: Float[np.ndarray, "batch_size height width channels num_labels"],
        x2: Float[np.ndarray, "batch_size height width channels num_labels"],
        normalise: bool = True,
) -> Float[torch.Tensor, "num_labels"]:
    """
    Calculate normalised mean L2 distance between two sets of images
    (`x1`, `x2`) per label over the batch size.

    Input arrays are expected to have the shape of (batch_size, height, width,
    channels, num_labels).

    Each image (H, W) is flattened (i.e. to H*W) and the L2 distance is
    calculated per element.
    Each array is normalised by default, after summing channels, to
    the range [0, 1].
    """
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    x1 = einops.rearrange(x1, "b h w c l -> b l h w c")
    x2 = einops.rearrange(x2, "b h w c l -> b l h w c")
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


def spearman_rank(
        x_original: Float[np.ndarray, "batch_size height width channels"],
        x_new: Float[np.ndarray, "batch_size height width channels"],
        show_plot: bool = False,
) -> Float[np.ndarray, "batch_size"]:
    """
    Calculate the Spearman rank correlation between two sets of ranked
    images/explanations. That is, inputs should be rankings ranging from 0 to
    num_pixels-1.

    Optionally also shows the scatter plots of the rankings for each image.

    A score of 1 indicates perfect correlation with the rankings agreeing
    entirely on the importance of pixels.
    A score of -1 indicates perfect anti-correlation with the rankings saying
    exactly the opposite of each other.
    A score of 0 indicates no correlation at all between images/explanations.
    """

    assert x_original.shape == x_new.shape, "x_original and x_new must have the same shape"

    num_x = x_original.shape[0]

    spearman_coeffs = np.zeros(0)

    if show_plot:
        fig, axes = plt.subplots(1, num_x, figsize=(10, 10))
    for i, (og, new) in enumerate(zip(x_original, x_new)):
        og_flat, new_flat = og.flatten(), new.flatten()
        spearman_coeffs = np.append(spearman_coeffs, spearmanr(og_flat, new_flat).statistic)

        if show_plot:
            ax = axes[i]
            ax.set_title(f"SCC={spearmanr(og_flat, new_flat).statistic:.3f}")
            ax.scatter(og_flat, new_flat)
            ax.set_xlabel("Ranking for original")
            ax.set_ylabel("Ranking for new")
            ax.plot(range(og.max()), "r--")
            ax.set_aspect("equal")

    if show_plot:
        fig.tight_layout()

    return spearman_coeffs


def top_k_intersection(
        x_original: Float[np.ndarray, "batch_size height width channels"],
        x_new: Float[np.ndarray, "batch_size height width channels"],
        k: int = 5000,
) -> Float[np.ndarray, "batch_size"]:
    """
    Calculate the percentage intersection of the top-k ranked pixels between
    two sets of ranked images/explanations. That is, inputs should be rankings
    ranging from 0 to num_pixels-1.

    A score of 1 indicates perfect intersection (the top k are the same in both
    images/explanations).
    A score of 0 indicates no intersection (the top k are completely different
    in both images/explanations).
    """

    assert x_original.shape == x_new.shape, "x_original and x_new must have the same shape"

    originals_flattened = x_original.reshape(x_original.shape[0], -1)
    new_flattened = x_new.reshape(x_new.shape[0], -1)
    intersection = np.sum(np.logical_and(originals_flattened < k, new_flattened < k), axis=1) / k

    return intersection


def structural_similarity(
        x_original: Float[np.ndarray, "batch_size height width channels"],
        x_new: Float[np.ndarray, "batch_size height width channels"],
) -> Float[np.ndarray, "batch_size"]:
    """
    Calculate the structural similarity index (SSIM) between two sets of ranked
    images/explanations. That is, inputs should be rankings ranging from 0 to
    num_pixels-1.

    A score of 1 indicates perfect structural similarity.
    A score of 0 indicates no structural similarity between the two
    images/explanations.
    """

    ssims = np.zeros(0)
    for og, new in zip(x_original, x_new):
        ssims = np.append(ssims, ssim(og, new, data_range=new.max()))

    return ssims
