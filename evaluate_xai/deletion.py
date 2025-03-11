import typing as t

import torch
import numpy as np
from jaxtyping import Int, Float
import scipy
import skimage

DELETION_METHODS = t.Union[float, int, np.random.Generator, t.Literal["blur", "inpaint", "nn", "shuffle"]]


def delete_top_k_important(
        x: Float[torch.Tensor, "n_samples channels height width"],
        importance_ranking: Int[np.ndarray, "n_samples height width"],
        k: int,
        method: DELETION_METHODS,
) -> Float[np.ndarray, "n_samples channels height width"]:
    """
    'Delete' the top k most important pixels (as specified by `importance_rank`)
    in `x` (values in [-1,1]) using one of several methods
    determined by the `method` parameter:

    - If method is a float or int, set the top k pixels to this value.
    - If method is 'blur', apply a Gaussian blur (sigma=5) to the top k pixels.
    - If method is a numpy random generator (`np.random.Generator`), add
        Gaussian noise to the top k pixels.
    - If method is 'inpaint', replace/inpaint the top k pixels using
        `skimage.restoration.inpaint_biharmonic`. This is by far the most
        time-consuming method.
    - If method is 'nn', replace the top k pixels using nearest neighbour
        interpolation.
    - If method is 'shuffle', randomly perturb the top k pixels by shuffling
        them with the other pixels in the image. This is the 2nd slowest method.

    If importance_rank is a tuple of an int and a np.random.Generator, a random
    ranking, with grid element size of the first argument, is generated instead.
    """

    masked_imgs = x.numpy(force=True)
    n_channels = x.shape[1]

    top_k_mask = importance_ranking < k
    top_k_mask = np.expand_dims(top_k_mask, 1).repeat(n_channels, 1)  # mask across all input channels
    # top_k_mask.shape = (n_samples, n_channels, height, width)
    target_regions: np.ndarray = masked_imgs[top_k_mask]

    # We need at least one 'known' pixel (where the top_k_mask is False)
    # for the inpaint and nn methods. Set the central pixel of every channel
    if method in ["inpaint", "nn"]:
        for i, img_mask in enumerate(top_k_mask):  # type: int, np.ndarray
            if np.sum(~img_mask) < 1:
                for c in range(n_channels):
                    top_k_mask[i, c, img_mask.shape[0] // 2, img_mask.shape[1] // 2] = False

    if isinstance(method, (float, int)):
        masked_imgs[top_k_mask] = np.clip(method, -1, 1)

    elif method == "blur":
        masked_imgs[top_k_mask] = scipy.ndimage.gaussian_filter(
            target_regions, sigma=5,
        )

    elif isinstance(method, np.random.Generator):
        noise = method.normal(size=target_regions.shape) / 5
        masked_imgs[top_k_mask] = np.clip(target_regions + noise, -1, 1)

    elif method == "inpaint":
        for i in range(len(masked_imgs)):
            mask = np.zeros_like(importance_ranking[i])
            mask[top_k_mask[i, 0]] = 1  # indicate unknown pixels
            masked_imgs[i] = skimage.restoration.inpaint_biharmonic(
                masked_imgs[i], mask, channel_axis=0
            )

    elif method == "nn":
        masked_imgs[top_k_mask] = np.nan
        for i in range(len(masked_imgs)):
            # Adapted from https://stackoverflow.com/a/27745627/7253717
            filled_ind = scipy.ndimage.distance_transform_edt(
                np.isnan(masked_imgs[i]), return_distances=False, return_indices=True)
            masked_imgs[i] = masked_imgs[i][tuple(filled_ind)]

    elif method == "shuffle":
        for i in range(len(masked_imgs)):
            if target_regions.size == 0:
                continue  # skip if no pixels to shuffle
            pixels_to_scramble = masked_imgs[i, :, top_k_mask[i, 0]]

            # === Shuffle among all pixels ===
            # np.random.shuffle shuffles along first axis only (positions)
            # (don't shuffle colour channels).
            # NB: pixels_to_scramble.shape = (k, n_channels)
            np.random.shuffle(pixels_to_scramble)

            # === Shuffle along neighbourhood *lines* ===
            # neighbourhood_size = 20
            # for i in range(0, pixels_to_scramble.shape[1], neighbourhood_size):
            #     np.random.shuffle(
            #         pixels_to_scramble.transpose(1, 0)[i:i + neighbourhood_size]
            #     )

            masked_imgs[i, :, top_k_mask[i, 0]] = pixels_to_scramble

    return masked_imgs


def generate_random_ranking(
        shape: Int[np.ndarray, "height width"],
        resolution: int = 16,
        random_seed: int = 42
) -> Int[np.ndarray, "height width"]:
    h, w = shape

    random_gen = np.random.default_rng(random_seed)
    random_importance = random_gen.permuted(
        np.floor(np.linspace(0, h*w, resolution ** 2))
    ).reshape(resolution, resolution)

    return skimage.transform.resize(
        random_importance, output_shape=(h, w),
        order=0, clip=False, preserve_range=True
    )
