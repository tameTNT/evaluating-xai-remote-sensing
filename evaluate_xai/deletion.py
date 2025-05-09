import typing as t

import torch
import numpy as np
from jaxtyping import Int, Float
import scipy
import skimage
import cv2

METHODS = t.Union[float, int, np.random.Generator, t.Literal["blur", "inpaint", "nn", "shuffle"]]


def delete_top_k_important(
        x: Float[torch.Tensor, "n_samples channels height width"],
        importance_ranking: Int[np.ndarray, "n_samples height width"],
        k: int,
        method: METHODS,
) -> Float[np.ndarray, "n_samples channels height width"]:
    """
    'Delete' the top k most important pixels (as specified by `importance_rank`)
    in `x` (values generally expected to be in [-1,1]) using one of several methods
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

    Returns a new numpy array with the same shape as `x`, where the top k pixels have been deleted.
    """

    masked_imgs = x.clone().numpy(force=True)
    n_channels = masked_imgs.shape[1]

    top_k_masks = importance_ranking < k
    top_k_masks = np.expand_dims(top_k_masks, 1).repeat(n_channels, 1)  # mask across all input channels
    # top_k_masks.shape = (n_samples, n_channels, height, width), True where deletion should be applied
    target_regions: np.ndarray = masked_imgs[top_k_masks]

    # We need at least one 'known' pixel (a pixel where top_k_masks is False)
    # for the inpaint and nn methods. Set the central pixel of every channel
    if method in ["inpaint", "nn"]:
        for i, mask in enumerate(top_k_masks):  # type: int, np.ndarray
            if np.sum(~mask) < 1:
                for c in range(n_channels):
                    top_k_masks[i, c, mask.shape[0] // 2, mask.shape[1] // 2] = False

    if isinstance(method, (float, int)):
        masked_imgs[top_k_masks] = method

    elif method == "blur":
        # masked_imgs[top_k_masks] = scipy.ndimage.gaussian_filter(
        #     target_regions, sigma=100,
        # )
        for i, (img, mask) in enumerate(zip(masked_imgs, top_k_masks)):
            # RMB: opencv expects channels to come last!
            blurred_img = cv2.GaussianBlur(img.transpose(1, 2, 0), (127, 127), 100)
            # blurred_img = cv2.blur(img.transpose(1, 2, 0), (128, 128))
            masked_imgs[i] = np.where(mask, blurred_img.transpose(2, 0, 1), img)

    elif isinstance(method, np.random.Generator):
        noise = method.normal(size=target_regions.shape) / 2
        masked_imgs[top_k_masks] = np.clip(target_regions + noise, -1, 1)

    elif method == "inpaint":
        for i in range(len(masked_imgs)):
            mask = np.zeros_like(importance_ranking[i])
            mask[top_k_masks[i, 0]] = 1  # indicate unknown pixels
            masked_imgs[i] = skimage.restoration.inpaint_biharmonic(
                masked_imgs[i], mask, channel_axis=0
            )

    elif method == "nn":
        masked_imgs[top_k_masks] = np.nan
        for i in range(len(masked_imgs)):
            # Adapted from https://stackoverflow.com/a/27745627/7253717
            filled_ind = scipy.ndimage.distance_transform_edt(
                np.isnan(masked_imgs[i]), return_distances=False, return_indices=True)
            masked_imgs[i] = masked_imgs[i][tuple(filled_ind)]

    elif method == "shuffle":
        for i in range(len(masked_imgs)):
            if target_regions.size == 0:  # no pixels in target_regions ndarray
                continue  # skip if no pixels to shuffle
            pixels_to_scramble = masked_imgs[i, :, top_k_masks[i, 0]]

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

            masked_imgs[i, :, top_k_masks[i, 0]] = pixels_to_scramble

    return masked_imgs


def generate_random_ranking(
        shape: tuple[int, int],
        resolution_d: int = 16,
        random_seed: int = 42
) -> Int[np.ndarray, "height width"]:
    """
    Generate a random importance ranking in a 2D grid form.
    The final grid has a 'resolution_d' as given by `resolution_d` (`d` in our final paper).

    :param shape: The final shape of the importance ranking.
        This should be the normally same as the input image being 'explained'.
    :param resolution_d: The number of squares per edge in the final grid.
        (This is the `d` parameter from our final paper).
    :param random_seed: Random seed used for generating the random grid. Defaults to 42.
    :return: An importance ranking in a 2D grid form of shape given by `shape`.
    """

    # futuretodo: add alternative method to do this via gaussian spotting to blend regions better?
    #  compared to current harsh edge blocks
    h, w = shape

    random_gen = np.random.default_rng(random_seed)
    random_importance = random_gen.permuted(
        np.floor(np.linspace(0, h * w, resolution_d ** 2)).astype(int)
    ).reshape(resolution_d, resolution_d)

    return skimage.transform.resize(
        random_importance, output_shape=(h, w),
        order=0, clip=False, preserve_range=True
    )
