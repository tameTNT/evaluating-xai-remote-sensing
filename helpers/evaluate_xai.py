import typing as t

import numpy as np
import pandas as pd
import scipy
import skimage
import torch
from jaxtyping import Float, Int
from tqdm.autonotebook import tqdm

from helpers import utils


def reset_child_params(model: torch.nn.Module):
    """
    Reset all parameters of the model to their defaults **inplace**.
    Adapted from https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        reset_child_params(layer)


DELETION_METHODS = t.Union[float, int, np.random.Generator, t.Literal["blur", "inpaint", "nn", "shuffle"]]


def delete_top_k_important(
        x: Float[np.ndarray, "channels height width"],
        importance_rank: t.Union[Int[np.ndarray, "height width"], t.Tuple[int, np.random.Generator]],
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
    - If method is 'shuffle', randomly perturb the top k pixels by shuffling
        them with the other pixels in the image.

    If importance_rank is a tuple of an int and a np.random.Generator, a random
    ranking, with grid element size of the first argument, is generated instead.
    """

    masked_img = x.copy()

    if isinstance(importance_rank, tuple):
        num_pixels = x.shape[-2] * x.shape[-1]
        random_res, random_gen = importance_rank

        random_importance = random_gen.permuted(
            np.floor(np.linspace(0, num_pixels, random_res ** 2))
        ).reshape(random_res, random_res)

        importance_rank = skimage.transform.resize(
            random_importance, output_shape=x.shape[1:],
            order=0, clip=False, preserve_range=True
        )

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

    elif method == "shuffle":
        pixels_to_scramble = masked_img[:, top_k_mask]

        # === Shuffle among all pixels ===
        # np.random.shuffle shuffles along first axis only
        # so transpose to shuffle RGB pixels (and not colour channels)
        np.random.shuffle(pixels_to_scramble.transpose(1, 0))

        # === Shuffle along neighbourhood *lines* ===
        # neighbourhood_size = 20
        # for i in range(0, pixels_to_scramble.shape[1], neighbourhood_size):
        #     np.random.shuffle(
        #         pixels_to_scramble.transpose(1, 0)[i:i + neighbourhood_size]
        #     )

        masked_img[:, top_k_mask] = pixels_to_scramble

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
    a random rank grid of edge length (1st arg) given is used instead, shuffled
    using the random generator provided (a consistent one per call if generator
    is None).
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

                x_j = delete_top_k_important(x, (random_res, random_gen), k, method)

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
    model_device = utils.get_model_device(model)
    x = torch.from_numpy(x)

    preds = []
    for minibatch in utils.make_device_batches(x, max_batch_size, model_device):
        batch_preds = model(minibatch).softmax(dim=-1).detach().cpu()
        preds.append(batch_preds)
    preds = torch.cat(preds, dim=0)

    if index is None:
        index = range(preds.shape[0])
    if columns is None:
        columns = [f"cls_{i}" for i in range(preds.shape[1])]

    df = pd.DataFrame(preds.numpy(force=True), index=index, columns=columns)
    return df


def perturb(
        x: Float[torch.Tensor, "batch_size channels height width"],
        degree: float,
) -> Float[torch.Tensor, "batch_size channels height width"]:
    """
    Perturb `x` (images) by adding Gaussian noise to it to the degree given.
    Output is clamped to [-1, 1].
    """
    return (x + degree * torch.randn_like(x)).clamp(-1, 1)


def pred_change_df(
        model: torch.nn.Module,
        original: Float[torch.Tensor, "batch_size channels height width"],
        perturbed: Float[torch.Tensor, "batch_size channels height width"],
        max_batch_size: int = 32,
) -> pd.DataFrame:
    """
    Returns a dataframe containing 3 columns for each input image index:

        - `original_pred`: the prediction class index on the original image
        - `change_in_confidence`: the change in confidence from the original
          prediction on the perturbed image
        - `perturbed_pred`: the prediction class index on the perturbed image

    The predictions are made using the model provided with batches of size the
    given `max_batch_size`.
    """

    model.eval()
    model_device = utils.get_model_device(model)

    preds = []
    for x in [original, perturbed]:
        preds.append([])
        mb_gen = utils.make_device_batches(x, max_batch_size, model_device)
        for minibatch in mb_gen:
            batch_preds = model(minibatch).softmax(dim=-1).detach().cpu()
            preds[-1].append(batch_preds)
        preds[-1] = torch.cat(preds[-1], dim=0)

    df = pd.DataFrame(
        preds[0].argmax(-1),
        columns=["original_pred"],
    )
    df["change_in_confidence"] = torch.gather(
        preds[1] - preds[0], dim=1, index=preds[0].argmax(-1, keepdim=True)
    )
    df["perturbed_pred"] = preds[1].argmax(-1)

    return df


def compactness(
        x: Float[np.ndarray, "batch_size height width"],
        threshold: float,
) -> Float[np.ndarray, "batch_size"]:
    """
    Calculate the compactness of each image in `x` given a threshold ([0, 1]).
    The compactness is defined as the proportion of the number of pixels
    (normalised per image) above the threshold. Note that the absolute value of
    the pixels is used since negative values also contribute to visual clutter
    when plotted.
    """

    x = np.abs(x)  # negative values also contribute to visual clutter
    x /= x.max(axis=(1, 2), keepdims=True)  # normalise

    return np.sum(x > threshold, axis=(1, 2)) / x[0].size
