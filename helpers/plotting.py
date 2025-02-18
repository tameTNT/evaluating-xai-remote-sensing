import typing as t

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float, Int


def show_sample_of_incorrect(
        idx_where_wrong,
        dataset,
        dataset_classes,
        test_predictions,
        test_labels
):
    np.random.seed(69)
    random_wrong_idxs = np.random.choice(idx_where_wrong, size=25, replace=False)
    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    for i, idx in enumerate(random_wrong_idxs):
        ax = plt.subplot(5, 5, i + 1)
        plt.title(f"'{dataset_classes[int(test_predictions[idx])]}'"
                  f"\n({dataset_classes[int(test_labels[idx])]})")
        plt.imshow(einops.rearrange(dataset[idx]["image"], "c h w -> h w c"))
        plt.axis("off")

    return fig


def plot_image_with_annotation(
        predictions_array,
        true_label,
        img
):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    img = np.transpose(img, (1, 2, 0))  # move colour channel to end
    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    color = "green" if predicted_label == true_label else "red"

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                         100 * np.max(predictions_array),
                                         true_label),
               color=color)


def plot_pred_bars(predictions_array: torch.Tensor, true_label):
    x_range = range(predictions_array.size(0))
    plt.grid(False)
    plt.xticks([])
    plt.xticks(list(x_range), minor=True)
    plt.yticks([])
    thisplot = plt.bar(x_range, predictions_array, color="grey")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("orange")
    thisplot[true_label].set_color("green")


def show_image(
        img: t.Union[Float[torch.Tensor, "channels height width"],
        Float[np.ndarray, "channels height width"]],
        **kwargs
):
    """
    Show an image using matplotlib.
    The image is expected to be normalised (i.e. in the range [-1, 1])
    and with dimensions (C, H, W).

    kwargs are passed to `plt.imshow`.
    """

    if isinstance(img, torch.Tensor):
        img = img.numpy(force=True)

    if img.min() < 0:
        img = (img + 1) / 2  # un-normalise
    img = np.transpose(img, (1, 2, 0))  # move colour channel to end
    plt.imshow(img, **kwargs)
    plt.axis("off")


def visualise_importance(
        x: Float[np.ndarray, "batch_size height width channels"],
        importance_rank: Int[np.ndarray, "batch_size height width"],
        alpha: float = 0.2,
):
    """
    Overlay (with transparency `alpha`) the importance rank over the image with
    a colour bar.
    """

    show_image(x)
    plt.imshow(importance_rank, alpha=alpha, cmap="jet_r")

    cb = plt.colorbar(label="Importance Rank (0 = most important)")
    cb.ax.invert_yaxis()
    _ = cb.solids.set(alpha=1)
