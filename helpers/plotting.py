import typing as t

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils
from jaxtyping import Float, Int
import pandas as pd


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
        x: t.Union[torch.Tensor, np.ndarray],
        **kwargs
):
    """
    Show an image (or n images) using matplotlib.
    The image is assumed to be normalised if it contains values < 0.
    Otherwise, it is expected to be in the range [0, 1].
    The shape of x (a numpy array OR torch Tensor) can be:

        - (c, h, w) OR (h, w, c) for a single image
        - (n, c, h, w) OR (n, h, w, c) for n images

    The number of channels (c) should either be 1 (grayscale) or 3 (RGB).

    kwargs are passed to `plt.imshow`.
    """

    if isinstance(x, torch.Tensor):
        x = x.numpy(force=True)

    if x.ndim == 3:
        if x.shape[0] in (1, 3):
            x = einops.rearrange(x, "c h w -> h w c")

    elif x.ndim == 4:
        if x.shape[-1] in (1, 3):
            x = einops.rearrange(x, "n h w c -> n c h w")

        x = einops.rearrange(x, "n c h w -> h (n w) c")
        # x = torchvision.utils.make_grid(x, nrow=x.shape[0])
    else:
        raise ValueError(f"Invalid shape for x: {x.shape}")

    if x.min() < 0:
        x = (x + 1) / 2  # un-normalise

    plt.imshow(x, **kwargs)
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


def make_deletions_plot(
        *args: pd.DataFrame,
        method_names: list[str] = None,
        return_aucs: bool = False,
        plot_class: str = None,
        plt_title: str = None
) -> t.Optional[dict[str, float]]:
    """
    Plots a line chart, where each line represents confidence over iterations
    of a different method (names specified in method names) and the x-axis is
    the number of deletion iterations, from multiple dataframes.
    The legend includes the Area under Curve metric for each method which is
    optionally returned as a dictionary in the format `method_name: auc`.

    Specify a `plot_class` to plot only that class, otherwise the first row's
    max value is used (i.e. the DL model's most confident prediction for no
    perturbations).
    """

    if method_names is None:
        method_names = [f"method_{i}" for i in range(len(args))]
    concatenated_df = pd.concat(
        {name: df for name, df in zip(method_names, args)},
        names=["method", "iteration"]
    )

    aucs = dict()

    fig, ax = plt.subplots()
    for method, data in concatenated_df.groupby(level=0):
        if plot_class is None:
            # get the first row's max value (assumed to be correct prediction)
            plot_class = data.idxmax(axis=1).iloc[0]

        # take cross-section of just relevant column, drop the method from index
        series = data.xs(plot_class, axis=1).reset_index(level=0, drop=True)
        # then plot a line for this method
        auc = np.trapz(series.to_numpy())
        series.plot(
            kind="line",
            xlabel="iterations", rot=45, xlim=(0, len(series)-1),
            ylabel="model confidence", ylim=(0, 1),
            label=f"{method} (AuC={auc:.4f})",
            legend=True, title=plt_title, ax=ax, grid=True,
        )
        aucs[method] = auc

    ax.axhline(0.5, color="black", linestyle="--")
    ax.text(
        1, 0.5, "decision boundary (ranking/compactness)",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )

    if return_aucs:
        return aucs
