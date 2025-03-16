import typing as t

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from skimage import color


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
        _ = plt.subplot(5, 5, i + 1)
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
    annotation_colour = "green" if predicted_label == true_label else "red"

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                         100 * np.max(predictions_array),
                                         true_label),
               color=annotation_colour)


def plot_pred_bars(predictions_array: torch.Tensor, true_label):
    x_range = range(predictions_array.size(0))
    plt.grid(False)
    plt.xticks([])
    plt.xticks(list(x_range), minor=True)
    plt.yticks([])
    this_plot = plt.bar(x_range, predictions_array, color="grey")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color("orange")
    this_plot[true_label].set_color("green")


def show_image(
        x: t.Union[torch.Tensor, np.ndarray],
        is_normalised: bool = True,
        grayscale: bool = False,
        **kwargs
):
    """
    Show an image (or n images) using matplotlib.
    If `is_normalised` is True, the image is assumed to be in the range [-1, 1]
    and will be scaled to [0, 1] for display.

    The shape of x (a numpy array OR torch Tensor) can be:

        - (c, h, w) OR (h, w, c) for a single image
        - (n, c, h, w) OR (n, h, w, c) for n images

    The number of channels (c) should either be 1 (grayscale) or 3 (RGB).

    kwargs (e.g. vmin, vmax) are passed to `plt.imshow`.
    """

    if isinstance(x, torch.Tensor):
        x = x.numpy(force=True)

    if x.ndim == 3:
        if x.shape[0] in (1, 3):
            x = einops.rearrange(x, "c h w -> h w c")

    elif x.ndim == 4:
        if x.shape[-1] in (1, 3):
            x = einops.rearrange(x, "n h w c -> n c h w")

        # todo: use torchvision image grid instead to add padding
        x = einops.rearrange(x, "n c h w -> h (n w) c")
        # x = torchvision.utils.make_grid(x, nrow=x.shape[0])
    else:
        raise ValueError(f"Invalid shape for x: {x.shape}")

    if is_normalised:
        x = (x + 1) / 2  # un-normalise from [-1, 1] to [0, 1]

    if grayscale:
        x = color.rgb2gray(x)
        kwargs["cmap"] = "gray"

    plt.imshow(x, **kwargs)
    plt.axis("off")


def show_ms_images(
        x: Float[t.Union[torch.Tensor, np.ndarray], "n_samples channels height width"],
        normalisation_type: t.Literal["all", "each", "img", "channel", "none"] = "all",
):
    """
    Show all channels of multiple multi-spectral images (n, c, h, w) in a nice labelled plot.

    Performs normalisation based on `normalisation_type`:
        - "all": use min/max across all images and channels to normalise images
        - "each": use min/max per image and channel (i.e. each plot is normalised separately)
        - "img": min/max per image (across all channels in a row)
        - "channel": min/max per channel (across all plots in a column)
        - "none": no normalisation (let plt.imshow do its thing)
    """

    n_imgs, n_channels, h, w = x.shape

    _, axes = plt.subplots(nrows=n_imgs, ncols=n_channels, figsize=(8, 8))

    if n_imgs == 1:
        axes = np.array([axes])

    for i in range(n_imgs):
        for c in range(n_channels):
            plt.subplot(n_imgs, n_channels, i * n_channels + c + 1)

            norm_args = dict()
            if normalisation_type == "all":
                norm_args = dict(vmin=x.min(), vmax=x.max())
            elif normalisation_type == "each":
                norm_args = dict(vmin=x[i, c].min(), vmax=x[i, c].max())
            elif normalisation_type == "img":
                norm_args = dict(vmin=x[i].min(), vmax=x[i].max())
            elif normalisation_type == "channel":
                norm_args = dict(vmin=x[:, c].min(), vmax=x[:, c].max())

            show_image(x[i, [c]], is_normalised=False, cmap="viridis", **norm_args)

    for ax, col_name in zip(axes[0], [f"{c}" for c in range(n_channels)]):
        ax.set_title(col_name)
    for ax, row_name in zip(axes[:, 0], [f"{c}" for c in range(n_imgs)]):
        ax.annotate(row_name, xy=(0, 0.5), xytext=(-1, 0),
                    xycoords="axes fraction", textcoords="offset fontsize",
                    size="large", ha="right", va="center")

    plt.tight_layout()


def visualise_importance(
        x: Float[t.Union[np.ndarray, torch.Tensor], "n_samples height width channels"],
        importance_rank: Int[np.ndarray, "n_samples height width"],
        alpha: float = 0.2,
        with_colorbar: bool = True,
        **kwargs,
):
    """
    Overlay (with transparency `alpha`) the importance rank over the image with
    a colour bar.
    """

    show_image(x, grayscale=True, **kwargs)
    rank_img = einops.rearrange(importance_rank, "n h w -> h (n w)")
    plt.imshow(rank_img, alpha=alpha, cmap="plasma_r", **kwargs)

    if with_colorbar:
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
