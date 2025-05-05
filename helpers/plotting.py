import typing as t

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import einops
import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid
from jaxtyping import Float, Int
from skimage import color


def show_image(
        x: t.Union[torch.Tensor, np.ndarray],
        is_01_normalised: bool = False,
        grayscale: bool = False,
        final_fig_size: tuple[float, float] = None,  # width, height
        imgs_per_row: int = 8,
        padding: int = 10,
        padding_value: int = 1,
        ax: plt.Axes = None,
        **kwargs
):
    """
    Show an image (or n images) using matplotlib.

    :param x: Input image or images (a numpy array OR torch Tensor) of shape:
        [(c, h, w) OR (h, w, c) for a single image] OR [(n, c, h, w) OR (n, h, w, c) for n images]
        The number of channels (c) should either be 13 or less.
        Image dimensions (h, w) should also be at least 14x14.
    :param is_01_normalised: If True, the image is assumed to be in the range [0, 1]
        and no rescaling is performed. Otherwise, assumes input is in range [-1, 1]. Default is False.
    :param grayscale: If True, the image is converted from RGB to grayscale skimage.color.rgb2gray().
        Default is False.
    :param final_fig_size: The final figure size in inches (width, height) to set for the plot.
        Default is None (no resizing).
    :param imgs_per_row: Number of images to show per row. Default is 8.
    :param padding: Padding between images in the grid. Default is 10.
    :param padding_value: Padding value to use between images. Default is 1 (usually white if RGB).
    :param ax: Matplotlib Axes object to plot on. If None (default), a new figure is created.
    :param kwargs: Keyword arguments (e.g. vmin, vmax) are passed on to `plt.imshow`.
        If c > 3, the function will call `show_ms_images` to show all channels.
        kwargs are then instead passed to `show_ms_images`.
    """

    if isinstance(x, torch.Tensor):
        x = x.numpy(force=True)

    if x.ndim == 3:
        if x.shape[0] <= 13:  # to allow for up to 13 band images (we assume images are at least 14x14)
            x = einops.rearrange(x, "c h w -> h w c")
        # channel is now final dimension
        if x.shape[-1] not in (1, 3):
            x = einops.rearrange(x, "h w c -> c h w")
            show_ms_images(x.reshape(1, *x.shape), **kwargs)
            return

    elif x.ndim == 4:
        if x.shape[-1] <= 13:  # same logic as above
            x = einops.rearrange(x, "n h w c -> n c h w")
        # channel is now second dimension
        if x.shape[1] not in (1, 3):
            if "normalisation_type" not in kwargs:
                kwargs["normalisation_type"] = "channel"
            show_ms_images(x, **kwargs)
            if final_fig_size:
                plt.gcf().set_size_inches(final_fig_size)
            return

        single_channel = False
        if x.shape[1] == 1:
            single_channel = True

        # Pad with white value (1) by default
        x = make_grid(torch.from_numpy(x), nrow=imgs_per_row, padding=padding, pad_value=padding_value).numpy()
        if single_channel:
            x = x[0][None,]  # reduce additional channels added by make_grid
        # x = einops.rearrange(x, "(n1 n2) c h w -> (n1 h) (n2 w) c", n2=8)
        x = einops.rearrange(x, "c h w -> h w c")

    else:
        raise ValueError(f"Invalid shape for x: {x.shape}")

    if not is_01_normalised:
        x = (x + 1) / 2  # un-normalise from [-1, 1] to [0, 1]

    if grayscale:
        x = color.rgb2gray(x)
        kwargs["cmap"] = "gray"

    plot_target = plt
    if ax is not None:
        plot_target = ax
    plot_target.imshow(x, **kwargs)
    plot_target.axis("off")
    if final_fig_size:
        plot_target.gcf().set_size_inches(final_fig_size)


def show_ms_images(
        x: Float[t.Union[torch.Tensor, np.ndarray], "n_samples channels height width"],
        normalisation_type: t.Literal["all", "each", "img", "channel", "none"] = "all",
        total_fig_size: tuple[float, float] = None,  # width, height
        img_stacking: t.Literal["horizontal", "vertical"] = "horizontal",
        show_indices: bool = True,
        img_label_str: str = "img",
):
    """
    Show all channels of multiple multi-spectral images x (with shape (n, c, h, w)) in a nice labelled plot.

    Performs normalisation based on `normalisation_type`:
        - "all": use min/max across all images and channels to normalise images
        - "each": use min/max per image and channel (i.e. each individual plot is normalised separately)
        - "img": min/max per image (across all plots in a column (if using horizontal stacking))
        - "channel": min/max per channel (across all plots in a row (if using horizontal stacking))
        - "none": no normalisation (let plt.imshow do its own thing for every image)

    :param x: Input images with shape (n, c, h, w)
    :param normalisation_type: Type of normalisation to apply (see above).
    :param total_fig_size: The final overall figure size in inches (width, height) to set for the plot.
        Default is None (no resizing).
    :param img_stacking: How to stack the images multispectral. Either "horizontal" (default) or "vertical".
        E.g. if "horizontal", each row is a channel and each column is an image.
    :param show_indices: If True (default), adds labels to each row and column indicating index.
    :param img_label_str: String to prefix the image index labels with. Default is "img".
        E.g. if img_stacking is "horizontal", each column will be labelled "img0", "img1", etc.
    """

    n_imgs, n_channels, h, w = x.shape

    if img_stacking == "horizontal":
        nrows, ncols = n_channels, n_imgs

        def index_calc(i1, i2):
            return i1 * ncols + i2 + 1

    elif img_stacking == "vertical":
        nrows, ncols = n_imgs, n_channels

        def index_calc(i1, i2):
            return i2 * ncols + i1 + 1

    else:
        raise ValueError(f"Invalid img_stacking: {img_stacking}. Must be 'horizontal' or 'vertical'.")

    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=total_fig_size)

    if n_channels == 1:
        axes = np.array([axes])

    for c in range(n_channels):
        for i in range(n_imgs):
            plt.subplot(nrows, ncols, index_calc(c, i))

            norm_args = dict()
            if normalisation_type == "all":
                norm_args = dict(vmin=x.min(), vmax=x.max())
            elif normalisation_type == "each":
                norm_args = dict(vmin=x[i, c].min(), vmax=x[i, c].max())
            elif normalisation_type == "img":
                norm_args = dict(vmin=x[i].min(), vmax=x[i].max())
            elif normalisation_type == "channel":
                norm_args = dict(vmin=x[:, c].min(), vmax=x[:, c].max())

            show_image(x[i, [c]], is_01_normalised=True, cmap="viridis", **norm_args)

    if show_indices:
        # Add label to each row using annotate
        for ax, row_name in zip(axes[:, 0], range(nrows)):  # type: plt.Axes, int
            row_label = img_label_str if img_stacking == "vertical" else ""
            ax.annotate(f"{row_label}{row_name}", xy=(0, 0.5), xytext=(-1, 0),
                        xycoords="axes fraction", textcoords="offset fontsize",
                        fontsize=12, ha="right", va="center")
        # Add label to each column via a title
        for ax, col_name in zip(axes[0], range(ncols)):  # type: plt.Axes, int
            col_label = img_label_str if img_stacking == "horizontal" else ""
            ax.set_title(f"{col_label}{col_name}", fontsize=12)

    # plt.tight_layout()


def visualise_importance(
        x: Float[t.Union[np.ndarray, torch.Tensor], "n_samples channels height width"],
        importance_rank: t.Union[Int[np.ndarray, "n_samples height width"],
                                 Float[np.ndarray, "n_samples height width"]],
        alpha: float = 0.7,
        with_colorbar: bool = True,
        band_idxs: list[int] = None,
        show_samples_separate: bool = False,
        **kwargs,
):
    """
    Visualise an importance heatmap for a set of images.
    Yellow indicates the most important/highest activation regions.
    Use `show_image` internally.

    :param x: Either a numpy ndarray or Tensor of shape (n, c, h, w) of images being explained.
        Displayed as a grayscale background image.
    :param importance_rank: A numpy ndarray of shape (n, h, w) of importance ranks/explanations.
        This can be floats (for a pure importance value) or integers (for a rank).
    :param alpha: The transparency of the explanation heatmap.
        1 shows only the heatmap and not x. Defaults to 0.7.
    :param with_colorbar: If True, a colour bar scale is shown. Defaults to True.
    :param band_idxs: If x is a multi-spectral image, specifies the channel indices of x
        to use to show the image. Defaults to None (use all bands for RGB images).
    :param show_samples_separate: If True, show the images in x and explanation
        heatmap separately in two rows of plots.
    :param kwargs: Any additional keyword arguments are passed to `show_image`,
        both when displaying x and the importance rank heatmap.
    """

    if x.shape[1] > 3:  # multi-spectral image
        if band_idxs is None:
            raise ValueError(f"bands must be specified for multi-spectral images. "
                             f"x.shape[1] = {x.shape[1]} > 3")
        else:
            x = x[:, band_idxs]

    if show_samples_separate:
        plt.subplots(2, 1)
        plt.subplot(2, 1, 1)
        show_image(x, grayscale=False, padding=20, **kwargs)
        alpha = 1.
        plt.subplot(2, 1, 2)
    else:
        show_image(x, grayscale=True, padding=20, **kwargs)

    if np.issubdtype(importance_rank.dtype, np.integer):  # ranked explanation from 0 to a high int
        ranked = True
        cmap = "plasma_r"  # yellow for minimum value (0 = most important)
        colourmap = plt.get_cmap(cmap, lut=(importance_rank.max()+1))  # lut specifies number of entries (0->max)
        colourmap_input = importance_rank
    else:  # floating point raw explanation
        ranked = False
        cmap = "plasma"    # yellow for maximum value
        colourmap = plt.get_cmap(cmap)
        colourmap_input = Normalize()(importance_rank)  # rescales as expected by colourmap(...)

    coloured_rank_img = colourmap(colourmap_input)[..., :3]  # remove added alpha channel

    show_image(coloured_rank_img, is_01_normalised=True, grayscale=False, padding=20,
               alpha=alpha, **kwargs)

    if with_colorbar:
        ranking_min, ranking_max = importance_rank.min(), importance_rank.max()
        num_ticks = 5
        ticks = np.linspace(ranking_min, ranking_max, num_ticks)
        if ranked:
            # To get a discrete colour bar, we need to set the boundaries and values
            boundaries = np.arange(ranking_max+1)
            values = np.arange(ranking_max)
        else:
            cb_res = 50
            boundaries = np.linspace(ranking_min, ranking_max, cb_res+1)
            values = np.linspace(ranking_min, ranking_max, cb_res)

        cb = plt.colorbar(
            ScalarMappable(cmap=colourmap),
            ticks=ticks, boundaries=boundaries, values=values,

            ax=plt.gca(),  # todo: this currently pulls the figure down, leaving whitespace at top of figure
            label=f"Importance{' Rank (0 = most important)' if cmap == 'plasma_r' else ''}",
            location="bottom",
            pad=0.02,   # move closer distance to image
            shrink=0.9,  # make smaller
            aspect=25,  # make thinner
        )
        # cb.ax.invert_yaxis()
        _ = cb.solids.set(alpha=1)
