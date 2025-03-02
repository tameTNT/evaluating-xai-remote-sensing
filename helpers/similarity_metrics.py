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