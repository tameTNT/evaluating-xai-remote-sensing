import typing as t

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float

import helpers
from . import Co12Property

logger = helpers.log.main_logger


class Compactness(Co12Property):
    """
    "Compactness considers the size of the explanation and is motivated by human cognitive capacity
    limitations. Explanations should be sparse, short and not redundant to avoid presenting an
    explanation that is too big to understand."
    """
    def evaluate(
            self,
            method: t.Literal["threshold"],
            **kwargs,
    ) -> Float[np.ndarray, "n_samples"]:
        super().evaluate(method, **kwargs)

        if method == "threshold":
            return self._threshold(**kwargs)
        else:
            raise ValueError(f"Unknown evaluation method: {method}")

    def _threshold(
            self,
            threshold: float = 0.5,
            **kwargs,
    ) -> Float[np.ndarray, "n_samples"]:
        """
        Calculate the compactness of each explanation relative to a given threshold ([0, 1]).
        The compactness is defined as the proportion of pixels (normalised per image)
        equal to or below the threshold, τ. Note that the absolute value of
        the pixels is used since negative values also contribute to visual clutter
        when plotted.
        """

        abs_exp = np.abs(self.exp.explanation)  # negative values also contribute to visual clutter
        norm_exp: np.ndarray = abs_exp / abs_exp.max(axis=(1, 2), keepdims=True)

        # check for any explanations with NaN values (from division by 0)
        nan_mask = np.isnan(norm_exp).any(axis=(1, 2))
        if nan_mask.any():
            logger.warning(f"Normalised explanation (at idxs={np.where(nan_mask)[0]}) contains NaN values. "
                           f"Dropping these explanations.")
            norm_exp = norm_exp[~nan_mask]

        # nparray.size returns the number of elements so this divides by the number of pixels per image
        proportion_under_threshold = np.sum(norm_exp <= threshold, axis=(1, 2)) / norm_exp[0].size

        if self.visualise:
            above_threshold_only = norm_exp.copy()
            above_threshold_only[above_threshold_only <= threshold] = 0
            # add channel dimension for show_image's expected format: (n, c, h, w)
            helpers.plotting.show_image(np.expand_dims(above_threshold_only, 1), is_01_normalised=True, cmap="viridis")
            plt.title(f"Normalised explanation above threshold, τ={threshold}")
            plt.show()

        return proportion_under_threshold
