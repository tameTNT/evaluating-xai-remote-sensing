import numpy as np
from jaxtyping import Float

from helpers import log
from xai import Explainer

logger = log.get_logger("main")


class Similarity:
    def __init__(
            self,
            x1: Float[np.ndarray, "n_samples height width channels"],
            x2: Float[np.ndarray, "n_samples height width channels"],
    ):
        assert x1.shape == x2.shape, "x1 and x2 to compare should have the same shape"
        self.shape = x1.shape

        self.x1 = x1
        self.x2 = x2

    def __call__(
            self,
            l2_args: dict = None,
    ) -> dict:
        if l2_args is None:
            l2_args = {"normalise": True}

        metrics = {
            "l2_distance": self.l2_distance(**l2_args)
        }
        return metrics

    def l2_distance(self, normalise: bool = True) -> Float[np.ndarray, "n_samples"]:
        n, h, w = self.shape

        x1: np.ndarray = self.x1
        x2: np.ndarray = self.x2

        if normalise:
            x1_range = x1.max() - x1.min()
            x2_range = x2.max() - x2.min()

            if x1_range != 0:
                x1 = (x1 - x1.min()) / x1_range
            elif x1.max() > 0:  # Array is all same constant. If all 0s leave.
                # If greater than 0, set to 1
                x1 = np.ones(x1.shape)
            else:
                x1 = np.zeros(x1.shape)

            if x2_range != 0:
                x2 = (x2 - x2.min()) / x2_range
            elif x2.max() > 0:
                x2 = np.ones(x2.shape)
            else:
                x2 = np.zeros(x2.shape)

        # flatten image dimensions (HxW)
        # then sum squared differences and take average over batch_size (first dim)
        # This is the L2 (Euclidian) distance.
        p = 2  # =2 for L2 distance
        return np.power(np.power(np.abs(x1 - x2).reshape(n, -1), p).sum(-1), 1 / p) / (h * w)

    def __bool__(self):
        if self.x1 is None or self.x2 is None:
            return False
        return True


class Co12Metric:
    def __init__(self, exp: Explainer):
        self.exp = exp

    def evaluate(self, method: str):
        logger.info(f"Evaluating {self.__class__.__name__} (via {method}) "
                    f"of {self.exp.__class__.__name__} "
                    f"for {self.exp.model.__class__.__name__}.")
