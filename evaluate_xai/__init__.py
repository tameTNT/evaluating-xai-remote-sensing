import numpy as np
from jaxtyping import Float, Int
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim

from helpers import log
from xai import Explainer

logger = log.get_logger("main")


class Similarity:
    def __init__(self, exp1: Explainer, exp2: Explainer):
        x1 = exp1.explanation
        x2 = exp2.explanation
        x1_rank = exp1.ranked_explanation
        x2_rank = exp2.ranked_explanation

        assert x1.shape == x2.shape, "explanations of exp1 and exp2 should have the same shape"
        self.shape = x1.shape

        self.x1: Float[np.ndarray, "n_samples height width"] = x1
        self.x2: Float[np.ndarray, "n_samples height width"] = x2
        self.x1_rank: Int[np.ndarray, "n_samples height width"] = x1_rank
        self.x2_rank: Int[np.ndarray, "n_samples height width"] = x2_rank

    def __call__(
            self,
            l2_normalise: bool = True,
            intersection_k: int = 5000,
    ) -> dict:
        metrics = {
            "l2_distance": self.l2_distance(normalise=l2_normalise),
            "spearman_rank": self.spearman_rank(),
            "top_k_intersection": self.top_k_intersection(k=intersection_k),
            "structural_similarity": self.structural_similarity(),
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

        # flatten image dimensions (HxW) then sum squared differences, averaging over image dimensions
        # This is the mean L2 (Euclidian) distance per pixel. Returned for each image in batch.
        p = 2  # =2 for L2 distance
        return np.power(np.power(np.abs(x1 - x2).reshape(n, -1), p).sum(-1), 1 / p) / (h * w)

    def spearman_rank(self) -> Float[np.ndarray, "n_samples"]:
        """
        Calculate the Spearman rank correlation between two sets of ranked
        images/explanations. That is, inputs should be rankings ranging from 0 to
        num_pixels-1.

        Optionally also shows the scatter plots of the rankings for each image.

        A score of 1 indicates perfect correlation with the rankings agreeing
        entirely on the importance of pixels.
        A score of -1 indicates perfect anti-correlation with the rankings saying
        exactly the opposite of each other.
        A score of 0 indicates no correlation at all between images/explanations.
        """

        spearman_coeffs = np.zeros(0)

        for i, (im1, im2) in enumerate(zip(self.x1_rank, self.x2_rank)):
            im1_flat, im2_flat = im1.flatten(), im2.flatten()
            spearman_coeffs = np.append(spearman_coeffs, spearmanr(im1_flat, im2_flat).statistic)

        return spearman_coeffs

    def top_k_intersection(self, k: int = 5000) -> Float[np.ndarray, "n_samples"]:
        """
        Calculate the percentage intersection of the top-k ranked pixels between
        two sets of ranked images/explanations. That is, inputs should be rankings
        ranging from 0 to num_pixels-1.

        A score of 1 indicates perfect intersection (the top k are the same in both
        images/explanations).
        A score of 0 indicates no intersection (the top k are completely different
        in both images/explanations).
        """

        x1_flat = self.x1_rank.reshape(self.shape[0], -1)
        x2_flat = self.x2_rank.reshape(self.shape[0], -1)
        intersection = np.sum(np.logical_and(x1_flat < k, x2_flat < k), axis=1) / k

        return intersection

    def structural_similarity(self) -> Float[np.ndarray, "n_samples"]:
        """
        Calculate the structural similarity index (SSIM) between two sets of ranked
        images/explanations. That is, inputs should be rankings ranging from 0 to
        num_pixels-1.

        A score of 1 indicates perfect structural similarity.
        A score of 0 indicates no structural similarity between the two
        images/explanations.
        """

        ssims = np.zeros(0)
        for im1, im2 in zip(self.x1_rank, self.x2_rank):
            ssims = np.append(ssims, ssim(im1, im2, data_range=im2.max()))

        return ssims

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
