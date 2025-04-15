import typing as t

import numpy as np
import torch
from jaxtyping import Float, Int
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim
from tqdm.autonotebook import tqdm
import einops
import matplotlib.pyplot as plt

import helpers
from xai import Explainer
from . import deletion

logger = helpers.log.main_logger
SIMILARITY_METRICS = t.Literal["l2_distance", "spearman_rank", "top_k_intersection", "structural_similarity"]


class Similarity:
    def __init__(self, exp1: Explainer, exp2: Explainer, mask: np.ndarray[bool] = None):
        """
        An object to calculate similarity metrics between two sets of explanations.
        Supports masking particular indices to ignore them in the similarity metrics
        (desirable for some evaluation metrics, e.g. continuity).
        Call the created instance to calculate and return the metrics in dictionary form.

        :param exp1: An Explainer object with populated explanation and ranked_explanation attributes.
        :param exp2: An Explainer object with populated explanation and ranked_explanation attributes.
        :param mask: An optional boolean mask indicating which indices to use for the similarity metrics.
          If None or all True, all samples are used. The actual indices used are stored in self.return_idxs.
          The indices hidden (i.e., idxs where mask==False) are stored in self.hidden_idxs.
        """

        x1 = exp1.explanation
        x2 = exp2.explanation

        assert x1.shape == x2.shape, "explanations of exp1 and exp2 should have the same shape"
        self.shape = x1.shape

        self.x1: Float[np.ndarray, "n_samples height width"] = x1
        self.x2: Float[np.ndarray, "n_samples height width"] = x2
        self.x1_rank: Int[np.ndarray, "n_samples height width"] = exp1.ranked_explanation
        self.x2_rank: Int[np.ndarray, "n_samples height width"] = exp2.ranked_explanation

        if mask is None:
            mask = np.full(x1.shape[0], True)
        self.return_idxs = np.where(mask)[0]
        self.hidden_idxs = np.where(~mask)[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, num_hidden_idxs={len(self.hidden_idxs)})"

    def __call__(
            self,
            l2_normalise: bool = True,
            intersection_k: int = 5000,
    ) -> dict[SIMILARITY_METRICS, Float[np.ndarray, "n_samples"]]:

        logger.info(f"Generating similarity metrics for {self}.")

        return {
            "l2_distance": self.l2_distance(normalise=l2_normalise)[self.return_idxs],
            "spearman_rank": self.spearman_rank()[self.return_idxs],
            "top_k_intersection": self.top_k_intersection(k=intersection_k)[self.return_idxs],
            "structural_similarity": self.structural_similarity()[self.return_idxs],
        }

    def l2_distance(self, normalise: bool = True) -> Float[np.ndarray, "n_samples"]:
        # todo: add docstring

        logger.debug(f"Calculating L2 distance for {self}.")
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

        # Flatten image dimensions (HxW) then sum squared differences, averaging over image dimensions
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

        logger.debug(f"Calculating Spearman rank for {self}.")

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

        logger.debug(f"Calculating Top-k-intersection for {self}.")

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

        logger.debug(f"Calculating structural similarity for {self}.")

        ssims = np.zeros(0)
        for im1, im2 in zip(self.x1_rank, self.x2_rank):
            ssims = np.append(ssims, ssim(im1, im2, data_range=im2.max()))

        return ssims


class Co12Metric:
    def __init__(self, exp: Explainer, max_batch_size: int = 32):
        self.exp = exp
        self.max_batch_size = max_batch_size
        self.visualise = False

        self.n_samples = self.exp.input.shape[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(exp={self.exp})"

    def evaluate(self, method: str, visualise: bool = False, **kwargs):
        logger.info(f"Evaluating {self.__class__.__name__} (via {method} with kwargs={kwargs}) "
                    f"of {self.exp.__class__.__name__} "
                    f"for {self.exp.model.__class__.__name__}.")
        self.visualise = visualise

    def run_model(
            self,
            x: Float[t.Union[np.ndarray, torch.Tensor], "n_samples channels height width"],
    ) -> Float[np.ndarray, "n_samples n_classes"]:
        """
        Run the model on the given input data x and return the softmax-ed output for all classes.
        """

        self.exp.model.eval()
        model_device = helpers.utils.get_model_device(self.exp.model)
        model_dtype = helpers.utils.get_model_dtype(self.exp.model)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(model_dtype)

        logger.info(f"Generating model predictions on new images (e.g. perturbed) "
                    f"for {self.__class__.__name__} with batch size {self.max_batch_size} "
                    f"for model {self.exp.model.__class__.__name__}.")
        preds = []
        for minibatch in tqdm(
            helpers.utils.make_device_batches(x, self.max_batch_size, model_device),
            total=np.ceil(x.shape[0] / self.max_batch_size).astype(int), unit="batch", ncols=110,
            desc=f"Predicting for {self.__class__.__name__}", leave=False,
            mininterval=5,
        ):
            batch_preds = self.exp.model(minibatch).softmax(dim=-1).detach().cpu()
            preds.append(batch_preds)

        return torch.cat(preds, dim=0).numpy(force=True)

    def get_sub_explainer(
            self,
            name: str,
            x: torch.Tensor,
            model: torch.nn.Module = None
    ) -> Explainer:
        if model is None:  # use the same model as the original explainer
            model = self.exp.model

        new_exp = self.exp.__class__(
            model, extra_path=self.exp.extra_path/name, attempt_load=x,
            batch_size=self.max_batch_size,  # new explainers should use the same batch size
        )
        if not new_exp.has_explanation_for(x):
            logger.info(f"No existing explanation found for provided samples "
                        f"(shape={x.shape}) in '{name}' "
                        f"(a sub-explainer for {self.__class__.__name__}). "
                        f"Generating a new one.")
            # Use the same kwargs as the original explainer
            new_exp.explain(x, **self.exp.kwargs)
        else:
            logger.info(f"Existing explanation found for provided samples "
                        f"(shape={x.shape}) in '{name}' "
                        f"(a sub-explainer for {self.__class__.__name__}).")
        return new_exp

    def compare_sub_explainer(
            self,
            sub_exp: Explainer,
            alpha: float = .2,
            title: str = "Explanation on original/perturbed input",
    ):
        stacked_samples = einops.rearrange(
            torch.stack([self.exp.input, self.exp.input]), "i n c h w -> n (i h) w c")
        stacked_explanations = einops.rearrange(
            np.stack([self.exp.ranked_explanation, sub_exp.ranked_explanation]),
            "i n h w -> n (i h) w")
        # noinspection PyUnboundLocalVariable
        helpers.plotting.visualise_importance(stacked_samples, stacked_explanations,
                                              alpha=alpha, with_colorbar=False)
        plt.title(title)
        plt.show()
