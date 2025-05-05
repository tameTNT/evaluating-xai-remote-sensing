import typing as t

import numpy as np
import torch
from jaxtyping import Float, Int, Bool
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

import helpers
from xai import Explainer
from . import deletion

logger = helpers.log.main_logger
SIMILARITY_METRICS = t.Literal["l2_distance", "spearman_rank", "top_m_intersection", "structural_similarity"]


class Similarity:
    def __init__(self, exp1: Explainer, exp2: Explainer, mask: Bool[np.ndarray, "n_samples"] = None):
        """
        Creates an object to calculate similarity metrics between two sets of explanations (both of the same shape).
        Supports masking particular indices to ignore them in the similarity metrics
        (desirable for some evaluation metrics, e.g. target_sensitivity in contrastivity).
        Call the created instance to calculate and return the metrics in dictionary form (see __call__ method).

        :param exp1: An Explainer object with populated explanation and ranked_explanation attributes.
        :param exp2: An Explainer object with populated explanation and ranked_explanation attributes.
        :param mask: An optional mask of bools indicating which indices to use for the similarity metrics.
          If None or all True, all samples are used. The actual indices used are stored in self.return_idxs.
          The indices hidden (i.e. idxs where mask==False) are stored in self.hidden_idxs.
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
            intersection_m: int = 5000,
            show_scatter: bool = False,
    ) -> dict[SIMILARITY_METRICS, Float[np.ndarray, "n_samples"]]:
        """
        Calculate the similarity metrics between two sets of explanations.

        :param l2_normalise: If True, normalise the images to the range [0, 1] before calculating the L2 distance.
        :param intersection_m: The number of top-m pixels to use for the top-m intersection metric.
        :return: A dictionary with the similarity metrics as keys and the corresponding values for each sample.
        """

        logger.info(f"Generating similarity metrics for {self}.")

        return {
            "l2_distance": self.l2_distance(normalise=l2_normalise)[self.return_idxs],
            "spearman_rank": self.spearman_rank(show_scatter=show_scatter)[self.return_idxs],
            "top_m_intersection": self.top_m_intersection(m=intersection_m)[self.return_idxs],
            "structural_similarity": self.structural_similarity()[self.return_idxs],
        }

    def l2_distance(self, normalise: bool = True) -> Float[np.ndarray, "n_samples"]:
        """
        Calculate the L2 distance between self.x1 and self.x2.
        If normalise is True, normalise the images to the range [0, 1] before calculating the L2 distance.
        The L2 distance is calculated as the mean L2 (Euclidean) distance per pixel.

        A higher distance indicates a lower similarity.
        """

        logger.debug(f"Calculating L2 distance for {self}.")
        n, h, w = self.shape

        x1: np.ndarray = self.x1
        x2: np.ndarray = self.x2

        if normalise:
            x1_range = x1.max() - x1.min()
            x2_range = x2.max() - x2.min()

            if x1_range != 0:
                x1 = (x1 - x1.min()) / x1_range
            elif x1.max() > 0:  # Array is all the same constant. If all 0s leave.
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
        # This is the mean L2 (Euclidian) distance per pixel. Returned for each sample separately.
        p = 2  # =2 for L2 distance
        return np.power(np.power(np.abs(x1 - x2).reshape(n, -1), p).sum(-1), 1 / p) / (h * w)

    def spearman_rank(self, show_scatter=False) -> Float[np.ndarray, "n_samples"]:
        """
        Calculate the Spearman rank correlation between self.x1_rank and self.x2_rank.
        These should each be rankings ranging from 0 to num_pixels-1.

        Optionally also shows the scatter plots of the rankings for each image if show_scatter==True.

        A score of 1 indicates a perfect correlation with the rankings agreeing
        entirely on the importance of pixels.
        A score of -1 indicates perfect anti-correlation with the rankings saying
        exactly the opposite of each other.
        A score of 0 indicates no correlation at all (very low similarity) between images/explanations.
        """

        logger.debug(f"Calculating Spearman rank for {self}.")

        spearman_coeffs = np.zeros(0)

        for i, (im1, im2) in enumerate(zip(self.x1_rank, self.x2_rank)):
            rank1, rank2 = im1.flatten(), im2.flatten()
            scc = spearmanr(rank1, rank2).statistic
            spearman_coeffs = np.append(spearman_coeffs, scc)

            if show_scatter:
                plt.title(f"Image {i}: scc={scc:.3f}")
                plt.scatter(rank1, rank2)
                plt.plot(range(rank1.max()), "r--", label="scc$=+1$")
                plt.plot(range(rank1.max(), 0, -1), "y--", label="scc$=-1$")
                plt.xlim((0, rank1.max()))
                plt.ylim((0, rank2.max()))
                plt.legend()
                plt.show()

        return spearman_coeffs

    def top_m_intersection(self, m: int = 5000) -> Float[np.ndarray, "n_samples"]:
        """
        Calculate the percentage intersection of the top-m ranked pixels between
        two sets of ranked images/explanations. That is, inputs should be rankings
        ranging from 0 to num_pixels-1.

        A score of 1 indicates a perfect intersection (the top m pixels are the same in both
        images/explanations).
        A score of 0 indicates no intersection (the top m are completely different
        in both images/explanations).

        A lower intersection indicates a lower similarity.
        """

        logger.debug(f"Calculating Top-m intersection for {self}.")

        x1_flat = self.x1_rank.reshape(self.shape[0], -1)
        x2_flat = self.x2_rank.reshape(self.shape[0], -1)
        intersection_proportion = np.sum(np.logical_and(x1_flat < m, x2_flat < m), axis=1) / m

        return intersection_proportion

    def structural_similarity(self) -> Float[np.ndarray, "n_samples"]:
        """
        Calculate the structural similarity index (SSIM) between two sets of ranked
        images/explanations. That is, inputs should be rankings ranging from 0 to
        num_pixels-1.

        A score of 1 indicates perfect structural similarity.
        A score of 0 indicates no structural similarity between the two
        images/explanations.

        A lower SSIM indicates a lower similarity.
        """

        logger.debug(f"Calculating structural similarity for {self}.")

        ssims = np.zeros(0)
        for im1, im2 in zip(self.x1_rank, self.x2_rank):
            ssims = np.append(ssims, ssim(im1, im2, data_range=im2.max()))

        return ssims


class Co12Property:
    def __init__(self, exp: Explainer, batch_size: int = 32):
        """
        Base class for all Co12 Properties which implement evaluation metrics.

        This categorisation is based off the following paper:

            Meike Nauta, Jan Trienes, Shreyasi Pathak, Elisa Nguyen, Michelle Peters, Yasmin Schmitt,
            Jörg Schlötterer, Maurice van Keulen, and Christin Seifert. 2023.
            From Anecdotal Evidence to Quantitative Evaluation Methods:
            A Systematic Review on Evaluating Explainable AI.
            ACM Comput. Surv. 55, 13s, Article 295 (December 2023), 42 pages.
            https://doi.org/10.1145/3583558

        This is also the source of the definitions used for each property.

        :param exp: The explainer object with an explanation for a set of input images to evaluate.
        :param batch_size: The batch size to use when passing images to exp.model
        """

        self.exp = exp
        self.batch_size = batch_size

        self.visualise = False
        self.store_full_data = False
        self.full_data = dict()

        self.n_samples = self.exp.input.shape[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(exp={self.exp})"

    def evaluate(self, method: str, visualise: bool = False, store_full_data: bool = False, **kwargs):
        """
        Evaluates the Co12 property of the explainer object via a given method/metric.
        :param method: Method/Metric to use for evaluation.
        :param visualise: Whether to show appropriate visualisations for the metric if applicable.
        :param store_full_data: Whether to store additional data in self.full_data for the metric.
            Can use a lot of memory if this extra data are images so defaults to False.
        :param kwargs: Additional keyword arguments to pass to the particular method call.
            Must also be included in all overriding methods to match call signature.
        """
        logger.info(f"Evaluating {self.__class__.__name__} (via {method} with kwargs={kwargs}) "
                    f"of {self.exp.__class__.__name__} "
                    f"for {self.exp.model.__class__.__name__} (with store_full_data={store_full_data}).")
        self.visualise = visualise
        self.store_full_data = store_full_data

    def run_model(
            self,
            x: Float[t.Union[np.ndarray, torch.Tensor], "n_samples channels height width"],
    ) -> Float[np.ndarray, "n_samples n_classes"]:
        """
        Run exp.model on the given input data x (an ndarray or Tensor) and
        return the output for all classes (with softmax applied) on the cpu.
        """

        self.exp.model.eval()
        model_device = helpers.utils.get_model_device(self.exp.model)
        model_dtype = helpers.utils.get_model_dtype(self.exp.model)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(model_dtype)

        logger.info(f"Generating model predictions on new images (e.g. perturbed) "
                    f"for {self.__class__.__name__} with batch size {self.batch_size} "
                    f"for model {self.exp.model.__class__.__name__}.")
        preds = []
        for minibatch in tqdm(
            helpers.utils.make_device_batches(x, self.batch_size, model_device),
            total=np.ceil(x.shape[0] / self.batch_size).astype(int), unit="batch", ncols=110,
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
        """
        Create a new temporary explainer object - useful for some metrics which
        require the generation of new explanations for a new input or using a new model -
        and generates an explanation for the given input x (i.e. populate new_exp.explanation).

        :param name: The name of the new explainer. Used in extra_path arg for the new explainer.
        :param x: The input to use for the new explainer.
        :param model: The model to explain for the new explainer.
            If None, the same model as the original explainer is used.
        :return: A new Explainer object for input x and model.
        """

        if model is None:  # use the same model as the original explainer
            model = self.exp.model

        new_exp = self.exp.__class__(
            model, extra_path=self.exp.extra_path/name, attempt_load=x,
            batch_size=self.batch_size,  # new explainers should use the same batch size
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
            alpha: float = .7,
            title: str = "Explanation on original/perturbed input",
            ranked: bool = True,
    ):
        """
        Plot and compare the explanations of the Explainer objects self.exp and sub_exp.
        :param sub_exp: Explainer object to compare with self.exp.
        :param alpha: Transparency of the explanation over image in the plot.
        :param title: Title to use for the plot.
        :param ranked: If True (default) compare ranked explanations.
        """

        double_input = torch.concatenate([self.exp.input, self.exp.input])

        if ranked:
            exps = [self.exp.ranked_explanation, sub_exp.ranked_explanation]
        else:
            exps = [self.exp.explanation, sub_exp.explanation]
        stacked_explanations = np.concatenate(exps)

        # noinspection PyUnboundLocalVariable
        helpers.plotting.visualise_importance(double_input, stacked_explanations, imgs_per_row=self.exp.input.shape[0],
                                              alpha=alpha, with_colorbar=False)
        plt.title(title)
        plt.show()  # todo: change all instances of plt.show() in to instead return a figure object
