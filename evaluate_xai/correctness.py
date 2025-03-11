import copy
import typing as t
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import torch
from jaxtyping import Int, Float
from tqdm.autonotebook import tqdm

import helpers
from xai import Explainer
from . import Co12Metric, Similarity

logger = helpers.log.get_logger("main")

DELETION_METHODS = t.Union[float, int, np.random.Generator, t.Literal["blur", "inpaint", "nn", "shuffle"]]


class Correctness(Co12Metric):
    # todo: add docstrings
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(
            self,
            method: t.Literal["model_randomisation", "incremental_deletion"],
            **kwargs,
    ) -> t.Union[Similarity, dict]:
        super().evaluate(method, **kwargs)

        if method == "model_randomisation":
            random_exp = self._model_randomisation()
            return Similarity(self.exp, random_exp)
        elif method == "incremental_deletion":
            return self._incremental_deletion(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _model_randomisation(self) -> Explainer:
        device = helpers.utils.get_model_device(self.exp.model)
        randomised_model = copy.deepcopy(self.exp.model).to(device)
        self.reset_child_params(randomised_model)
        randomised_model.eval()

        original_explainer = self.exp.__class__
        exp_for_randomised_model = original_explainer(
            randomised_model, Path(f"randomised_{self.exp.__class__.__name__}"), attempt_load=self.exp.attempt_load,
        )

        if not exp_for_randomised_model.has_explanation_for(self.exp.input):
            # only generate explanation if no existing one
            logger.info(f"No existing explanation for self.exp.input in exp_for_randomised_model. "
                        f"Generating a new one.")
            exp_for_randomised_model.explain(self.exp.input, **self.exp.kwargs)
        else:
            logger.info(f"Existing explanation found for self.exp.input in exp_for_randomised_model.")

        return exp_for_randomised_model

    def _incremental_deletion(
            self,
            iterations: int = 30,
            n_random_ranks: int = 16,
            random_seed: int = 42,
            deletion_method: DELETION_METHODS = "nn",
            visualise: bool = False,
    ) -> dict[t.Literal["informed", "random"], Float[np.ndarray, "n_samples"]]:

        n_samples = self.exp.input.shape[0]
        image_shape = self.exp.input.shape[1:]
        imgs_with_deletions, k_values = self.incrementally_delete(
            self.exp.ranked_explanation, iterations, deletion_method
        )
        if visualise:
            helpers.plotting.show_image(
                einops.rearrange(imgs_with_deletions, "n i c h w -> (n h) (i w) c"),
            )
            plt.tight_layout()
            plt.show()

        flattened_imgs = imgs_with_deletions.reshape(n_samples * iterations, *image_shape)
        exp_informed_model_confidences = (self.run_model(flattened_imgs)  # final dim is num_classes
                                          .reshape(n_samples, iterations, -1))

        # max confidence class on the original img
        original_pred_class = exp_informed_model_confidences[:, 0].argmax(axis=1)
        # we only care about confidence of the original prediction class
        exp_informed_class_confidence = exp_informed_model_confidences[np.arange(n_samples), ..., original_pred_class]

        # calculate area under the curve along iterations axis
        exp_informed_area_under_curves_per_img = np.trapz(exp_informed_class_confidence, axis=1)

        logger.debug("Repeating for randomised deletions.")
        seeds = np.random.default_rng(random_seed).choice(100, n_random_ranks, replace=False)
        imgs_with_random_deletions = np.zeros((n_random_ranks, n_samples, iterations, *image_shape))
        for i, seed in tqdm(enumerate(seeds), total=len(seeds), ncols=110,
                            desc="Randomly perturbing"):
            a_random_ranking = self.generate_random_ranking(16, seed)
            random_rankings = a_random_ranking[np.newaxis, ...].repeat(n_samples, axis=0)
            imgs_with_random_deletions[i] = self.incrementally_delete(random_rankings, iterations, deletion_method)[0]

        flattened_imgs = imgs_with_random_deletions.reshape(n_random_ranks * n_samples * iterations, *image_shape)
        random_model_confidences = (self.run_model(flattened_imgs)  # take mean over n_random_ranks
                                    .reshape(n_random_ranks, n_samples, iterations, -1).mean(axis=0))
        random_class_confidence = random_model_confidences[np.arange(n_samples), ..., original_pred_class]

        random_area_under_curves_per_img = np.trapz(random_class_confidence, axis=1)

        if visualise:
            fig, axes = plt.subplots(1, n_samples, sharey=True, figsize=(3 * n_samples, 3))
            for i, ax in enumerate(axes):  # type: int, plt.Axes
                ax.plot(range(len(k_values)), exp_informed_class_confidence[i], "-", label="exp_informed")
                ax.plot(range(len(k_values)), random_class_confidence[i], "--", label="random")
                ax.set_title(f"Image {i}")
                ax.set_xlim([0, len(k_values) - 1])
                ax.set_ylim([0, 1])
            fig.suptitle(f"Model confidence over deletion process")
            fig.tight_layout()
            plt.show()

        return {"informed": exp_informed_area_under_curves_per_img,
                "random": random_area_under_curves_per_img}

    def reset_child_params(self, model: torch.nn.Module):
        """
        Reset all parameters of the model to their defaults **inplace**.
        Adapted from https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        """
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            self.reset_child_params(layer)

    def incrementally_delete(
            self,
            importance_ranking: Int[np.ndarray, "n_samples height width"],
            num_iterations: int,
            method: DELETION_METHODS,
    ) -> t.Tuple[
        Float[np.ndarray, "n_samples num_iterations channels height width"],
        Int[np.ndarray, "num_iterations"]
    ]:
        """
        Iteratively delete the top k most important pixels (as specified by
        `importance_rank`).

        k is increased linearly from 0 (inclusive) to the total number pixels in the
        image over `num_iterations`.
        An ndarray of shape (num_iterations, *x.shape) is returned alongside the
        values of k used.
        """

        num_img_pixels = self.exp.input.shape[-2] * self.exp.input.shape[-1]

        k_values = np.floor(
            np.linspace(0, num_img_pixels, num_iterations)
        ).astype(int)

        incrementally_deleted = np.zeros((num_iterations, *self.exp.input.shape))

        for i, k in tqdm(enumerate(k_values), total=len(k_values), ncols=110,
                         desc="Deleting important pixels", leave=False):
            output = self.delete_top_k_important(importance_ranking, k, method)
            incrementally_deleted[i] = output

        # swap n_samples and num_iterations axes
        return incrementally_deleted.swapaxes(0, 1), k_values

    def delete_top_k_important(
            self,
            importance_ranking: Int[np.ndarray, "n_samples height width"],
            k: int,
            method: DELETION_METHODS,
    ) -> Float[np.ndarray, "n_samples channels height width"]:
        """
        'Delete' the top k most important pixels (as specified by `importance_rank`)
        in `x` (values in [-1,1]) using one of several methods
        determined by the `method` parameter:

        - If method is a float or int, set the top k pixels to this value.
        - If method is 'blur', apply a Gaussian blur to the top k pixels.
        - If method is a numpy random generator (`np.random.Generator`), add
            Gaussian noise to the top k pixels.
        - If method is 'inpaint', replace/inpaint the top k pixels using
            `skimage.restoration.inpaint_biharmonic`.
        - If method is 'nn', replace the top k pixels using nearest neighbour
            interpolation.
        - If method is 'shuffle', randomly perturb the top k pixels by shuffling
            them with the other pixels in the image.

        If importance_rank is a tuple of an int and a np.random.Generator, a random
        ranking, with grid element size of the first argument, is generated instead.
        """

        masked_imgs = self.exp.input.numpy(force=True)
        n_channels = self.exp.input.shape[1]

        top_k_mask = importance_ranking < k
        top_k_mask = np.expand_dims(top_k_mask, 1).repeat(n_channels, 1)  # mask across all input channels
        # top_k_mask.shape = (n_samples, 1, height, width)
        target_regions = masked_imgs[top_k_mask]

        # we need at least one 'known' pixel (where the top_k_mask is False)
        # for the inpaint and nn methods
        if method in ["inpaint", "nn"]:
            for img_mask in top_k_mask:
                if np.sum(~img_mask) < 1:
                    img_mask[img_mask.shape[0] // 2,
                             img_mask.shape[1] // 2] = False

        if isinstance(method, (float, int)):
            masked_imgs[top_k_mask] = np.clip(method, -1, 1)

        elif method == "blur":
            masked_imgs[top_k_mask] = scipy.ndimage.gaussian_filter(
                target_regions, sigma=5,
            )

        elif isinstance(method, np.random.Generator):
            noise = method.normal(size=target_regions.shape) / 5
            masked_imgs[top_k_mask] = np.clip(target_regions + noise, -1, 1)

        elif method == "inpaint":
            for i in range(len(masked_imgs)):
                mask = np.zeros_like(importance_ranking[i])
                mask[top_k_mask[i, 0]] = 1  # indicate unknown pixels
                masked_imgs[i] = skimage.restoration.inpaint_biharmonic(
                    masked_imgs[i], mask, channel_axis=0
                )

        elif method == "nn":
            masked_imgs[top_k_mask] = np.nan
            for i in range(len(masked_imgs)):
                # Adapted from https://stackoverflow.com/a/27745627/7253717
                filled_ind = scipy.ndimage.distance_transform_edt(
                    np.isnan(masked_imgs[i]), return_distances=False, return_indices=True)
                masked_imgs[i] = masked_imgs[i][tuple(filled_ind)]

        elif method == "shuffle":
            for i in range(len(masked_imgs)):
                pixels_to_scramble = masked_imgs[top_k_mask][i]

                # === Shuffle among all pixels ===
                # np.random.shuffle shuffles along first axis only
                # so transpose to shuffle RGB pixels (and not colour channels)
                np.random.shuffle(pixels_to_scramble.transpose(1, 0))

                # === Shuffle along neighbourhood *lines* ===
                # neighbourhood_size = 20
                # for i in range(0, pixels_to_scramble.shape[1], neighbourhood_size):
                #     np.random.shuffle(
                #         pixels_to_scramble.transpose(1, 0)[i:i + neighbourhood_size]
                #     )

                masked_imgs[top_k_mask][i] = pixels_to_scramble

        return masked_imgs

    def generate_random_ranking(
            self,
            resolution: int = 16,
            random_seed: int = 42
    ) -> Int[np.ndarray, "height width"]:
        _, _, h, w = self.exp.input.shape

        random_gen = np.random.default_rng(random_seed)
        random_importance = random_gen.permuted(
            np.floor(np.linspace(0, h*w, resolution ** 2))
        ).reshape(resolution, resolution)

        return skimage.transform.resize(
            random_importance, output_shape=(h, w),
            order=0, clip=False, preserve_range=True
        )
