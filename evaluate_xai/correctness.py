import copy
import typing as t

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Int, Float
from tqdm.autonotebook import tqdm

import helpers
from . import Co12Metric, Similarity
from . import deletion

logger = helpers.log.main_logger


def visualise_incremental_deletion(
        imgs_with_deletions: Float[np.ndarray, "n_samples num_iterations channels height width"],
):
    num_iterations = imgs_with_deletions.shape[1]

    selected_images: np.ndarray = imgs_with_deletions.copy()
    if num_iterations > 10:
        selected_images = selected_images.take(np.floor(np.linspace(0, num_iterations - 1, 10)).astype(int), axis=1)

    n, i, c, h, w = selected_images.shape
    helpers.plotting.show_image(einops.rearrange(selected_images, "n i c h w -> (n h) (i w) c"))
    # current_size = plt.gcf().get_size_inches()
    # width gets scaled if displaying a multi-spectral image (with more than 3 channels)
    fig = plt.gcf()
    fig.set_size_inches((c if c > 3 else 1) * i/5, n * 3/5)  # width, height
    fig.tight_layout(w_pad=1.5)
    plt.gcf().set_dpi(200)
    # plt.tight_layout()


class Correctness(Co12Metric):
    # todo: add docstrings - discuss execution time and add definitions from review paper

    def evaluate(
            self,
            method: t.Literal["model_randomisation", "incremental_deletion"],
            **kwargs,
    ) -> t.Union[Similarity,
                 dict[t.Literal["informed", "random"], Float[np.ndarray, "n_samples"]]]:

        super().evaluate(method, **kwargs)

        if method == "model_randomisation":
            return self._model_randomisation()
        elif method == "incremental_deletion":
            return self._incremental_deletion(**kwargs)
        else:
            raise ValueError(f"Unknown evaluation method: {method}")

    def _model_randomisation(
            self,
            **kwargs,
    ) -> Similarity:
        device = helpers.utils.get_model_device(self.exp.model)
        randomised_model = copy.deepcopy(self.exp.model).to(device)
        self.reset_child_params(randomised_model)
        randomised_model.eval()

        exp_for_randomised_model = self.get_sub_explainer(
            "randomised", self.exp.input, model=randomised_model
        )

        if self.visualise:
            self.compare_sub_explainer(
                exp_for_randomised_model,
                title="Explanation of original/randomised model"
            )

        return Similarity(self.exp, exp_for_randomised_model)

    def _incremental_deletion(
            self,
            deletion_method: deletion.METHODS = "nn",
            iterations: int = 30,
            n_random_rankings: int = 5,
            random_seed: int = 42,
            **kwargs,
    ) -> dict[t.Literal["informed", "random"], Float[np.ndarray, "n_samples"]]:

        n_samples = self.exp.input.shape[0]
        image_shape = self.exp.input.shape[1:]
        base_n = n_samples * iterations

        imgs_with_deletions, k_values = self.incrementally_delete(
            self.exp.ranked_explanation, iterations, deletion_method
        )
        flattened_imgs_a = imgs_with_deletions.reshape(base_n, *image_shape)

        if self.visualise:
            visualise_incremental_deletion(imgs_with_deletions)
            plt.suptitle(f"Incremental informed deletion over {iterations} iterations",
                         fontsize=20)
            plt.show()

        # Do the same thing for randomised deletions
        logger.debug("Repeating _incremental_deletion for randomised deletions.")
        seeds = np.random.default_rng(random_seed).choice(10*n_random_rankings, n_random_rankings, replace=False)
        imgs_with_random_deletions = np.zeros((n_random_rankings, n_samples, iterations, *image_shape))
        for i, seed in tqdm(enumerate(seeds), total=len(seeds), ncols=110, mininterval=5,
                            desc="Randomly perturbing", leave=False):  # type: int, int
            a_random_ranking = deletion.generate_random_ranking(
                image_shape[-2:], 16, seed
            )
            random_rankings = a_random_ranking[np.newaxis, ...].repeat(n_samples, axis=0)
            imgs_with_random_deletions[i] = self.incrementally_delete(
                random_rankings, iterations, deletion_method
            )[0]
        flattened_imgs_b = imgs_with_random_deletions.reshape(n_random_rankings * base_n, *image_shape)
        # todo: save/load perturbations?
        if self.visualise:
            # show each different random ranking on 0th image
            visualise_incremental_deletion(imgs_with_random_deletions[:, 0])
            plt.suptitle(f"Incremental randomised deletion {n_random_rankings} times over {iterations} iterations",
                         fontsize=20)
            plt.show()

        # Generate model confidence for all images (in one pass for efficiency)
        all_outputs = self.run_model(np.concatenate([flattened_imgs_a, flattened_imgs_b], axis=0))
        # Split up the outputs again from the concatenation
        informed_outputs = all_outputs[:base_n]
        random_outputs = all_outputs[base_n:]

        # final dim is num_classes so use -1
        exp_informed_model_confidences = informed_outputs.reshape(n_samples, iterations, -1)
        # max confidence class on the original img
        original_pred_class = exp_informed_model_confidences[:, 0].argmax(axis=1)
        # we only care about confidence of the original prediction class
        exp_informed_class_confidence = exp_informed_model_confidences[np.arange(n_samples), ..., original_pred_class]
        # Calculate area under the curve along iterations axis. This is our final output.
        exp_informed_area_under_curve_per_img = np.trapz(exp_informed_class_confidence, axis=1)

        # take mean over n_random_rankings
        random_model_confidences = random_outputs.reshape(n_random_rankings, n_samples, iterations, -1).mean(axis=0)
        # confidence class in the original prediction over the iterations for each sample
        random_class_confidence = random_model_confidences[np.arange(n_samples), :, original_pred_class]

        no_deletion_confidence_diff = random_class_confidence[:, 0] - exp_informed_class_confidence[:, 0]
        assert np.all(no_deletion_confidence_diff <= 0.01), \
            f"For no deletions, prediction confidences should be identical, not {no_deletion_confidence_diff} > 0.01."

        # Second part of final output.
        random_area_under_curve_per_img = np.trapz(random_class_confidence, axis=1)

        # Visualise the area under curve results by plotting confidence against iterations
        if self.visualise:
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

        return {"informed": exp_informed_area_under_curve_per_img,
                "random": random_area_under_curve_per_img}

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
            method: deletion.METHODS,
    ) -> tuple[
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

        for i, k in tqdm(enumerate(k_values), total=len(k_values), ncols=110, mininterval=5,
                         desc="Deleting important pixels", leave=False):
            output = deletion.delete_top_k_important(
                self.exp.input, importance_ranking, k, method
            )
            incrementally_deleted[i] = output

        # swap n_samples and num_iterations axes
        return incrementally_deleted.swapaxes(0, 1), k_values
