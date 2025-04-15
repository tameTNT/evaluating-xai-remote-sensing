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
    fig = plt.gcf()
    fig_scale_factor = .2
    # width gets additionally scaled if displaying a multi-spectral image (with more than 3 channels)
    fig.set_size_inches((c if c > 3 else 1) * i * fig_scale_factor, n * fig_scale_factor)  # width, height
    fig.tight_layout(w_pad=1.5)
    fig.set_dpi(200)
    # plt.tight_layout()


def reset_child_params(model: torch.nn.Module):
    """
    Reset all parameters of a model to their defaults **inplace**.
    Adapted from https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        reset_child_params(layer)


class Correctness(Co12Metric):
    # todo: add docstrings - discuss execution time and add definition of property from review paper

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
        reset_child_params(randomised_model)
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
            iterations: int = 15,  # NB: more iterations means more memory (RAM) used by self.incrementally_delete
            n_random_rankings: int = 5,
            random_seed: int = 42,
            **kwargs,
    ) -> dict[t.Literal["informed", "random"], Float[np.ndarray, "n_samples"]]:

        image_shape = self.exp.input.shape[1:]
        total_num_samples = self.n_samples * iterations

        imgs_with_deletions, k_values = self.incrementally_delete_from_input(
            self.exp.ranked_explanation, iterations, deletion_method
        )
        imgs_flat = imgs_with_deletions.reshape(total_num_samples, *image_shape)
        informed_outputs = self.run_model(imgs_flat)

        if self.visualise:
            visualise_incremental_deletion(imgs_with_deletions)
            plt.suptitle(f"Incremental informed deletion over {iterations} iterations")
            plt.show()

        # Do the same thing for randomised deletions
        logger.debug(f"Performing incremental deletion for {n_random_rankings} rounds of randomised deletions.")
        # pick n_random_rankings seeds
        seeds = np.random.default_rng(random_seed).choice(10*n_random_rankings, n_random_rankings, replace=False)

        random_outputs = []
        sample_perturbation_history = []
        for i, seed in tqdm(enumerate(seeds), total=len(seeds), ncols=110, mininterval=5,
                            desc="Randomly perturbing", leave=False):  # type: int, int
            a_random_ranking = deletion.generate_random_ranking(
                image_shape[-2:], 16, seed
            )
            random_rankings = a_random_ranking[np.newaxis, ...].repeat(self.n_samples, axis=0)

            imgs_with_random_deletions = self.incrementally_delete_from_input(
                random_rankings, iterations, deletion_method
            )[0]
            if self.visualise:
                # Just use the first image as an example to showcase random deletions
                sample_perturbation_history.append(imgs_with_random_deletions[0])

            imgs_flat = imgs_with_random_deletions.reshape(total_num_samples, *image_shape)
            random_outputs_for_seed = self.run_model(imgs_flat)
            # Build up the model's outputs for each seed as we go along
            # Leaving it until the end with all the perturbations in memory uses too much RAM!
            random_outputs.append(random_outputs_for_seed)

        random_outputs = np.concatenate(random_outputs, axis=0)  # as if all fed into model in one go

        # futuretodo: save/load perturbations (like adversarial examples in contrastivity.py)
        #  to disk to save having to regenerate
        if self.visualise:
            # show each different random ranking on the 0th image (collected progressively above)
            visualise_incremental_deletion(np.stack(sample_perturbation_history, axis=0))
            plt.suptitle(f"Incremental randomised deletion {n_random_rankings} times over {iterations} iterations")
            plt.show()

        # final dim is num_classes so use -1
        exp_informed_model_confidences = informed_outputs.reshape(self.n_samples, iterations, -1)
        # max confidence class on the original img
        original_pred_class = exp_informed_model_confidences[:, 0].argmax(axis=1)
        # we only care about confidence of the original prediction class
        exp_informed_class_confidence = exp_informed_model_confidences[np.arange(self.n_samples), ..., original_pred_class]
        # Calculate area under the curve along iterations axis. This is our final output.
        exp_informed_area_under_curve_per_img = np.trapz(exp_informed_class_confidence, axis=1)

        # take mean over n_random_rankings
        random_model_confidences = random_outputs.reshape(n_random_rankings, self.n_samples, iterations, -1).mean(axis=0)
        # confidence class in the original prediction over the iterations for each sample
        random_class_confidence = random_model_confidences[np.arange(self.n_samples), :, original_pred_class]

        no_deletion_confidence_diff = random_class_confidence[:, 0] - exp_informed_class_confidence[:, 0]
        assert np.all(no_deletion_confidence_diff <= 0.01), \
            f"For no deletions, prediction confidences should be identical, not {no_deletion_confidence_diff} > 0.01."

        # Second part of final output.
        random_area_under_curve_per_img = np.trapz(random_class_confidence, axis=1)

        # Visualise the area under curve results by plotting confidence against iterations
        if self.visualise:
            fig, axes = plt.subplots(1, self.n_samples, sharey=True, figsize=(3 * self.n_samples, 3))
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

    def incrementally_delete_from_input(
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
