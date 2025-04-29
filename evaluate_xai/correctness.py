import copy
import typing as t

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Int, Float
from tqdm.autonotebook import tqdm

import helpers
from . import Co12Property, Similarity
from . import deletion

logger = helpers.log.main_logger


def visualise_incremental_deletion(x: Float[np.ndarray, "n_samples num_iterations channels height width"]):
    """
    A helper function to visualise the incremental deletion process. Does *not* call plt.show().
    Expects an input with dimensions (n_samples, num_deletion_iterations, channels, height, width).
    If num_deletion_iterations > 10, 10 linearly spaced iterations are selected.
    """

    num_iterations = x.shape[1]

    selected_images: np.ndarray = x.copy()
    if num_iterations > 10:  # extract 10 evenly spaced iterations
        selected_images = selected_images.take(np.floor(np.linspace(0, num_iterations - 1, 10)).astype(int), axis=1)

    n, i, c, h, w = selected_images.shape
    helpers.plotting.show_image(einops.rearrange(selected_images, "n i c h w -> (n h) (i w) c"))
    # current_size = plt.gcf().get_size_inches()
    fig = plt.gcf()
    fig_scale_factor = .5
    # width gets additionally scaled if displaying a multi-spectral image (with more than 3 channels)
    fig.set_size_inches((c if c > 3 else 1) * i * fig_scale_factor, n * fig_scale_factor)  # width, height
    fig.tight_layout(w_pad=1.5)
    fig.set_dpi(200)
    # plt.tight_layout()


def reset_child_params(model: torch.nn.Module):
    """
    Reset all parameters of a PyTorch Module/model to their defaults **inplace** and recursively.
    Adapted from https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        reset_child_params(layer)


class Correctness(Co12Property):
    """
    "Correctness addresses the truthfulness/faithfulness of the explanation with respect to predictive
    model f, the model to be explained. Hence, it indicates how truthful the explanations are
    compared to the “true” black box behaviour (either locally or globally). Note that this property is
    not about the predictive accuracy of the black box model, but about the descriptive accuracy of the
    explanation. Ideally, an explanation is “nothing but the truth”, and high correctness is
    desired"
    """
    def evaluate(
            self,
            method: t.Literal["model_randomisation", "incremental_deletion"],
            **kwargs,
    ) -> t.Union[Similarity, dict[str, Float[np.ndarray, "n_samples"]]]:

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
        """
        Performs a model randomisation test by creating a copy of the original model,
        resetting its parameters, and computing and return the Similarity between the explanations of the
        original and randomised models. We desire a low similarity score, indicating that the
        explanation of the original model is faithful to the model's workings.

        🐌 Since this method requires generating new explanations, it can be slow.
        """

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
    ) -> dict[str, Float[np.ndarray, "n_samples"]]:
        """
        Performs an incremental deletion check to investigate the effect of deleting
        pixels or regions on a model's predictions when informed by explanations compared to a randomised baseline.
        The method computes model confidence over iterations during these deletion processes and returns the area
        under curve (AUC) for explanation-informed deletion and randomised deletion as two numpy arrays.
        Typically, we take the ratio of the two AUCs (informed / randomised)
        and desire a ratio below 1 indicating better than random performance.

        🐌 Since this method requires a lot of model invocations, it can be slow.
        Some deletion methods can also take some time (e.g. inpaint).

        :param deletion_method: Method used for deletion: one of deletion.METHODS (see `deletion.py`).
            Defaults to "nn" (nearest neighbour).
        :param iterations: Total number of incremental deletion steps, `K`.
        :param n_random_rankings: Size of ensemble, `T`, of random rankings to use as a baseline.
        :param random_seed: Random seed used for generating randomised explanation for random deletion. Defaults to 42.
        :return: A dictionary containing results of the incremental deletion
            process. The main keys are "informed" and "random", each containing
            area under curve (AUC) values for model confidence over deletion
            iterations for each sample considered.
            Additional keys may be included if `self.full_data` is True,
            such as "informed_full", "random_full" (complete confidence trends),
            "informed_deleted_imgs", "random_deleted_imgs" (deleted images),
            and "random_rankings" (random rankings used).
        """
        assert n_random_rankings > 0, "n_random_rankings must be greater than 0."

        image_shape = self.exp.input.shape[1:]
        total_num_samples = self.n_samples * iterations

        imgs_with_deletions, k_values = self.incrementally_delete_from_input(
            self.exp.ranked_explanation, iterations, deletion_method
        )
        imgs_flat_list = imgs_with_deletions.reshape(total_num_samples, *image_shape)
        informed_outputs = self.run_model(imgs_flat_list)

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
                image_shape[-2:], 16, seed  # futuretodo: make resolution_d configurable
            )
            random_rankings = a_random_ranking[np.newaxis, ...].repeat(self.n_samples, axis=0)

            imgs_with_random_deletions = self.incrementally_delete_from_input(
                random_rankings, iterations, deletion_method
            )[0]  # don't care about k_values here, so take the first output
            if self.visualise:
                # Use just the first image as an example to showcase random deletions
                sample_perturbation_history.append(imgs_with_random_deletions[0])

            imgs_flat_list = imgs_with_random_deletions.reshape(total_num_samples, *image_shape)
            random_outputs_for_seed = self.run_model(imgs_flat_list)
            # Build up the model's outputs for each seed as we go along
            # Leaving it until the end with all the perturbations in memory uses too much RAM (especially for MS data)!
            random_outputs.append(random_outputs_for_seed)

        random_outputs = np.concatenate(random_outputs, axis=0)  # as if all fed into the model in one go

        # futuretodo: save/load perturbations (like adversarial examples in contrastivity.py)
        #  to disk to save having to regenerate
        if self.visualise:
            # show each different random ranking on the 0th image (collected progressively above)
            visualise_incremental_deletion(np.stack(sample_perturbation_history, axis=0))
            plt.suptitle(f"Incremental randomised deletion {n_random_rankings} times over {iterations} iterations")
            plt.show()

        # The final dimension is num_classes (unknown here), so use -1
        informed_confidences = informed_outputs.reshape(self.n_samples, iterations, -1)
        # max confidence class on the original img
        original_pred_class = informed_confidences[:, 0].argmax(axis=1)
        # we only care about the confidence in the original prediction class
        exp_informed_class_confidence = informed_confidences[np.arange(self.n_samples), ..., original_pred_class]
        # Calculate area under the curve along the discrete iteration (k) axis.
        # This is the first part of the final output.
        exp_informed_area_under_curve_per_img = np.trapz(exp_informed_class_confidence, axis=1)

        # Perform the same steps for random ensemble
        # Take mean over n_random_rankings
        random_confidences = random_outputs.reshape(n_random_rankings, self.n_samples, iterations, -1).mean(axis=0)
        # confidence class in the original prediction over the iterations for each sample
        random_class_confidence = random_confidences[np.arange(self.n_samples), :, original_pred_class]

        no_deletion_confidence_diff = random_class_confidence[:, 0] - exp_informed_class_confidence[:, 0]
        assert np.all(no_deletion_confidence_diff <= 0.01), \
            f"For no deletions, prediction confidences should be identical, not {no_deletion_confidence_diff} > 0.01."

        # Second part of the final output.
        random_area_under_curve_per_img = np.trapz(random_class_confidence, axis=1)

        # Visualise the area under curve results by plotting confidence against iterations
        if self.visualise:
            fig, axes = plt.subplots(1, self.n_samples, sharey=True, figsize=(3 * self.n_samples, 3))
            for i, ax in enumerate(axes):  # type: int, plt.Axes
                ax.plot(range(len(k_values)), exp_informed_class_confidence[i], "-", label="exp_informed")
                ax.plot(range(len(k_values)), random_class_confidence[i], "--", label="random")
                ax.set_title(f"Image {i}")
                ax.set_xlim((0, len(k_values) - 1))
                ax.set_xticks(range(len(k_values)))
                ax.set_ylim((0, 1))
            fig.suptitle(f"Model confidence over deletion process")
            fig.tight_layout()
            plt.show()

        return_dict = {
            "informed": exp_informed_area_under_curve_per_img,
            "random": random_area_under_curve_per_img
        }
        if self.full_data:
            return_dict["informed_full"] = exp_informed_class_confidence
            return_dict["random_full"] = random_class_confidence
            return_dict["informed_deleted_imgs"] = imgs_with_deletions
            return_dict["random_deleted_imgs"] = imgs_with_random_deletions
            return_dict["random_rankings"] = random_rankings

        return return_dict

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
        Iteratively delete the top k most important pixels (as specified by `importance_rank`).

        k is increased linearly from 0 to the total number of pixels in the image
        over `num_iterations` (both inclusive).

        A tuple is returned: an ndarray of shape (num_iterations, *x.shape) and the exact values of k used.
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
