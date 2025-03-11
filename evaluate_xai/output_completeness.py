import typing as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.autonotebook import tqdm
from jaxtyping import Float

import helpers.plotting
from . import Co12Metric
from . import deletion

logger = helpers.log.get_logger("main")


class OutputCompleteness(Co12Metric):
    # todo: add docstrings based on paper
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(
            self,
            method: t.Literal["deletion_check"],
            **kwargs,
    ) -> Float[np.ndarray, "n_samples"]:
        super().evaluate(method, **kwargs)

        if method == "deletion_check":
            return self._deletion_check(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _deletion_check(
            self,
            deletion_method: deletion.METHODS = "shuffle",
            threshold: float = 0.1,  # "all important features"
            n_random_rankings: int = 5,
            random_seed: int = 42,
    ) -> Float[np.ndarray, "n_samples"]:

        n_samples = self.exp.input.shape[0]

        num_pixels = self.exp.input.shape[-2] * self.exp.input.shape[-1]
        imgs_with_deletions = deletion.delete_top_k_important(
            self.exp.input, self.exp.ranked_explanation, threshold*num_pixels, method=deletion_method,
        )

        helpers.plotting.show_image(imgs_with_deletions)
        plt.show()

        logger.debug("Repeating _deletion_check for randomised deletions.")
        seeds = np.random.default_rng(random_seed).choice(10*n_random_rankings, n_random_rankings, replace=False)
        imgs_with_random_deletions = np.zeros((n_random_rankings, *self.exp.input.shape))
        for i, seed in tqdm(enumerate(seeds), total=len(seeds), ncols=110,
                            desc="Randomly perturbing"):  # type: int, int
            a_random_ranking = deletion.generate_random_ranking(
                self.exp.input.shape[-2:], 16, seed
            )
            random_rankings = a_random_ranking[np.newaxis, ...].repeat(self.exp.input.shape[0], axis=0)
            imgs_with_random_deletions[i] = deletion.delete_top_k_important(
                self.exp.input, random_rankings, threshold*num_pixels, method=deletion_method,
            )

        helpers.plotting.show_image(imgs_with_random_deletions[0])
        plt.show()

        # flatten out n_random_rankings dimension into n_samples dimension
        flattened_random_deletions = np.concatenate(imgs_with_random_deletions, axis=0)

        # Generate model confidence for all candidates (original, informed del, randomised del)
        all_outputs = self.run_model(
            np.concatenate([
                self.exp.input.numpy(force=True), imgs_with_deletions, flattened_random_deletions
            ], axis=0)
        )
        # Get the predictions for the original images. Let's see how the perturbations can decrease it!
        original_outputs = all_outputs[:n_samples]
        original_prediction = original_outputs.argmax(axis=1)

        informed_outputs = all_outputs[n_samples:2*n_samples]
        # take average across n_random_rankings
        random_outputs = all_outputs[2*n_samples:].reshape(n_random_rankings, n_samples, -1).mean(axis=0)

        confidence_after_informed_deletion = informed_outputs[np.arange(n_samples), original_prediction]
        confidence_after_random_deletion = random_outputs[np.arange(n_samples), original_prediction]

        # drop in acc vs random = (rand_del_acc - org_acc) - (del_acc - org_acc) = rand_del_acc - del_acc
        # we want del_acc to be small (the deletion was effective at removing important info),
        # so best value is 1. Worst is -1 (the random deletion was much more effective than the informed one)
        return confidence_after_random_deletion - confidence_after_informed_deletion
