import typing as t

import matplotlib.pyplot as plt
import numpy as np
from tqdm.autonotebook import tqdm
from jaxtyping import Float

import helpers
from . import Co12Metric
from . import deletion

logger = helpers.log.main_logger


class OutputCompleteness(Co12Metric):
    def evaluate(
            self,
            method: t.Literal["deletion_check", "preservation_check"],
            **kwargs,
    ) -> Float[np.ndarray, "n_samples"]:
        super().evaluate(method, **kwargs)

        if method == "deletion_check":
            return self._deletion_check(**kwargs)
        elif method == "preservation_check":
            return self._preservation_check(**kwargs)
        else:
            raise ValueError(f"Unknown evaluation method: {method}")

    def _deletion_check(
            self,
            deletion_method: deletion.METHODS = "shuffle",
            threshold: float = 0.1,  # delete only top 10% of features
            n_random_rankings: int = 5,
            random_seed: int = 42,
            **kwargs,
    ) -> Float[np.ndarray, "n_samples"]:
        informed_del_conf, random_del_conf = self.delete_via_ranking(
            self.exp.ranked_explanation, deletion_method, threshold, n_random_rankings, random_seed,
        )

        # futuretodo: perhaps a fractional/proportional metric is better?
        #  since different classes' random deletion might be more/less effective
        # Drop in acc vs random = (rand_del_acc - org_acc) - (del_acc - org_acc)
        #                       = rand_del_acc - del_acc
        # We want del_acc to be small (the deletion was effective at removing important info),
        # while rand_del_acc should be high (the random deletion didn't successfully remove important info)
        # so best value is 1. Worst is -1 (the random deletion was much more effective than the informed one)
        return random_del_conf - informed_del_conf

    def _preservation_check(
            self,
            deletion_method: deletion.METHODS = "shuffle",
            threshold: float = 0.1,  # keep only top 10% of features
            n_random_rankings: int = 5,
            random_seed: int = 42,
            **kwargs,
    ):
        # Since 0 is the 'most important' pixel, in ranked_explanation,
        # we can invert it by simply by ranking it again (so 0 becomes least important)
        inverted_importance_ranking = helpers.utils.rank_pixel_importance(self.exp.ranked_explanation)

        informed_pres_conf, random_pres_conf = self.delete_via_ranking(
            inverted_importance_ranking, deletion_method, 1-threshold, n_random_rankings, random_seed,
        )

        # drop in acc vs random = (rand_pres_acc - org_acc) - (pres_acc - org_acc)
        #                       = rand_pres_acc - pres_acc
        # We want pres_acc to remain high (we only removed non-important stuff),
        # while rand_pres_acc should be low (we accidentally removed important stuff)
        # so best score is -1, worst is 1.
        return random_pres_conf - informed_pres_conf

    def delete_via_ranking(
            self,
            importance_ranking: Float[np.ndarray, "n_samples height width"],
            deletion_method: deletion.METHODS,
            threshold: float,
            n_random_rankings: int,
            random_seed: int,
    ) -> tuple[Float[np.ndarray, "n_samples"], Float[np.ndarray, "n_samples"]]:
        n_samples = self.exp.input.shape[0]

        num_pixels = self.exp.input.shape[-2] * self.exp.input.shape[-1]
        imgs_with_deletions = deletion.delete_top_k_important(
            self.exp.input, importance_ranking, threshold*num_pixels, method=deletion_method,
        )

        if self.visualise:
            helpers.plotting.show_image(imgs_with_deletions)
            plt.suptitle(f"Explanation informed deletion/preservation (threshold={threshold})",
                         fontsize=15)
            plt.show()

        logger.debug("Repeating for randomised deletions.")
        seeds = np.random.default_rng(random_seed).choice(10*n_random_rankings, n_random_rankings, replace=False)
        imgs_with_random_deletions = np.zeros((n_random_rankings, *self.exp.input.shape))
        for i, seed in tqdm(enumerate(seeds), total=len(seeds), ncols=110, mininterval=5,
                            desc="Randomly perturbing", leave=False):  # type: int, int
            a_random_ranking = deletion.generate_random_ranking(
                self.exp.input.shape[-2:], 16, seed
            )
            random_rankings = a_random_ranking[np.newaxis, ...].repeat(self.exp.input.shape[0], axis=0)
            imgs_with_random_deletions[i] = deletion.delete_top_k_important(
                self.exp.input, random_rankings, threshold*num_pixels, method=deletion_method,
            )

        if self.visualise:
            # show each different random ranking on 0th image
            helpers.plotting.show_image(imgs_with_random_deletions[:, 0])
            plt.suptitle(f"{n_random_rankings} random deletion/preservation rounds (threshold={threshold})",
                         fontsize=15)
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

        conf_informed_deletion = informed_outputs[np.arange(n_samples), original_prediction]
        conf_random_deletion = random_outputs[np.arange(n_samples), original_prediction]

        return conf_informed_deletion, conf_random_deletion
