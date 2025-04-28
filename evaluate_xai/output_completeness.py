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
    ) -> Float[np.ndarray, "n_samples"] | tuple[
        Float[np.ndarray, "n_samples"],
        tuple[Float[np.ndarray, "n_samples channels height width"],]
    ]:
        (informed_del_conf, random_del_conf), imgs_tuple = self._perform_check(
            self.exp.ranked_explanation, deletion_method, threshold, n_random_rankings, random_seed,
        )

        # futuretodo: perhaps a fractional/proportional metric is better?
        #  since different classes' random deletion might be more/less effective
        #  But this doesn't make complete sense because we need to include the original confidence somehow
        #  A ratio also wouldn't have a best/worst value?
        # Drop in acc vs random = (rand_del_acc - org_acc) - (del_acc - org_acc)
        #                       = rand_del_acc - del_acc
        # We want del_acc to be small (the deletion was effective at removing important info),
        # while rand_del_acc should be high (the random deletion didn't successfully remove important info)
        # so best value is 1. Worst is -1 (the random deletion was much more effective than the informed one)
        result = random_del_conf - informed_del_conf

        if self.full_data:
            return result, imgs_tuple
        else:
            return result

    def _preservation_check(
            self,
            deletion_method: deletion.METHODS = "shuffle",
            threshold: float = 0.1,  # keep only top 10% of features
            n_random_rankings: int = 5,
            random_seed: int = 42,
            **kwargs,
    ) -> Float[np.ndarray, "n_samples"] | tuple[
        Float[np.ndarray, "n_samples"],
        tuple[Float[np.ndarray, "n_samples channels height width"],]
    ]:
        # Since 0 is the 'most important' pixel, in ranked_explanation,
        # we can invert it by simply by ranking it again (so 0 becomes least important)
        inverted_importance_ranking = helpers.utils.rank_pixel_importance(self.exp.ranked_explanation)

        (informed_pres_conf, random_pres_conf), imgs_tuple = self._perform_check(
            inverted_importance_ranking, deletion_method, 1-threshold, n_random_rankings, random_seed,
        )

        # drop in acc vs random = (rand_pres_acc - org_acc) - (pres_acc - org_acc)
        #                       = rand_pres_acc - pres_acc
        # We want pres_acc to remain high (we only removed non-important stuff),
        # while rand_pres_acc should be low (we accidentally removed important stuff)
        # After inverting (for alignment with deletion check), best score is 1 and worst is -1.
        result = informed_pres_conf - random_pres_conf

        if self.full_data:
            # fixme: clean up full_data return process
            return result, imgs_tuple
        else:
            return result

    def _perform_check(
            self,
            importance_ranking: Float[np.ndarray, "n_samples height width"],
            deletion_method: deletion.METHODS,
            threshold: float,
            n_random_rankings: int,
            random_seed: int,
    ) -> tuple[
        tuple[Float[np.ndarray, "n_samples"], Float[np.ndarray, "n_samples"]],
        tuple[
            Float[np.ndarray, "n_samples channels height width"],
            Float[np.ndarray, "n_samples channels height width"],
        ]
    ]:
        informed_deletions, random_deletions = self.delete_from_input(
            importance_ranking, deletion_method, threshold, n_random_rankings, random_seed,
        )
        return self._predict_on_deleted(informed_deletions, random_deletions), (informed_deletions, random_deletions)

    def delete_from_input(
            self,
            importance_ranking: Float[np.ndarray, "n_samples height width"],
            deletion_method: deletion.METHODS,
            threshold: float,
            n_random_rankings: int,
            random_seed: int,
    ) -> tuple[
        Float[np.ndarray, "n_samples channels height width"],
        Float[np.ndarray, "n_random_rankings n_samples channels height width"],
    ]:
        num_pixels = self.exp.input.shape[-2] * self.exp.input.shape[-1]
        imgs_with_deletions = deletion.delete_top_k_important(
            self.exp.input, importance_ranking, threshold*num_pixels, method=deletion_method,
        )

        if self.visualise:
            helpers.plotting.show_image(imgs_with_deletions, final_fig_size=(8, (self.n_samples+8)//8))
            plt.suptitle(f"Explanation informed deletion/preservation (threshold={threshold})")
            plt.show()

        logger.debug("Repeating for randomised deletions.")
        seeds = np.random.default_rng(random_seed).choice(10*n_random_rankings, n_random_rankings, replace=False)
        imgs_with_random_deletions = np.zeros((n_random_rankings, *self.exp.input.shape))
        for i, seed in tqdm(enumerate(seeds), total=len(seeds), ncols=110, mininterval=5,
                            desc="Randomly perturbing", leave=False):  # type: int, int
            a_random_ranking = deletion.generate_random_ranking(
                self.exp.input.shape[-2:], 16, seed
            )
            random_rankings = a_random_ranking[np.newaxis, ...].repeat(self.n_samples, axis=0)
            imgs_with_random_deletions[i] = deletion.delete_top_k_important(
                self.exp.input, random_rankings, threshold*num_pixels, method=deletion_method,
            )

        if self.visualise:
            # show each different random ranking on 0th image
            helpers.plotting.show_image(imgs_with_random_deletions[:, 0],
                                        final_fig_size=(8, (n_random_rankings+8)//8 + .75))
            plt.suptitle(f"{n_random_rankings} random deletion/preservation rounds (threshold={threshold})")
            plt.show()

        return imgs_with_deletions, imgs_with_random_deletions

    def _predict_on_deleted(
            self,
            informed_deletions: Float[np.ndarray, "n_samples channels height width"],
            random_deletions: Float[np.ndarray, "n_random_rankings n_samples channels height width"]
    ) -> tuple[Float[np.ndarray, "n_samples"], Float[np.ndarray, "n_samples"]]:

        n_random_rankings = random_deletions.shape[0]  # to reshape later
        # flatten out n_random_rankings dimension into n_samples dimension
        flattened_random_deletions = np.concatenate(random_deletions, axis=0)

        # Generate model confidence for all candidates (original, informed del, randomised del)
        all_outputs = self.run_model(
            np.concatenate([
                self.exp.input.numpy(force=True), informed_deletions, flattened_random_deletions
            ], axis=0)
        )
        # Get the predictions for the original images.
        # Let's see how the perturbations can decrease/preserve it!
        original_outputs = all_outputs[:self.n_samples]
        original_prediction = original_outputs.argmax(axis=1)
        # todo: return this for full_data rather than just printing
        print(f"Original predictions (confidence): "
              f"{original_prediction} ({original_outputs[:, original_prediction]})") if self.visualise else None

        informed_outputs = all_outputs[self.n_samples:2*self.n_samples]
        # take average across n_random_rankings
        random_outputs = all_outputs[2*self.n_samples:].reshape(n_random_rankings, self.n_samples, -1).mean(axis=0)

        informed_deletion_confidences = informed_outputs[np.arange(self.n_samples), original_prediction]
        print(f"Confidence after informed: "
              f"{informed_deletion_confidences}") if self.visualise else None

        random_deletion_confidences = random_outputs[np.arange(self.n_samples), original_prediction]
        print(f"Confidence after random: "
              f"{random_deletion_confidences}") if self.visualise else None

        return informed_deletion_confidences, random_deletion_confidences
