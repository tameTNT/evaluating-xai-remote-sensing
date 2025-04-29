import typing as t

import matplotlib.pyplot as plt
import numpy as np
from tqdm.autonotebook import tqdm
from jaxtyping import Float

import helpers
from . import Co12Property
from . import deletion

logger = helpers.log.main_logger


class OutputCompleteness(Co12Property):
    """
    "Output-completeness addresses the extent to which the explanation covers the output of
    model f. Thus, it is a “quantification of unexplainable feature components” and measures
    how well the explanation method agrees with the predictions of the original predictive
    model."
    """
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
            proportion: float = 0.1,  # delete only the top 10% of features
            n_random_rankings: int = 5,
            random_seed: int = 42,
            **kwargs,
    ) -> t.Union[
            Float[np.ndarray, "n_samples"],
            tuple[
                Float[np.ndarray, "n_samples"],
                tuple[Float[np.ndarray, "n_samples channels height width"],
                      Float[np.ndarray, "n_samples channels height width"]]
            ]
    ]:
        """
        Calculate the deletion check metric, computing the drop in model
        confidence when evaluated on an image with the top, e.g. 10%, of pixels
        according to the explanation deleted. We desire a large drop in model confidence.
        This is compared to the effect of randomly deleting pixels.
        The best possible score is 1, the worst is -1.

        :param deletion_method: Method used for deletion: one of deletion.METHODS (see `deletion.py`).
            Defaults to "shuffle".
        :param proportion: The proportion for deletion, μ, representing the proportion of pixels
            deleted (starting from the most important). Defaults to 0.1.
        :param n_random_rankings: Size of ensemble, `T`, of random rankings to use as a baseline. Defaults to 5.
        :param random_seed: Random seed used for generating randomised explanation for random deletion. Defaults to 42.
        :return: An array of size n_samples containing the deletion check metric for each sample.
            If `self.full_data` is True, it also returns the images themselves with informed and random deletions.
        """

        (informed_del_conf, random_del_conf), imgs_tuple = self._perform_check(
            self.exp.ranked_explanation, deletion_method, proportion, n_random_rankings, random_seed,
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
            proportion: float = 0.1,  # keep only the top 10% of features
            n_random_rankings: int = 5,
            random_seed: int = 42,
            **kwargs,
    ) -> t.Union[
        Float[np.ndarray, "n_samples"],
        tuple[
            Float[np.ndarray, "n_samples"],
            tuple[Float[np.ndarray, "n_samples channels height width"],
                  Float[np.ndarray, "n_samples channels height width"]]
        ]
    ]:
        """
        Calculate the preservation check metric, computing the drop in model
        confidence when evaluated on an image with only the top, e.g. 10%, of pixels
        according to the explanation kept and the rest deleted.
        In contrast to the deletion check, we hence desire a *small* drop in model confidence.
        This is compared to the effect of randomly deleting pixels.
        The best possible score is still 1, the worst is -1.

        :param deletion_method: Method used for deletion: one of deletion.METHODS (see `deletion.py`).
            Defaults to "shuffle".
        :param proportion: The proportion for preservation, μ, representing the proportion of pixels
            preserved (starting from the most important). Defaults to 0.1.
        :param n_random_rankings: Size of ensemble, `T`, of random rankings to use as a baseline. Defaults to 5.
        :param random_seed: Random seed used for generating randomised explanation for random deletion. Defaults to 42.
        :return: An array of size n_samples containing the preservation check metric for each sample.
            If `self.full_data` is True, it also returns the images themselves with informed and random deletions.
        """

        # Since 0 is the 'most important' pixel, in ranked_explanation,
        # we can invert it by simply by ranking it again (so 0 becomes least important)
        inverted_importance_ranking = helpers.utils.rank_pixel_importance(self.exp.ranked_explanation)

        (informed_pres_conf, random_pres_conf), imgs_tuple = self._perform_check(
            inverted_importance_ranking, deletion_method, 1-proportion, n_random_rankings, random_seed,
        )  # we thereby delete the bottom (1-proportion) of pixels

        # drop in acc vs random = (rand_pres_acc - org_acc) - (pres_acc - org_acc)
        #                       = rand_pres_acc - pres_acc
        # We want pres_acc to remain high (we only removed non-important stuff),
        # while rand_pres_acc should be low (we accidentally removed important stuff)
        # After inverting (for alignment with deletion check): the best score is 1, and the worst is -1.
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
            proportion: float,
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
            importance_ranking, deletion_method, proportion, n_random_rankings, random_seed,
        )
        return self._predict_on_deleted(informed_deletions, random_deletions), (informed_deletions, random_deletions)

    def delete_from_input(
            self,
            importance_ranking: Float[np.ndarray, "n_samples height width"],
            deletion_method: deletion.METHODS,
            proportion: float,
            n_random_rankings: int,
            random_seed: int,
    ) -> tuple[
        Float[np.ndarray, "n_samples channels height width"],
        Float[np.ndarray, "n_random_rankings n_samples channels height width"],
    ]:
        num_pixels = self.exp.input.shape[-2] * self.exp.input.shape[-1]
        imgs_with_deletions = deletion.delete_top_k_important(
            self.exp.input, importance_ranking, proportion*num_pixels, method=deletion_method,
        )

        if self.visualise:
            helpers.plotting.show_image(imgs_with_deletions, final_fig_size=(8, (self.n_samples+8)//8))
            plt.suptitle(f"Explanation informed deletion/preservation (proportion={proportion})")
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
                self.exp.input, random_rankings, proportion*num_pixels, method=deletion_method,
            )

        if self.visualise:
            # show each different random ranking on the 0th image
            helpers.plotting.show_image(imgs_with_random_deletions[:, 0],
                                        final_fig_size=(8, (n_random_rankings+8)//8 + .75))
            plt.suptitle(f"{n_random_rankings} random deletion/preservation rounds (proportion={proportion})")
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

        informed_confidences = informed_outputs[np.arange(self.n_samples), original_prediction]
        print(f"Confidence after informed: "
              f"{informed_confidences}") if self.visualise else None

        random_confidences = random_outputs[np.arange(self.n_samples), original_prediction]
        print(f"Confidence after random: "
              f"{random_confidences}") if self.visualise else None

        return informed_confidences, random_confidences
