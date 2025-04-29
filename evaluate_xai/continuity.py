import typing as t

import matplotlib.pyplot as plt
import numpy as np
import torch
import einops

import helpers
from . import Co12Property, Similarity

logger = helpers.log.main_logger


class Continuity(Co12Property):
    """
    "Continuity considers how continuous (i.e., smooth) the explanation function is that is learned
    by the explanation method. A continuous function ensures that small variations in the input, for
    which the model response is nearly identical, do not lead to large changes in the explanation."
    """
    def evaluate(
            self,
            method: t.Literal["perturbation"],
            **kwargs,
    ) -> Similarity:
        super().evaluate(method, **kwargs)

        if method == "perturbation":
            return self._perturbation(**kwargs)
        else:
            raise ValueError(f"Unknown evaluation method: {method}")

    def _perturbation(
            self,
            degree: float = 0.15,
            random_seed: int = 42,
            **kwargs,
    ) -> Similarity:
        """
        Computes the similarity between the explanation of original inputs and perturbed inputs
        while accounting only for cases where the model's predictions remain unchanged after
        perturbations. This process involves introducing Gaussian noise to the inputs
        and calculating the explanation Similarity (higher is better).

        🐌 Since this method requires generating new explanations, it can be slow.

        ⚠️ WARNING: This way of measuring continuity is not very good since it often changes the model's prediction
        (since it only adds one fixed degree of noise) so we often can't compare explanations
        and get a usable metric result. This metric was not used or discussed in our final paper.
        futuretodo: Implement max sensitivity method as in "On the (In)fidelity and Sensitivity of Explanations"
         https://github.com/chihkuanyeh/saliency_evaluation/blob/8eb095575cf5502290a5a32e27163d1aca224580/infid_sen_utils.py#L301

        :param degree: The scaling factor for the perturbation, representing the intensity
            of noise added to the input. Defaults to 0.15.
        :param random_seed: Random seed used for generating perturbations reproducibly. Defaults to 42.
        :return: A Similarity object quantifying the similarity between the explanation of
                 original inputs and perturbed inputs.
        """

        # Use numpy random generator for reproducibility
        np_rng = np.random.default_rng(random_seed)
        perturbation = (torch.from_numpy(np_rng.normal(size=self.exp.input.shape))
                        .to(self.exp.device, dtype=self.exp.input.dtype))
        noisy_samples = self.exp.input + degree * perturbation

        if self.visualise:
            stacked_samples = einops.rearrange(
                torch.stack([self.exp.input, noisy_samples]), "i n c h w -> n c (i h) w")
            helpers.plotting.show_image(stacked_samples)
            plt.title(f"Continuity Perturbation (degree={degree})")
            plt.show()

        # futuretodo: only bother computing the explanation for samples that didn't change f (currently below)
        #  Saves significant computation if explanations take a long time to generate
        exp_for_perturbed = self.get_sub_explainer("perturbed", noisy_samples)

        if self.visualise:
            self.compare_sub_explainer(
                exp_for_perturbed,
                title=f"Explanation on original/perturbed input (degree={degree})"
            )

        # Check if any model predictions changed - not fair to compare explanations for these
        model_output = self.run_model(torch.cat([self.exp.input, noisy_samples]))
        original_preds: np.ndarray = model_output[:self.n_samples].argmax(1)
        perturbed_preds: np.ndarray = model_output[self.n_samples:].argmax(1)

        # Only compare the similarity of explanations where the model's predictions
        # didn't change, since here we're only interested in perturbations' effect on the *explanation*.
        # If the model's prediction changed, then the explanation *should* be very different.

        # noinspection PyTypeChecker
        same_preds: np.ndarray = perturbed_preds == original_preds
        if not same_preds.all():
            changed_idxs = np.flatnonzero(~same_preds)
            logger.info(f"Perturbations changed the model's prediction for "
                        f"idxs={changed_idxs} (n={len(changed_idxs)}/{len(same_preds)}).")

        return Similarity(self.exp, exp_for_perturbed, mask=same_preds)
