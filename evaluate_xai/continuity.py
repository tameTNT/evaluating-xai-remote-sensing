import typing as t

import matplotlib.pyplot as plt
import numpy as np
import torch
import einops

import helpers
from . import Co12Metric, Similarity

logger = helpers.log.main_logger


class Continuity(Co12Metric):
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
        # Use numpy random generator for reproducibility
        np_rng = np.random.default_rng(random_seed)
        perturbation = (torch.from_numpy(np_rng.normal(size=self.exp.input.shape))
                        .to(self.exp.device, dtype=self.exp.input.dtype))
        noisy_samples = (self.exp.input + degree * perturbation).clamp(-1, 1)
        if self.visualise:
            stacked_samples = einops.rearrange(
                torch.stack([self.exp.input, noisy_samples]), "i n c h w -> n c (i h) w")
            helpers.plotting.show_image(stacked_samples)
            plt.title(f"Continuity Perturbation (degree={degree})")
            plt.show()

        exp_for_perturbed = self.get_sub_explainer("perturbed", noisy_samples)

        if self.visualise:
            self.compare_sub_explainer(
                exp_for_perturbed,
                title=f"Explanation on original/perturbed input (degree={degree})"
            )

        # Check if any model predictions changed - not fair to compare explanations for these
        n_samples = self.exp.input.shape[0]
        model_output = self.run_model(torch.cat([self.exp.input, noisy_samples]))
        original_preds: np.ndarray[int] = model_output[:n_samples].argmax(1)
        perturbed_preds: np.ndarray[int] = model_output[n_samples:].argmax(1)

        # Only compare the similarity of explanations where the model's predictions
        # didn't change, since here we're only interested in perturbations' effect on the *explanation*.
        # If the model's prediction changed then obviously the explanation *should* be very different.

        # noinspection PyTypeChecker
        same_preds: np.ndarray[bool] = perturbed_preds == original_preds
        if not same_preds.all():
            changed_idxs = np.flatnonzero(~same_preds)
            logger.info(f"Perturbations changed the model's prediction for "
                        f"idxs={changed_idxs} (n={len(changed_idxs)}/{len(same_preds)}).")

        return Similarity(self.exp, exp_for_perturbed, mask=same_preds)
