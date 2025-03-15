import typing as t

import matplotlib.pyplot as plt
import numpy as np
import torch
import einops

import helpers
from . import Co12Metric, Similarity

logger = helpers.log.get_logger("main")


class Continuity(Co12Metric):
    # todo: add docstrings based on paper

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

        original_explainer = self.exp.__class__
        exp_for_perturbed = original_explainer(
            self.exp.model, self.exp.extra_path / "perturbed", attempt_load=noisy_samples,
        )
        if not exp_for_perturbed.has_explanation_for(noisy_samples):
            # only generate explanation if no existing one
            logger.info(f"No existing explanation for self.exp.input in exp_for_perturbed. "
                        f"Generating a new one.")
            # Use the same kwargs as the original explainer
            exp_for_perturbed.explain(noisy_samples, **self.exp.kwargs)
        else:
            logger.info(f"Existing explanation found for self.exp.input in exp_for_perturbed.")

        if self.visualise:
            stacked_explanations = einops.rearrange(
                np.stack([self.exp.ranked_explanation, exp_for_perturbed.ranked_explanation]),
                "i n h w -> n (i h) w")
            helpers.plotting.visualise_importance(stacked_samples, stacked_explanations,
                                                  alpha=.2, with_colorbar=False)
            plt.title(f"Explanation on original/perturbed input (degree={degree})")
            plt.show()

        # Check if any model predictions changed - not fair to compare explanations for these
        n_samples = self.exp.input.shape[0]
        model_output = self.run_model(
            torch.cat([self.exp.input, noisy_samples]).numpy(force=True)
        )
        original_preds: np.ndarray[int] = model_output[:n_samples].argmax(1)
        perturbed_preds: np.ndarray[int] = model_output[n_samples:].argmax(1)

        same_preds: np.ndarray[bool] = perturbed_preds == original_preds

        return Similarity(self.exp, exp_for_perturbed, mask=same_preds)
