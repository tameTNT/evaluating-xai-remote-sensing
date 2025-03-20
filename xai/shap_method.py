import einops
import numpy as np
import shap
import shap.maskers
import torch
from jaxtyping import Float

from helpers import log
from xai import Explainer

logger = log.main_logger


class PartitionSHAP(Explainer):
    """
    An Explainer object using PartitionSHAP explanations for a model
    """

    def explain(
            self,
            x: Float[torch.Tensor, "n_samples channels height width"],
            max_evals: int = 10000,
            batch_size: int = 96,  # todo: make this an arg
    ):
        """
        Explains the model's predictions for the given images using the PartitionSHAP
        Explainer (shap.PartitionExplainer). Requires a decent amount of memory
        (approx 6GB for n_samples=5, batch_size=64).
        NB: Even though PartitionSHAP generates explanations for all model outputs, only
        the model's most confident prediction is saved.

        :param x: Normalised images in [-1, 1] with shape
          (batch_size, channels, height, width)
        :param max_evals: Maximum number of partition explainer evaluations to
            perform. Effectively controls the 'resolution' of the explanation
            with a factor of 10 approximately doubling the resolution.
        :param batch_size: Batch size for evaluation.
          NB: batch_size=5 takes 4m 21s; =32 takes 3m40s; =64 takes 3m34s
        :return:
        """

        super().explain(x, max_evals=max_evals, batch_size=batch_size)

        np01_x = einops.rearrange((x + 1) / 2, "b c h w -> b h w c").numpy(force=True)

        def predict_fn(np_imgs: np.ndarray):
            model_input_img = einops.rearrange(torch.from_numpy(np_imgs * 2) - 1, "b h w c -> b c h w").to(self.device)
            model_output: torch.Tensor = self.model(model_input_img)

            return model_output.numpy(force=True)

        blur_masker = shap.maskers.Image("blur(128,128)", np01_x[0].shape)
        explainer = shap.PartitionExplainer(predict_fn, blur_masker)

        # noinspection PyUnresolvedReferences
        shap_values = explainer(
            np01_x,
            max_evals=max_evals,
            batch_size=batch_size,
            # order from most confident prediction (left) to lowest
            outputs=shap.Explanation.argsort.flip[:1],
        )

        # only save the most confident prediction (sorted to be first above)
        # summing over the colour channels (final axis of output)
        self.explanation = shap_values.values[..., 0].sum(-1)
        self.save_state()
