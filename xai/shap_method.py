import einops
import numpy as np
import shap
import shap.maskers
import torch
from jaxtyping import Float

import helpers
from xai import Explainer

logger = helpers.log.main_logger


class PartitionSHAP(Explainer):
    """
    An Explainer object using PartitionSHAP explanations for a model
    """

    def explain(
            self,
            x: Float[torch.Tensor, "n_samples channels height width"],
            max_evals: int = 10000,
            shap_batch_size: int = 64,
    ):
        """
        Explains the model's predictions for the given images using the PartitionSHAP
        Explainer (shap.PartitionExplainer). Requires a decent amount of memory
        (approx 6GB for n_samples=5, shap_batch_size=64).
        NB: Even though PartitionSHAP generates explanations for all model outputs, only
        the model's most confident prediction is saved.

        :param x: Normalised images in [-1, 1] with shape
          (n_samples, channels, height, width)
        :param max_evals: Maximum number of partition explainer evaluations to
            perform. Effectively controls the 'resolution' of the explanation
            with a factor of 10 approximately doubling the resolution.
        :param shap_batch_size: Batch size for shap evaluation. Does not need to store gradients from each so this can be high.
          e.g. batch_size=5 takes 4m 21s; =32 takes 3m40s; =64 takes 3m34s
        :return:
        """

        super().explain(x, max_evals=max_evals, shap_batch_size=shap_batch_size)

        np01_x = einops.rearrange((x + 1) / 2, "b c h w -> b h w c").numpy(force=True)

        def predict_fn(np_imgs: np.ndarray):
            model_input_imgs = einops.rearrange(torch.from_numpy(np_imgs * 2) - 1, "b h w c -> b c h w").to(self.device)
            if model_input_imgs.shape[0] > self.batch_size:  # enforce batch size in case of shap misbehaving
                outputs = []
                for img_batch in helpers.utils.make_device_batches(
                        model_input_imgs, self.batch_size, self.device):
                    batch_output: torch.Tensor = self.model(img_batch)
                    outputs.append(batch_output)
                model_output = torch.cat(outputs, dim=0)
            else:
                model_output: torch.Tensor = self.model(model_input_imgs)

            return model_output.numpy(force=True)
        # fixme: change blur size based on image
        # todo: use multiprocessing speedup like here: https://github.com/shap/shap/issues/77#issuecomment-2105595557
        blur_masker = shap.maskers.Image("blur(128,128)", np01_x[0].shape)
        explainer = shap.PartitionExplainer(predict_fn, blur_masker)

        # noinspection PyUnresolvedReferences
        shap_values = explainer(
            np01_x,
            max_evals=max_evals,
            batch_size=shap_batch_size,
            # order from most confident prediction (left) to lowest
            outputs=shap.Explanation.argsort.flip[:1],
        )

        # only save the most confident prediction (sorted to be first above)
        # summing over the colour channels (final axis of output)
        self.explanation = shap_values.values[..., 0].sum(-1)
        self.save_state()
