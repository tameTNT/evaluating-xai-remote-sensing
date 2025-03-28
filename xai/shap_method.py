import multiprocess as mp  # use pip's 'multiprocess' instead of 'multiprocessing' to use dill to pickle
import os
from functools import partial

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
            blur_size: tuple[int, int] = (128, 128),
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
        :param blur_size: Size of the blur mask to use for the SHAP Partition explainer.
            This is eventually passed to cv2.blur as the kernel. Defaults to (128, 128). Good for 224x224 images.
        :return:
        """

        super().explain(x, max_evals=max_evals, shap_batch_size=shap_batch_size, blur_size=blur_size)

        np01_x = einops.rearrange((x + 1) / 2, "b c h w -> b h w c").numpy(force=True)

        blur_str = f"blur({blur_size[0]},{blur_size[1]})"
        blur_masker = shap.maskers.Image(blur_str, np01_x[0].shape)

        # multiprocessing approach based on https://github.com/shap/shap/issues/77#issuecomment-2105595557
        num_cpus = os.cpu_count()
        per_process_bs = len(x) // num_cpus
        if per_process_bs <= 0:
            per_process_bs = 1  # ensure at least 1 image per process

        x_mp_batches = [(
            np01_x[start_idx:end_idx], blur_masker, max_evals, shap_batch_size,
            self.device, self.batch_size, self.model
        ) for start_idx, end_idx in zip(range(0, len(x), per_process_bs),
                                        range(per_process_bs, len(x) + per_process_bs, per_process_bs))
        ]
        logger.info(f"Beginning multi-process SHAP evaluation with "
                    f"{len(x_mp_batches)} process batches of size {per_process_bs} each.")
        ctx: mp.context.ForkContext = mp.get_context("fork")
        with ctx.Pool(processes=num_cpus) as pool:  # use all available cores/cpus
            explainer_results: list[shap.Explanation] = pool.starmap(explain_via_partition_shap, x_mp_batches)
            shap_values = np.concatenate([mp_result.values for mp_result in explainer_results])

        # only save the most confident prediction (sorted to be first above)
        # summing over the colour channels (final axis of output)
        self.explanation = shap_values[..., 0].sum(-1)
        self.save_state()


def explain_via_partition_shap(
        np01_x: torch.Tensor, masker, max_evals, shap_batch_size,
        self_device, self_batch_size, self_model
):
    predict_fn = partial(predict_function, device=self_device, batch_size=self_batch_size, model=self_model)
    # noinspection PyTypeChecker
    explainer = shap.PartitionExplainer(predict_fn, masker)
    # noinspection PyUnresolvedReferences
    return explainer(
        np01_x,
        silent=True,
        max_evals=max_evals,
        batch_size=shap_batch_size,
        # order from most confident prediction (left) to lowest
        outputs=shap.Explanation.argsort.flip[:1],
    )


def predict_function(np_imgs: np.ndarray, device, batch_size, model):
    model_input_imgs = einops.rearrange(
        torch.from_numpy(np_imgs * 2) - 1, "b h w c -> b c h w"
    ).to(device)

    if model_input_imgs.shape[0] > batch_size:  # enforce batch size in case this function is misused
        outputs = []
        for img_batch in helpers.utils.make_device_batches(model_input_imgs, batch_size, device):
            batch_output: torch.Tensor = model(img_batch)
            outputs.append(batch_output)
        model_output = torch.cat(outputs, dim=0)
    else:
        model_output: torch.Tensor = model(model_input_imgs)

    return model_output.numpy(force=True)
