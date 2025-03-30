import logging

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

logger: logging.Logger = helpers.log.main_logger


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
            num_mp_processes: int = 1,
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
        :param num_mp_processes: Number of separate processes to use for multiprocessing. If <= 1, no multiprocessing is used.
        :return:
        """

        super().explain(x, max_evals=max_evals, shap_batch_size=shap_batch_size,
                        blur_size=blur_size, num_mp_processes=num_mp_processes)

        np01_x = einops.rearrange((x + 1) / 2, "b c h w -> b h w c").numpy(force=True)

        blur_str = f"blur({blur_size[0]},{blur_size[1]})"
        blur_masker = shap.maskers.Image(blur_str, np01_x[0].shape)

        if num_mp_processes > 1:  # fixme: this is not working yet?? it takes just as long as single threaded
            # multiprocessing approach based on https://github.com/shap/shap/issues/77#issuecomment-2105595557
            per_process_bs = int(np.ceil(len(x) / num_mp_processes))  # calculate batch size per process
            if per_process_bs == 0:
                per_process_bs = 1  # ensure at least 1 image per process

            batch_start_idxs = range(0, len(x), per_process_bs)
            batch_end_idxs = range(per_process_bs, len(x) + per_process_bs, per_process_bs)
            num_processes = len(batch_start_idxs)
            x_arg_mp_batches = [(
                # this is a tuple of args to pass to explain_via_partition_shap
                np01_x[start_idx:end_idx], blur_masker, max_evals, shap_batch_size//num_processes,
                # each process can only use a fraction of the batch_size since sharing same GPU
                self.device, self.batch_size//num_processes, self.model, logger.name
            ) for start_idx, end_idx in zip(batch_start_idxs, batch_end_idxs)]

            logger.info(f"🙏 Beginning multi-process SHAP evaluation with "
                        f"{num_processes} processes with data size at most {per_process_bs} each.")
            ctx: mp.context.SpawnContext = mp.get_context("spawn")  # use spawn to allow CUDA use for each process
            with ctx.Pool(processes=num_processes) as pool:  # use all available cores/cpus
                explainer_results: list[shap.Explanation] = pool.starmap(explain_via_partition_shap, x_arg_mp_batches)

            logger.info("😇 Finished multi-process SHAP evaluation.")
            shap_values = np.concatenate([mp_result.values for mp_result in explainer_results])
        else:  # single process approach
            logger.info("😇 Beginning single-process SHAP evaluation.")
            shap_values = explain_via_partition_shap(
                np01_x, blur_masker, max_evals, shap_batch_size,
                self.device, self.batch_size, self.model, logger.name
            ).values

        # only save the most confident prediction (sorted to be first by outputs arg)
        # summing over the image colour/band channels (final axis of output)
        self.explanation = shap_values[..., 0].sum(-1)
        self.save_state()


# functions are top level to enable pickling for multiprocess(ing)
def explain_via_partition_shap(
        np01_x: np.ndarray, masker, max_evals, shap_batch_size,
        self_device, self_split_batch_size, self_model, top_logger_name
):
    # todo: this doesn't work yet? doesn't log to the main logger
    # top_logger = logging.getLogger(top_logger_name)
    # top_logger.debug(f"🧵 New dedicated PartitionExplainer process (id={os.getpid()}) started with shap function "
    #                  f"batch size of {shap_batch_size} and model evaluation batch size of {self_split_batch_size}.")
    # print(f"🧵 New dedicated PartitionExplainer process (id={os.getpid()}) spawned "
    #       f"with x of size {len(np01_x)}, shap function batch size of {shap_batch_size} "
    #       f"and model evaluation batch size of {self_split_batch_size}.",
    #       flush=True)

    predict_fn = partial(predict_function, device=self_device, max_batch_size=self_split_batch_size, model=self_model)
    # noinspection PyTypeChecker
    explainer = shap.PartitionExplainer(predict_fn, masker)
    # noinspection PyUnresolvedReferences
    output = explainer(
        np01_x,
        silent=True,
        max_evals=max_evals,
        batch_size=shap_batch_size,
        # order from most confident prediction (left) to lowest
        outputs=shap.Explanation.argsort.flip[:1],
    )
    # print(f"😇 Finished PartitionExplainer process (id={os.getpid()}) with {len(output.values)} values.",)
    return output


def predict_function(np_imgs: np.ndarray, device, max_batch_size, model):
    model_input_imgs: torch.Tensor = einops.rearrange(
        torch.from_numpy(np_imgs * 2) - 1, "b h w c -> b c h w"
    ).to(device)

    # strictly enforce max batch size in case this function is misused by shap explainer
    # don't want to risk any COM errors 😬
    if model_input_imgs.shape[0] > max_batch_size:
        outputs = []
        for img_batch in helpers.utils.make_device_batches(model_input_imgs, max_batch_size, device):
            batch_output: torch.Tensor = model(img_batch)
            outputs.append(batch_output)
        model_output = torch.cat(outputs, dim=0)
    else:
        model_output: torch.Tensor = model(model_input_imgs)

    return model_output.numpy(force=True)
