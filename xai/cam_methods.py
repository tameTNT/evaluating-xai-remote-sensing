import typing as t

from pytorch_grad_cam import GradCAM as OriginalGradCAM
from pytorch_grad_cam import KPCA_CAM as OriginalKPCACAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
from jaxtyping import Float
import numpy as np

import helpers
from xai import Explainer

logger = helpers.log.main_logger


class CAMBase(Explainer):
    def __init__(
            self,
            cam_type: t.Type[t.Union[OriginalGradCAM, OriginalKPCACAM]],
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cam_type = cam_type

    def explain(
            self,
            x: Float[torch.Tensor, "n_samples channels height width"],
            **kwargs,
    ):
        """
        Explains self.model's predictions for the given images, x, using a CAM-based method.
        Requires that self.model.get_explanation_target_layers() is defined.
        """
        super().explain(x)

        gradcam_explainer = self.cam_type(
            self.model, target_layers=self.model.get_explanation_target_layers(),
            reshape_transform=self.model.reshape_transform
        )

        outputs = []
        cam_batch_size = self.batch_size
        if cam_batch_size == 0:  # if no batch size given, just pass the whole Tensor without batching
            cam_batch_size = x.shape[0]

        for batch_input in helpers.utils.make_device_batches(x, cam_batch_size, self.device):
            cam_output: np.ndarray = gradcam_explainer(
                input_tensor=batch_input,
                # futuretodo: add these as function arguments
                #  (see https://jacobgil.github.io/pytorch-gradcam-book/introduction.html#smoothing-to-get-nice-looking-cams)
                aug_smooth=False, eigen_smooth=False,
                targets=None  # no targets specified, so uses the model's output/most confident prediction
            )
            outputs.append(cam_output)

        cam_output = np.concatenate(outputs, axis=0)
        self.explanation = cam_output  # no additional processing needed (already numpy with correct channels)
        self._raw_return = cam_output  # same as explanation (no modifications for CAM methods)
        self.save_state()


class GradCAM(CAMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(OriginalGradCAM, *args, **kwargs)


class KPCACAM(CAMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(OriginalKPCACAM, *args, **kwargs)
