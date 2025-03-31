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


class GradCAMBase(Explainer):
    def __init__(
            self,
            cam_type: t.Union[t.Type[OriginalGradCAM], t.Type[OriginalKPCACAM]],
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
        super().explain(x)

        gradcam_explainer = self.cam_type(
            self.model, target_layers=self.model.get_explanation_target_layers(),
            reshape_transform=self.model.reshape_transform
        )

        outputs = []
        cam_batch_size = self.batch_size
        if cam_batch_size == 0:  # if no batch size given, just pass the whole tensor
            cam_batch_size = x.shape[0]

        for batch_input in helpers.utils.make_device_batches(x, cam_batch_size, self.device):
            cam_output: np.ndarray = gradcam_explainer(
                input_tensor=batch_input,
                # futuretodo: add these as options (see https://jacobgil.github.io/pytorch-gradcam-book/introduction.html#smoothing-to-get-nice-looking-cams)
                aug_smooth=False, eigen_smooth=False,
                targets=None  # no targets specified, so uses the model's output/most confident prediction
            )
            outputs.append(cam_output)

        cam_output = np.concatenate(outputs, axis=0)
        self.explanation = cam_output  # no additional processing needed (already numpy with correct channels)
        self.save_state()


class GradCAM(GradCAMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(OriginalGradCAM, *args, **kwargs)


class KPCACAM(GradCAMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(OriginalKPCACAM, *args, **kwargs)
