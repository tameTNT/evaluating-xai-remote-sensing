import typing as t

from pytorch_grad_cam import GradCAM as OriginalGradCAM
from pytorch_grad_cam import KPCA_CAM as OriginalKPCACAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
from jaxtyping import Float

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
        cam_output = gradcam_explainer(
            input_tensor=x,
            aug_smooth=False, eigen_smooth=False,  # todo: add these as options
            targets=None
        )  # if no targets specified, uses the model's output

        self.explanation = cam_output  # no additional processing needed
        self.save_state()


class GradCAM(GradCAMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(OriginalGradCAM, *args, **kwargs)


class KPCACAM(GradCAMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(OriginalKPCACAM, *args, **kwargs)
