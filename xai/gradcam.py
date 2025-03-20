from pytorch_grad_cam import GradCAM as OriginalGradCAM
from pytorch_grad_cam import KPCA_CAM as OriginalKPCACAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
from jaxtyping import Float

from helpers import log
from xai import Explainer

logger = log.main_logger


class GradCAMBase(Explainer):
    def __init__(self, cam_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cam_type = cam_type

    def explain(
            self,
            x: Float[torch.Tensor, "n_samples channels height width"],
            target_layer_func: str = "get_explanation_target_layers",
    ):
        assert hasattr(self.model, target_layer_func), \
            f"{self.__class__.__name__} does not have a method called {target_layer_func}"

        super().explain(x, target_layer_func=target_layer_func)

        gradcam_explainer = self.cam_type(
            self.model, target_layers=getattr(self.model, target_layer_func)(),
        )
        cam_output = gradcam_explainer(
            input_tensor=x,
            aug_smooth=False, eigen_smooth=False,
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
