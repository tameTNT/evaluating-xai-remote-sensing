from pytorch_grad_cam import GradCAM as GradCAMBase
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
from jaxtyping import Float

from helpers import log
from xai import Explainer

logger = log.get_logger("main")


class GradCAM(Explainer):
    """
    An Explainer object using the original GradCAM's explanations for a model
    """

    def explain(
            self,
            x: Float[torch.Tensor, "n_samples channels height width"],
            target_layer_func: str = "get_explanation_target_layers",
    ):
        assert hasattr(self.model, target_layer_func), \
            f"{self.__class__.__name__} does not have a method called {target_layer_func}"

        super().explain(x, target_layer_func=target_layer_func)

        gradcam_explainer = GradCAMBase(
            self.model, target_layers=getattr(self.model, target_layer_func)(),
        )
        cam_output = gradcam_explainer(
            input_tensor=x,
            aug_smooth=False, eigen_smooth=False,
            targets=None
        )

        self.explanation = cam_output  # no additional processing needed
        self.save_state()
