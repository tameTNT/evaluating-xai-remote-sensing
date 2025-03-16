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
            target_layers: list[torch.nn.Module] = None,
    ):
        assert target_layers is not None, "target_layers must be provided"

        super().explain(x)  # don't pass target_layers since not json serializable

        gradcam_explainer = GradCAMBase(self.model, target_layers=target_layers)
        cam_output = gradcam_explainer(
            input_tensor=x,
            aug_smooth=False, eigen_smooth=False,
            targets=None
        )

        self.explanation = cam_output  # no additional processing needed
        self.save_state()
