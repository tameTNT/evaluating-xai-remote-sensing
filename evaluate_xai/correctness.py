import copy
from pathlib import Path

import torch

import helpers
from xai import Explainer
from . import Co12Metric, Similarity


logger = helpers.log.get_logger("main")


class Correctness(Co12Metric):
    def __init__(self, exp: Explainer):
        super().__init__(exp)

    def evaluate(
            self,
            method: str = "model_randomisation",
    ) -> Similarity:
        super().evaluate(method)

        if method == "model_randomisation":
            random_exp = self._model_randomisation()
            return Similarity(self.exp.explanation, random_exp.explanation)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _model_randomisation(self) -> Explainer:
        device = helpers.utils.get_model_device(self.exp.model)
        randomised_model = copy.deepcopy(self.exp.model).to(device)
        self.reset_child_params(randomised_model)
        randomised_model.eval()

        original_explainer = self.exp.__class__
        exp_for_randomised_model = original_explainer(
            randomised_model, Path(f"randomised_{self.exp.__class__.__name__}"), attempt_load=self.exp.attempt_load,
        )

        if not exp_for_randomised_model.has_explanation_for(self.exp.input):
            # only generate explanation if no existing one
            logger.info(f"No existing explanation for self.exp.input in exp_for_randomised_model. "
                        f"Generating a new one.")
            exp_for_randomised_model.explain(self.exp.input, **self.exp.kwargs)
        else:
            logger.info(f"Existing explanation found for self.exp.input in exp_for_randomised_model.")

        return exp_for_randomised_model

    def reset_child_params(self, model: torch.nn.Module):
        """
        Reset all parameters of the model to their defaults **inplace**.
        Adapted from https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        """
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            self.reset_child_params(layer)
