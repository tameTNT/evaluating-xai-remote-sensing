import json
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Int, Float

import helpers

logger = helpers.log.get_logger("main")

BASE_OUTPUT_PATH = helpers.env_var.get_xai_output_root()
logger.debug(f"Explanation default loading/output path set to {BASE_OUTPUT_PATH}.")


class Explainer:
    """
    Base class for all explainers.
    The default save_path is BASE_OUTPUT_PATH / self.__class__.__name__ / {model.__class__.__name__}{.npz, .json}
    If extra_path is provided, save path is BASE_OUTPUT_PATH / extra_path / self.__class__.__name__ / "
    """
    def __init__(
            self,
            model: torch.nn.Module,
            extra_path: Path = Path(""),
            attempt_load: torch.Tensor = None,
    ):
        self.model = model
        self.device = helpers.utils.get_model_device(model)

        self.save_path = (BASE_OUTPUT_PATH / extra_path / self.__class__.__name__).resolve()
        self.save_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"self.save_path of {self.__class__.__name__} set to {self.save_path}.")

        # e.g. BASE_OUTPUT_PATH / EuroSATRGB / SHAP / ResNet50.npz
        self.npz_path = self.save_path / f"{self.model.__class__.__name__}.npz"
        self.json_path = self.save_path / f"{self.model.__class__.__name__}.json"

        self.input = torch.tensor(0).to(self.device)
        self.kwargs = dict()
        # All explanations should attribute one value to each pixel of each image in the batch
        self.explanation: Float[np.ndarray, "n_samples height width"] = np.ndarray(0)

        self.attempt_load = attempt_load
        if self.attempt_load is not None:
            self.attempt_load = self.attempt_load.to(self.device)
            try:
                self.load_state()
            except FileNotFoundError:
                logger.warning(f"Failed to load existing explanation from "
                               f"{self.npz_path} and {self.json_path}. "
                               f"Using null values.")

    @property
    def ranked_explanation(self) -> Int[np.ndarray, "n_samples height width"]:
        """
        Convert pixel importance to rank pixel importance (0 = most important)
        for each sample image in self.explanation.
        """

        return helpers.utils.rank_pixel_importance(self.explanation)

    def has_explanation_for(self, x: torch.Tensor) -> bool:
        """
        Returns True if the explanation has been generated for a specific input x.
        """

        if self.explanation is not None:
            return torch.equal(self.input, x.to(self.device))
        else:
            return False

    def explain(
            self,
            x: torch.Tensor,
            **kwargs
    ):
        logger.info(f"Generating explanations in {self.__class__.__name__} "
                    f"for x.shape={x.shape} with kwargs={kwargs}.")
        self.input = x.to(self.device)
        self.kwargs = kwargs

    def save_state(self):
        """
        Saves self.input, self.args and self.explanation to
        self.save_path / '{self.model.__class__.__name__}.npz'
        as a compressed npz file.
        """

        np.savez_compressed(
            self.npz_path,
            explanation_input=self.input.numpy(force=True),
            explanation=self.explanation,
        )

        json.dump(self.kwargs, self.json_path.open("w+"))

        logger.debug(f"Saved {self.__class__.__name__}'s explanation "
                     f"to {self.npz_path}.")

    def load_state(self):
        """
        Loads self.input, self.args and self.explanation from self.npz_path
        """

        logger.debug(f"Attempting to load explanation from "
                     f"{self.npz_path} to {self.__class__.__name__}.")
        with np.load(self.npz_path) as data:  # type: dict[str, np.ndarray]
            self.input = torch.from_numpy(data["explanation_input"]).to(self.device)
            temp = data["explanation"]

        diff = (self.input - self.attempt_load).abs().sum()
        if diff < 1e-5:
            logger.debug(f"Loaded input (shape={self.input.shape}) matches the provided check input "
                         f"(shape={self.attempt_load.shape}) with diff={diff}.")
            self.explanation = temp
        else:
            logger.warning(
                f"Loaded input (shape={self.input.shape}) does not match the provided check input "
                f"(shape={self.attempt_load.shape}) with diff={diff}. Using null values."
            )
        self.kwargs = json.load(self.json_path.open("r"))
