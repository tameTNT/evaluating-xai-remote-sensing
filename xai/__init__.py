import json
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Int, Float

import helpers

logger = helpers.log.get_logger("main")

BASE_OUTPUT_PATH = Path("~/l3_project/xai_output").expanduser()
logger.debug(f"Explanation default output path set to {BASE_OUTPUT_PATH}.")


class Explainer:
    """
    Base class for all explainers.
    The default save path is BASE_OUTPUT_PATH / {model.__class__.__name__}{.npz, .json}
    """
    def __init__(
            self,
            model: torch.nn.Module,
            save_path: Path = Path(""),
            attempt_load: bool = False
    ):
        self.model = model
        self.device = helpers.utils.get_model_device(model)

        self.save_path = (BASE_OUTPUT_PATH / save_path).resolve()
        self.save_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"self.save_path of {self.__class__.__name__} set to {self.save_path}.")

        self.npz_path = self.save_path / f"{self.model.__class__.__name__}.npz"
        self.json_path = self.save_path / f"{self.model.__class__.__name__}.json"

        self.input = torch.tensor(0)
        self.args = dict()
        # All explanations should attribute one value to each pixel of each image in the batch
        self.explanation: Float[np.ndarray, "n_samples height width"] = np.ndarray(0)

        if attempt_load:
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

    def explain(
            self,
            x: torch.Tensor,
            **kwargs
    ):
        logger.info(f"Generating explanations in {self.__class__.__name__} "
                    f"for x.shape={x.shape}.")
        self.input = x
        self.args = kwargs

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

        json.dump(self.args, self.json_path.open("w+"))

        logger.debug(f"Saved {self.__class__.__name__}'s explanation "
                     f"to {self.npz_path}.")

    def load_state(self):
        """
        Loads self.input, self.args and self.explanation from self.npz_path
        """

        logger.debug(f"Attempting to load explanation from "
                     f"{self.npz_path} to {self.__class__.__name__}.")
        with np.load(self.npz_path) as data:  # type: dict[str, np.ndarray]
            self.input = torch.from_numpy(data["explanation_input"])
            self.explanation = data["explanation"]

        self.args = json.load(self.json_path.open("r"))
