from pathlib import Path

import torch
import numpy as np

from helpers import utils, logging

logger = logging.get_logger("main")


class Explainer:
    def __init__(
            self,
            model: torch.nn.Module,
            save_path: Path = Path("~/l3_project/output/"),
    ):
        self.model = model
        self.device = utils.get_model_device(model)

        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.explanation = None

    def explain(
            self,
            x: torch.Tensor,
            **kwargs
    ):
        logger.info(f"Generating explanations in {self.__class__.__name__} "
                    f"for x.shape={x.shape}.")

    def save_explanation(self):
        """
        Saves self.explanation to
        self.save_path / '{self.model.__class__.__name__}.npz'
        as a compressed npz file.
        """

        np.savez_compressed(
            self.save_path / f"{self.model.__class__.__name__}.npz",
            explanation=self.explanation
        )
        logger.debug(f"Saved {self.__class__.__name__}'s explanation "
                     f"to {self.save_path / f'{self.model.__class__.__name__}.npz'}")

    def load_explanation(self):
        """
        Loads self.explanation from self.save_path / '{self.model.__class__.__name__}.npz'
        """

        logger.debug(f"Attempting to load explanation from "
                     f"{self.save_path / f'{self.model.__class__.__name__}.npz'} "
                     f"to {self.__class__.__name__}.")
        with np.load(self.save_path / f"{self.model.__class__.__name__}.npz") as data:
            self.explanation = data["explanation"]
