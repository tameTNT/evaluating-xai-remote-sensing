import json
import typing as t
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Int, Float

import helpers
from models.core import Model

logger = helpers.log.main_logger

BASE_OUTPUT_PATH = helpers.env_var.get_xai_output_root()
logger.debug(f"Explanation default loading/output path set to {BASE_OUTPUT_PATH}.")

EXPLAINER_NAMES = t.Literal["PartitionSHAP", "GradCAM", "KPCACAM"]


def tolerant_equal(a: torch.Tensor, b: torch.Tensor, eps=1e-5) -> tuple[bool, float]:
    """
    Returns True if the explanation is equal to the given eps.
    """
    diff = (a - b).abs().sum().item()
    return diff < eps, diff
    # return torch.allclose(a, b, atol=eps)


class Explainer:
    def __init__(
            self,
            model: Model,
            extra_path: Path = Path(""),
            attempt_load: torch.Tensor = None,
    ):
        """
        Initialise an Explainer object. Can generate explanations using .explain()
        which can then be accessed via .explanation and .ranked_explanation.

        :param model: The Model (subclass of torch.nn.Module) to explain.
        :param extra_path: An additional string to insert into the save path.
          Usually used to save explanations for different datasets.
          The default save_path is BASE_OUTPUT_PATH / self.__class__.__name__ / {model.__class__.__name__}{.npz, .json}
          If extra_path is provided, save path is BASE_OUTPUT_PATH / extra_path / self.__class__.__name__ / ...
        :param attempt_load: Whether to attempt to load an existing explanation.
          If provided, should be a torch.Tensor which the object will try to load the explanations for.
        """
        self.model = model
        self.device = helpers.utils.get_model_device(model)

        self.extra_path = extra_path
        self.save_path = (BASE_OUTPUT_PATH / self.extra_path / self.__class__.__name__).resolve()
        self.save_path.mkdir(parents=True, exist_ok=True)
        # e.g. BASE_OUTPUT_PATH / EuroSATRGB / PartitionSHAP / ResNet50.npz
        self.npz_path = self.save_path / f"{self.model.__class__.__name__}.npz"
        self.json_path = self.npz_path.with_suffix(".json")
        logger.debug(f"self.save_path of {self.__class__.__name__} set to {self.save_path}.")

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
            equal, diff = tolerant_equal(self.input, x.to(self.device))
            logger.debug(f"self.input and x {'are' if equal else 'are not'} equal with diff={diff}.")
            return equal
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

        # noinspection PyTypeChecker
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

        equal, diff = tolerant_equal(self.input, self.attempt_load)
        if equal:
            logger.debug(f"Loaded input (shape={self.input.shape}) matches the provided check input "
                         f"(shape={self.attempt_load.shape}) with diff={diff}.")
            self.explanation = temp
        else:
            logger.warning(
                f"Loaded input (shape={self.input.shape}) does not match the provided check input "
                f"(shape={self.attempt_load.shape}) with diff={diff}. Using null values."
            )

        # todo: enforce kwargs are the same as given (for explanation generated in same way)
        self.kwargs = json.load(self.json_path.open("r"))
        logger.info(f"Loaded {self.__class__.__name__} object state from {self.npz_path} successfully "
                    f"with kwargs={self.kwargs}.")


def get_explainer_object(
        name: EXPLAINER_NAMES,
        model: Model,
        extra_path: Path = Path(""),
        attempt_load: torch.Tensor = None,
) -> Explainer:
    """
    Returns an explainer object of the specified type.
    :param name: The name of the explainer to load. One of the explainer names in EXPLAINER_NAMES.
    :param model: The model to explain.
    :param extra_path: Extra path to save the explanation to.
    :param attempt_load: Attempt to load an existing explanation for this input.
    :return: An explainer object of the specified type.
    """

    if name == "PartitionSHAP":
        logger.debug("Building PartitionSHAP explainer...")
        from xai.shap_method import PartitionSHAP
        explainer = PartitionSHAP(model, extra_path=extra_path, attempt_load=attempt_load)
    elif name == "GradCAM":
        logger.debug("Building GradCAM explainer...")
        from xai.gradcam import GradCAM
        explainer = GradCAM(model, extra_path=extra_path, attempt_load=attempt_load)
    elif name == "KPCACAM":
        logger.debug("Building KPCACAM explainer...")
        from xai.gradcam import KPCACAM
        explainer = KPCACAM(model, extra_path=extra_path, attempt_load=attempt_load)
    else:
        logger.error(f"Invalid explainer name ({name}) provided to get_explainer_object. "
                     f"Must be one of {t.get_args(EXPLAINER_NAMES)}.")
        raise ValueError(f"Explainer {name} does not exist.")

    logger.info(f"Explainer {explainer.__class__.__name__} loaded.")
    return explainer
