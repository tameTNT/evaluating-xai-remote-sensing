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
    Returns True if the explanations are equal within the given eps.
    Also allows the inputs to be different sizes (this returns False, -1 immediately).
    """
    if a.shape != b.shape:
        logger.debug(f"Shapes of a ({a.shape}) and b ({b.shape}) are not equal.")
        return False, -1
    diff = (a - b).abs().sum().item()
    return diff < eps, diff
    # return torch.allclose(a, b, atol=eps)


class Explainer:
    """
    A base class for all explainers.
    Explainers are wrappers around implementations of explainers (e.g. PartitionSHAP, GradCAM, etc.),
    which provide a consistent interface for generating and saving explanations.
    Explainer objects are initialised with a model and potentially an input tensor (via attempt_load).
    Explanations are generated using the .explain() method, which takes an input tensor.
    The generated explanations are stored in the explanation property and enable access to ranked_explanation too.
    """
    def __init__(
            self,
            model: Model,
            extra_path: Path = Path(""),
            attempt_load: Float[torch.Tensor, "n_samples channels height width"] = None,
            batch_size: int = 0,
    ):
        """
        Initialise an Explainer object. Can generate explanations using .explain()
        which can then be accessed via .explanation and .ranked_explanation.
        The root path, BASE_OUTPUT_PATH, is given by `helpers.env_var.get_xai_output_root()`.

        :param model: The Model (subclass of torch.nn.Module) to explain.
        :param extra_path: An additional string to insert into the save path.
            Usually used to save explanations for different datasets.
            The default save_path is BASE_OUTPUT_PATH / self.__class__.__name__ / {model.__class__.__name__}{.npz, .json}
            If extra_path is provided, save_path is BASE_OUTPUT_PATH / extra_path / self.__class__.__name__ / ...
        :param attempt_load: Whether to attempt to load an existing explanation.
            If provided, it should be a Tensor input which the Explainer will try to load the explanations for.
        :param batch_size: Batch size to use when passing images to the underlying model or explainer object.
            If not given or 0, this is set to len(x) when .explain(x, ...) is called.
        """

        self.model = model
        self.device = helpers.utils.get_model_device(model)

        self.extra_path = extra_path

        self.input = torch.tensor(0).to(self.device)
        self.kwargs = dict()
        # All explanations should attribute just one value to each pixel of each image in the batch
        self._explanation: Float[np.ndarray, "n_samples height width"] = np.ndarray(0)
        self._raw_return = None  # not saved, just used for debugging

        # General batch size to use if Explainer doesn't natively support (e.g. GradCAM) but
        # requires gradient store, etc. which takes up a lot of memory, limiting possible batch size.
        # Note how this is different from e.g. the shap_batch_size kwarg for PartitionSHAP
        self.batch_size = batch_size

        self.attempt_load = attempt_load
        if self.attempt_load is not None:
            self.attempt_load = self.attempt_load.to(self.device)
            try:
                self.load_state()
            except FileNotFoundError:
                logger.info(f"Couldn't load existing explanations from "
                            f"{self.npz_path} and {self.json_path}. "
                            f"Using null values.")

    def __repr__(self):
        return (f"{self.__class__.__name__}(model={self.model.__class__.__name__}, input_shape={self.input.shape}, "
                f"has_explanation={self.explanation is not None}, save_path='{self.save_path}')")

    @property
    def save_path(self) -> Path:
        """The save directory for the explanation files (may not exist yet)."""
        return (BASE_OUTPUT_PATH / self.extra_path / self.__class__.__name__).resolve()

    @property
    def npz_path(self) -> Path:
        # delay the creation of the directory until we save to it
        # (avoids creating the directory unnecessarily if we call .parent as we do in evaluate_xai/contrastivity.py)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
            logger.debug(f"Created save_path of {self.__class__.__name__}: {self.save_path}.")

        # e.g. BASE_OUTPUT_PATH / EuroSATRGB / b000 / GradCAM / ResNet50.npz
        return self.save_path / f"{self.model.__class__.__name__}.npz"

    @property
    def json_path(self) -> Path:
        return self.npz_path.with_suffix(".json")

    @property
    def ranked_explanation(self) -> Int[np.ndarray, "n_samples height width"]:
        """
        Convert pixel importance to rank pixel importance (0 = most important)
        for each sample image in self.explanation, returning this ranked explanation.
        """

        return helpers.utils.rank_pixel_importance(self.explanation)

    def has_explanation_for(self, x: torch.Tensor) -> bool:
        """
        Returns True if an explanation has been generated for a specific input x.
        """

        if self.explanation is not None:
            equal, diff = tolerant_equal(self.input, x.to(self.device))
            logger.debug(f"self.input and x {'are' if equal else 'are not'} ≈equal with diff={diff}.")
            return equal
        else:
            return False

    def explain(
            self,
            x: Float[torch.Tensor, "n_samples channels height width"],
            **kwargs
    ):
        """
        Explains the model's predictions for the given images using the explainer.
        Sets self.explanation and saves the new state via self.save_state().
        Does not return anything.

        :param x: Input to explain. This will be moved to self.device.
        :param kwargs: Additional keyword arguments are left to the specific Explainer's implementation.
        """
        logger.info(f"Generating explanations in {self.__class__.__name__} "
                    f"for x.shape={x.shape} with kwargs={kwargs} (self.kwargs updated). "
                    f"General Explainer batch size (with gradients) is {self.batch_size}.")
        self.input = x.to(self.device)
        self.kwargs = kwargs
        self.batch_size = self.batch_size or len(x)  # if self.batch_size is currently 0, naively set it to len(x)

    @property
    def explanation(self) -> Float[np.ndarray, "n_samples height width"]:
        """The explanation for the input images, x, provided at the last call to explain(...)."""
        return self._explanation

    @explanation.setter
    def explanation(self, val: Float[np.ndarray, "n_samples height width"]):
        """Sets the _explanation property to the given value, checking for all 0 explanations (bad!)."""

        all_0_exps = np.where(np.all(val == 0, axis=(1, 2)))[0]
        if len(all_0_exps) > 0:
            logger.warning(f"{self.__class__.__name__} Explanation contains "
                           f"{len(all_0_exps)} all 0 explanations at idx={all_0_exps}.")

        self._explanation = val

    def save_state(self):
        """
        Saves self.input, self.kwargs and self.explanation to
        self.save_path / '{self.model.__class__.__name__}.npz' (i.e. self.npz_path)
        as a compressed npz file.
        """

        np.savez_compressed(
            self.npz_path,
            explanation_input=self.input.numpy(force=True),
            explanation=self.explanation,
        )

        json.dump(self.kwargs, self.json_path.open("w+"))

        logger.debug(f"Saved {self.__class__.__name__}'s explanation "
                     f"to {self.npz_path} (kwargs={self.kwargs}).")

    def load_state(self, _force: bool = False):
        """
        Loads self.input, self.kwargs and self.explanation from self.npz_path.
        If _force is True, the loaded input is *not* checked against self.attempt_load.
        """

        logger.debug(f"Attempting to load explanation from "
                     f"{self.npz_path} to {self.__class__.__name__}.")
        with np.load(self.npz_path) as data:  # type: dict[str, np.ndarray]
            self.input = torch.from_numpy(data["explanation_input"]).to(self.device)
            temp = data["explanation"]

        if _force:
            equal, diff = True, 0
            als = "N/A"
        else:
            equal, diff = tolerant_equal(self.input, self.attempt_load)
            als = self.attempt_load.shape

        if equal:
            logger.debug(f"Loaded input (shape={self.input.shape}) matches the provided check input "
                         f"(shape={als}) with diff={diff}.")
            self.explanation = temp
        else:
            logger.warning(
                f"Loaded input (shape={self.input.shape}) does not match the provided check input "
                f"(shape={als}) with diff={diff}. Using null values."
            )

        # futuretodo: enforce kwargs are the same as given (to ensure explanation was generated in expected way)
        self.kwargs = json.load(self.json_path.open("r"))
        logger.info(f"Loaded {self.__class__.__name__} object state from {self.npz_path} successfully "
                    f"with kwargs set to {self.kwargs}.")

    def force_load(self):
        """
        Populate self.input, self.kwargs and self.explanation from self.npz_path directly.
        Doesn't need to match any expected input.
        """
        # print(f"Loading input, explanation, and kwargs from {self.npz_path}...")
        logger.debug(f"Forcing (with self.attempt_load comparison) to load explanation from "
                     f"{self.npz_path} to {self.__class__.__name__}.")
        self.load_state(_force=True)

    def __or__(self, other: "Explainer") -> "Explainer":
        """
        Combine two compatible Explainer objects together via `c = a | b`.
        Being 'compatible' means they both have an explanation for inputs of the same image sizes
        (the batch size need not be the same) for the same model with the same kwargs.

        :param other: The other Explainer object to combine.
        :return: A new Explainer object with the combined explanations and inputs.
        """

        if not isinstance(other, self.__class__):
            raise TypeError(f"Mismatched classes for or operation: "
                            f"a is {self.__class__.__name__} and b is {other.__class__.__name__}.")
        assert (self.explanation is not None) and (other.explanation is not None), \
            "One or both explainers have no stored explanation."
        assert self.model is other.model, \
            "Explainers are for different models."
        assert self.kwargs == other.kwargs, \
            f"Explainers have different kwargs: a.kwargs={self.kwargs}, b.kwargs={other.kwargs}"

        assert self.input.shape[1:] == other.input.shape[1:], \
            (f"Explainers have different image shapes: "
             f"a.shape={self.input.shape[1:]}, b.shape={other.input.shape[1:]}")
        assert self.explanation.shape[1:] == other.explanation.shape[1:], \
            (f"Explainers have different explanation shapes: "
             f"a.shape={self.explanation.shape[1:]}, b.shape={other.explanation.shape[1:]}")

        new_exp = self.__class__(self.model)
        new_exp.input = torch.cat([self.input, other.input], dim=0)
        new_exp.explanation = np.concatenate([self.explanation, other.explanation], axis=0)
        new_exp.kwargs = self.kwargs
        new_exp.attempt_load = None

        return new_exp


def get_explainer_object(
        name: EXPLAINER_NAMES,
        model: Model,
        extra_path: Path = Path(""),
        attempt_load: torch.Tensor = None,
        batch_size: int = 0,
) -> Explainer:
    """
    Returns an Explainer object of the specified name.
    For a detailed description of the parameters, see the Explainer __init__() method.
    """

    if name == "PartitionSHAP":
        logger.debug("Building PartitionSHAP explainer...")
        from xai.shap_method import PartitionSHAP
        explainer = PartitionSHAP(model, extra_path=extra_path,
                                  attempt_load=attempt_load, batch_size=batch_size)
    elif name == "GradCAM":
        logger.debug("Building GradCAM explainer...")
        from xai.cam_methods import GradCAM
        explainer = GradCAM(model, extra_path=extra_path,
                            attempt_load=attempt_load, batch_size=batch_size)
    elif name == "KPCACAM":
        logger.debug("Building KPCACAM explainer...")
        from xai.cam_methods import KPCACAM
        explainer = KPCACAM(model, extra_path=extra_path,
                            attempt_load=attempt_load, batch_size=batch_size)
    else:
        logger.error(f"Invalid explainer name ({name}) provided to get_explainer_object. "
                     f"Must be one of {t.get_args(EXPLAINER_NAMES)}.")
        raise ValueError(f"Explainer {name} does not exist.")

    logger.info(f"Explainer {explainer.__class__.__name__} loaded.")
    return explainer
