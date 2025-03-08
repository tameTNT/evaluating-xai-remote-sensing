import typing as t

from helpers import logging
from . import core
from . import resnet

logger = logging.get_logger("main")

MODEL_NAMES = ["ResNet50", "ResNet101"]


def get_model_type(
        name: t.Literal["ResNet50", "ResNet101"],
) -> t.Type[core.FreezableModel]:
    """
    Get the model type corresponding to the given name.
    :param name: One of the model names in MODEL_NAMES.
    :return: A callable model type used to instantiate a model.
    """

    if name == "ResNet50":
        logger.debug("Returning ResNet50 model type...")
        m = resnet.ResNet50
    else:
        logger.error(f"Invalid model name ({name}) provided to get_model_type. "
                 f"Must be one of {MODEL_NAMES}.")
        raise ValueError(f"Model {name} does not exist.")

    return m
