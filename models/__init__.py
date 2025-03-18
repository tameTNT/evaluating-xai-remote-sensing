import typing as t

from helpers import log
from . import core
from . import resnet
from . import convnext

logger = log.get_logger("main")

# fixme: add Swin Transformer, ResNet101
MODEL_NAMES = t.Literal["ResNet50", "ConvNeXtTiny", "ConvNeXtSmall", "ConvNeXtBase"]


def get_model_type(
        name: MODEL_NAMES,
) -> t.Type[core.FreezableModel]:
    """
    Get the model type corresponding to the given name.
    :param name: One of the model names in MODEL_NAMES.
    :return: A callable model type used to instantiate a model.
    """

    logger.debug(f"Attempting to load {name} model...")
    if name == "ResNet50":
        m = resnet.ResNet50
    elif name == "ConvNeXtTiny":
        m = convnext.ConvNeXtTiny
    elif name == "ConvNeXtSmall":
        m = convnext.ConvNeXtSmall
    elif name == "ConvNeXtBase":
        m = convnext.ConvNeXtBase
    else:
        logger.error(f"Invalid model name ({name}) provided to get_model_type. "
                     f"Must be one of {t.get_args(MODEL_NAMES)}.")
        raise ValueError(f"Model {name} does not exist.")

    return m
