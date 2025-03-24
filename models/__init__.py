import typing as t

from helpers import log
from . import core
from . import resnet
from . import convnext
from core import Model
from resnet import ResNet50
from convnext import ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase

logger = log.main_logger

# fixme: add Swin Transformer, ResNet101
MODEL_NAMES = t.Literal["ResNet50", "ConvNeXtTiny", "ConvNeXtSmall", "ConvNeXtBase"]
MODEL_NAMES = t.Literal[
    "ResNet50",  # fixme: add ResNet101
    "ConvNeXtTiny", "ConvNeXtSmall", "ConvNeXtBase",
]


def get_model_type(
        name: MODEL_NAMES,
) -> t.Type[Model]:
    """
    Get the model type corresponding to the given name.
    :param name: One of the model names in MODEL_NAMES.
    :return: A callable model type used to instantiate a model.
    """

    logger.debug(f"Attempting to load {name} model...")
    assert name in t.get_args(MODEL_NAMES), (f"Invalid model name ({name}) provided to get_model_type. "
                                             f"Must be one of {t.get_args(MODEL_NAMES)}.")
    m = globals()[name]
    return m
