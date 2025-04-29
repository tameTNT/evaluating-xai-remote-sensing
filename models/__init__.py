import typing as t

from helpers import log

from .core import Model
from .resnet import ResNet50
from .convnext import ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase
from .swin_transformer import SwinTransformerTiny, SwinTransformerSmall, SwinTransformerBase

logger = log.main_logger

MODEL_NAMES = t.Literal[
    "ResNet50",
    "ConvNeXtTiny", "ConvNeXtSmall", "ConvNeXtBase",
    "SwinTransformerTiny", "SwinTransformerSmall", "SwinTransformerBase"
]


def get_model_type(
        name: MODEL_NAMES,
) -> t.Type[Model]:
    """
    Get the model *type* (the object still needs to be instantiated) corresponding to the given name.

    :param name: One of the model names in MODEL_NAMES.
    :return: A callable model type used to instantiate a model.
    """

    logger.debug(f"Attempting to load {name} model...")
    assert name in t.get_args(MODEL_NAMES), (f"Invalid model name ({name}) provided to get_model_type. "
                                             f"Must be one of {t.get_args(MODEL_NAMES)}.")
    m = globals()[name]
    return m
