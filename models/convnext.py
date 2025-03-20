import functools
import typing as t

import torch.nn as nn
import torchvision
from torchvision.models import convnext_tiny, convnext_small, convnext_base
from torchvision.models.convnext import LayerNorm2d
from torchvision.ops import Conv2dNormActivation

from models.core import FreezableModel
import helpers

logger = helpers.log.main_logger


class ConvNeXtTemplate(FreezableModel):
    expected_input_dim = 224
    input_layers_to_train = 1  # train just the Conv2dNormActivation layer at the top

    def __init__(
            self,
            pretrained: bool,
            n_input_bands: int,
            n_output_classes: int,
            *args,
            size: t.Literal["tiny", "small", "base"] = "small",
            **kwargs
    ):
        """
        Initialise a ConvNeXt model with:
            - the input layer replaced to accept the desired number of input bands.
            - the final linear layer replaced to output the desired number of classes.
        """

        super().__init__(pretrained, n_input_bands, n_output_classes, *args, **kwargs)

        if size == "tiny":
            convnext_constructor = convnext_tiny
            weights = torchvision.models.ConvNeXt_Tiny_Weights
        elif size == "small":
            convnext_constructor = convnext_small
            weights = torchvision.models.ConvNeXt_Small_Weights
        elif size == "base":
            convnext_constructor = convnext_base
            weights = torchvision.models.ConvNeXt_Base_Weights
        else:
            raise ValueError(f"size must be one of 'tiny', 'small', 'base', or 'base'; "
                             f"not '{size}'.")

        if self.pretrained:
            self.model = convnext_constructor(weights=weights.IMAGENET1K_V1)
        else:
            self.model = convnext_constructor(weights=None)

        logger.debug(f"Model {self.__class__.__name__} initialised "
                     f"{'with' if self.pretrained else 'without'} pretrained weights")

        # modify model after loading pretrained weights
        # if necessary, change the input convolution
        if n_input_bands != 3:
            logger.debug(f"Changing input conv layer from 3 to {n_input_bands} input channels.")

            old_input_conv: Conv2dNormActivation = self.model.features[0]
            new_conv = Conv2dNormActivation(
                n_input_bands,
                old_input_conv.out_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=functools.partial(LayerNorm2d, eps=1e-6),
                activation_layer=None,
                bias=True,
            )
            self.model.features[0] = new_conv
            self.modified_input_layer = True

        # update the output linear layer
        logger.debug(f"Changing output linear layer from to {n_output_classes} output channels.")
        old_fc: nn.Linear = self.model.classifier[-1]
        self.model.classifier[-1] = nn.Linear(old_fc.in_features, n_output_classes)

        logger.info(f"Model {self.__class__.__name__} successfully initialised with {n_input_bands} input channels "
                    f"and {n_output_classes} output classes.")

    def yield_layers(self) -> t.Generator[nn.Module, None, None]:
        for part in (self.model.features, self.model.avgpool, self.model.classifier):
            for child in part.children():
                yield child

    def forward(self, x):
        return self.model(x)

    def get_explanation_target_layers(self) -> list[nn.Module]:
        return [self.model.features[-1][-1]]


class ConvNeXtTiny(ConvNeXtTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size="tiny", **kwargs)


class ConvNeXtSmall(ConvNeXtTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size="small", **kwargs)


class ConvNeXtBase(ConvNeXtTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size="base", **kwargs)
