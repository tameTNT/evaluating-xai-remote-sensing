import typing as t

import torch.nn as nn
import torchvision
from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b

from models.core import Model
import helpers

logger = helpers.log.main_logger
AVAILABLE_SIZES = t.Literal["tiny", "small", "base"]


class SwinTransformerTemplate(Model):
    expected_input_dim = 256
    input_layers_to_train = 1  # train just the Conv2d layer at the top

    def __init__(
            self,
            pretrained: bool,
            n_input_bands: int,
            n_output_classes: int,
            *args,
            size: AVAILABLE_SIZES = "tiny",
            **kwargs
    ):
        """
        Initialise a SwinTransformer model with:
            - the input layer replaced to accept the desired number of input bands.
            - the final linear layer replaced to output the desired number of classes.
        """

        super().__init__(pretrained, n_input_bands, n_output_classes, *args, **kwargs)

        if size == "tiny":
            swin_v2_constructor = swin_v2_t
            weights = torchvision.models.Swin_V2_T_Weights
        elif size == "small":
            swin_v2_constructor = swin_v2_s
            weights = torchvision.models.Swin_V2_S_Weights
        elif size == "base":
            swin_v2_constructor = swin_v2_b
            weights = torchvision.models.Swin_V2_B_Weights
        else:
            raise ValueError(f"size must be one of {t.get_args(AVAILABLE_SIZES)}; not '{size}'.")

        if self.pretrained:
            self.model = swin_v2_constructor(weights=weights.IMAGENET1K_V1)
        else:
            self.model = swin_v2_constructor(weights=None)

        logger.debug(f"Model {self.__class__.__name__} initialised "
                     f"{'with' if self.pretrained else 'without'} pretrained weights")

        # modify model after loading pretrained weights (this is why we don't use the num_classes option)
        # if necessary, change the input convolution
        if n_input_bands != 3:
            logger.debug(f"Changing input transformer block from 3 to {n_input_bands} input channels.")

            old_input_conv: nn.Conv2d = self.model.features[0][0]
            # noinspection PyTypeChecker
            new_conv = nn.Conv2d(
                n_input_bands, old_input_conv.out_channels,
                kernel_size=old_input_conv.kernel_size, stride=old_input_conv.stride,
            )
            self.model.features[0][0] = new_conv
            self.modified_input_layer = True

        # update the output linear layer
        old_fc: nn.Linear = self.model.head
        logger.debug(f"Changing output linear layer (head) from {old_fc.out_features} "
                     f"to {n_output_classes} output channels.")
        self.model.head = nn.Linear(old_fc.in_features, n_output_classes)

        logger.info(f"Model {self.__class__.__name__} successfully initialised with {n_input_bands} input channels "
                    f"and {n_output_classes} output classes.")

    def yield_layers(self) -> t.Generator[nn.Module, None, None]:
        for part in (self.model.features, self.model.norm, self.model.avgpool, self.model.head):
            for child in part.children():
                yield child

    def forward(self, x):
        return self.model(x)

    def get_explanation_target_layers(self) -> list[nn.Module]:
        return [self.model.norm]


class SwinTransformerTiny(SwinTransformerTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size="tiny", **kwargs)


class SwinTransformerSmall(SwinTransformerTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size="small", **kwargs)


class SwinTransformerBase(SwinTransformerTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size="base", **kwargs)
