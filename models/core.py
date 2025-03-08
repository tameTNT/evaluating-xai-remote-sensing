import torch.nn as nn

from helpers import log

logger = log.get_logger("main")


class FreezableModel(nn.Module):
    modified_input_layer = False

    expected_input_dim: int
    input_layers_to_train: int  # the number of input layers from the top to unfreeze when fine-tuning

    def __init__(self, pretrained: bool, n_input_bands: int, n_output_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert n_input_bands > 0, "Number of input bands must be greater than 0."
        assert n_output_classes > 0, "Number of output classes must be greater than 0."

    def freeze_layers(self, keep: int):
        """
        Freeze layers (requires_grad = False) from the first input layer leaving the last `keep` layers (inc. output).
        :param keep: Number of layers from output (inc.) to keep unfrozen.
            e.g. keep=1 means only output layer is trainable.
        """

        n_frozen = 0
        for dist_from_output, layer in enumerate(reversed(list(self.model.children()))):
            # print(dist_from_output, layer)
            if dist_from_output >= keep:
                for param in layer.parameters():
                    param.requires_grad = False
                n_frozen += 1
                logger.debug(f"Froze {layer.__class__.__name__} layer of {self.__class__.__name__}")

        logger.info(f"Froze {n_frozen} layers of {self.__class__.__name__}")

    def unfreeze_layers(self):
        """
        Unfreeze all layers in the model.
        """

        for param in self.parameters():
            param.requires_grad = True
        logger.info(f"Unfroze all layers of {self.__class__.__name__}")

    def unfreeze_input_layers(self, k: int):
        """
        Unfreeze the first k input layers of the model.
        """

        for layer_k, layer in enumerate(self.model.children()):
            if layer_k >= k:
                break
            for param in layer.parameters():
                param.requires_grad = True
            logger.debug(f"Unfroze {layer.__class__.__name__} layer of {self.__class__.__name__}.")

        if not self.modified_input_layer:
            logger.debug(f"Input layers of {self.__class__.__name__} was unfrozen "
                         f"but input layer does not differ from default weights.")
        logger.info(f"Unfroze top {k} input layers of {self.__class__.__name__}.")

    def extra_repr(self):
        """
        Add additional detail on number of frozen layers.
        :return:
        """
        num_frozen = 0
        frozen_layers = []
        for layer in self.model.children():
            for param in layer.parameters():
                if not param.requires_grad:
                    num_frozen += 1
                    frozen_layers.append(layer)
                    break

        return f"> {num_frozen} layers frozen: {', '.join([layer.__class__.__name__ for layer in frozen_layers])} <"
