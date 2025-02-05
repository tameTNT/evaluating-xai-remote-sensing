import torch.nn as nn
import torchvision


class FineTunedResNet50(nn.Module):
    def __init__(self, num_classes: int):
        """
        Initialise a ResNet-50 model with the final linear layer replaced to output the desired number of classes.
        """

        super().__init__()
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def freeze_layers(self, keep: int):
        """
        Freeze layers (requires_grad = False) from the first input layer leaving the last `keep` layers (inc. output).
        :param keep: Number of layers from output (inc.) to keep unfrozen.
            e.g. keep=1 means only output layer is trainable.
        """

        for dist_from_output, layer in enumerate(reversed(list(self.model.children()))):
            # print(dist_from_output, layer)
            if dist_from_output >= keep:
                for param in layer.parameters():
                    param.requires_grad = False

    def unfreeze_layers(self):
        """
        Unfreeze all layers in the model.
        """

        for param in self.parameters():
            param.requires_grad = True

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
