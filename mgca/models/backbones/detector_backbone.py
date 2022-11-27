import ipdb
import torch
import torch.nn as nn
from mgca.models.backbones import cnn_backbones
from torch import nn


class ResNetDetector(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()

        model_function = getattr(cnn_backbones, model_name)
        self.model, self.feature_dim, self.interm_feature_dim = model_function(
            pretrained=pretrained
        )

        if model_name == "resnet_50":
            self.filters = [512, 1024, 2048]

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        out3 = self.model.layer2(x)   # bz, 512, 28
        out4 = self.model.layer3(out3)
        out5 = self.model.layer4(out4)

        return out3, out4, out5


if __name__ == "__main__":
    model = ResNetDetector("resnet_50")
    x = torch.rand(1, 3, 224, 224)
    out = model(x)
    ipdb.set_trace()
