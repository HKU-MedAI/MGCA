import torch.nn as nn
from torchvision import models as models_2d
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


################################################################################
# ResNet Family
################################################################################


def resnet_18(pretrained=True):
    model = models_2d.resnet18(weights=ResNet18_Weights.DEFAULT)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_34(pretrained=True):
    model = models_2d.resnet34(weights=ResNet34_Weights.DEFAULT)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_50(pretrained=True):
    model = models_2d.resnet50(weights=ResNet50_Weights.DEFAULT)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024
