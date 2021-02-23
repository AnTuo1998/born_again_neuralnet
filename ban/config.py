# -*- coding: utf-8 -*-
from ban.models.mlp import MLP
from ban.models.lenet import LeNet
from ban.models.resnet import resnet18
from ban.models.densenet import densenet121
from ban.models.mobilenet import mobilenet_v2
from ban.models.shuffle import shufflenet_v2_x0_5
from ban.models.squeeze import squeezenet1_0
"""
add your model.
from your_model_file import Model
model = Model()
"""

# model = ResNet50()


def get_model():
    model = MLP()
    return model
