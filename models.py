__author__ = "Richard Correro (richard@richardcorrero.com)"


import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import models

from script_utils import arg_is_true, parse_args

BACKBONES = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152
}

BACKBONE_WEIGHTS = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet34": models.ResNet34_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.DEFAULT,
    "resnet101": models.ResNet101_Weights.DEFAULT,
    "resnet152": models.ResNet152_Weights.DEFAULT  
}


class Fire(nn.Module):
    def __init__(
        self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, 
        expand3x3_planes: int
    ) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(
            squeeze_planes, expand1x1_planes, kernel_size=1
        )
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1
        )
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [
                self.expand1x1_activation(self.expand1x1(x)), 
                self.expand3x3_activation(self.expand3x3(x))
            ], 1
        )


class SqueezeNet(nn.Module):
    __name__ = "SqueezeNet"

    DEFAULT_NUM_CHANNLES: int = 4
    DEFAULT_NUM_CLASSES: int = 10
    DEFAULT_DROPOUT: float = 0.5


    def __init__(self) -> None:
        super().__init__()
        args = self.parse_args()
        num_channels: int = args["num_channels"]
        num_classes: int = args["num_classes"]
        dropout: float = args["dropout"]
        self.args = args

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), final_conv, nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--num-channels",
            default=self.DEFAULT_NUM_CHANNLES,
            type=int
        )
        parser.add_argument(
            "--num-classes",
            default=self.DEFAULT_NUM_CLASSES,
            type=int
        )
        parser.add_argument(
            "--dropout",
            default=self.DEFAULT_DROPOUT,
            type=float
        )
        args = parse_args(parser=parser)
        return args

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


class SpectrumNet(SqueezeNet):
    """
    NOTE: There is a typo in the original paper, found here:
    https://www.cs.montana.edu/sheppard/pubs/ijcnn-2019c.pdf
    The number of "1x1 expand planes" in "spectral8" and "spectral9" cannot
    be 385, as reported in Table 1 of that paper. I infer the authors meant to
    write 384.
    """
    __name__ = "SpectrumNet"


    def __init__(self) -> None:
        super().__init__()
        num_channels: int = self.args["num_channels"]
        num_classes: int = self.args["num_classes"]
        dropout: float = self.args["dropout"]
        
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 96, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            Fire(96, 16, 96, 32),
            Fire(128, 16, 96, 32),
            Fire(128, 32, 192, 64),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(256, 32, 192, 64),
            Fire(256, 48, 288, 96),
            Fire(384, 48, 288, 96),
            Fire(384, 64, 384, 128),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(512, 64, 384, 128),
        )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), final_conv, nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class CNNLSTM(nn.Module):
    __name__ = "CNNLSTM"

    DEFAULT_NUM_CHANNLES: int = 3
    DEFAULT_NUM_CLASSES: int = 2
    DEFAULT_DROPOUT: bool = False
    DEFAULT_FREEZE_BACKBONE_PARAMS: bool = True
    DEFAULT_BACKBONE_NAME: str = "resnet101"
    DEFAULT_LSTM_LAYERS: int = 3
    DEFAULT_LSTM_HIDDEN_SIZE: int = 256


    def __init__(self):
        super().__init__()
        args = self.parse_args()
        num_channels: int = args["num_channels"]
        num_classes: int = args["num_classes"]
        dropout: float = args["dropout"]
        freeze_backbone_params: bool = arg_is_true(args["freeze_backbone_params"])
        backbone_name: str = args["backbone"]
        num_layers: int = args["lstm_layers"]
        lstm_hidden_size: int = args["lstm_hidden_size"]

        assert num_channels == 3, f"Must have `num_channels == 3` for model {self.__name__}."
        assert not dropout, "Dropout not implemented for this class."

        self.args = args 

        # weights = BACKBONE_WEIGHTS[backbone_name]
        # self.transforms = weights.transforms

        self.resnet = BACKBONES[backbone_name](pretrained=True)

        if freeze_backbone_params:
            self.resnet.requires_grad_(False)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.resnet.fc.requires_grad_(True)
        self.lstm = nn.LSTM(input_size=300, hidden_size=lstm_hidden_size, num_layers=num_layers)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--num-channels",
            default=self.DEFAULT_NUM_CHANNLES,
            type=int
        )
        parser.add_argument(
            "--num-classes",
            default=self.DEFAULT_NUM_CLASSES,
            type=int
        )
        parser.add_argument(
            "--dropout",
            default=self.DEFAULT_DROPOUT,
            type=float
        )
        parser.add_argument(
            "--freeze-backbone-params",
            default=self.DEFAULT_FREEZE_BACKBONE_PARAMS
        )
        parser.add_argument(
            "--backbone",
            default=self.DEFAULT_BACKBONE_NAME
        )
        parser.add_argument(
            "--lstm-layers",
            default=self.DEFAULT_LSTM_LAYERS,
            type=int
        )
        parser.add_argument(
            "--lstm-hidden-size",
            default=self.DEFAULT_LSTM_HIDDEN_SIZE,
            type=int
        )                 
        args = parse_args(parser=parser)
        return args

       
    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x: torch.Tensor = self.resnet(x_3d[:, t, :, :, :])  
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x                    
