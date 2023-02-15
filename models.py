__author__ = "Richard Correro (richard@richardcorrero.com)"


import argparse
import math

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
    "resnet18": models.ResNet18_Weights,
    "resnet34": models.ResNet34_Weights,
    "resnet50": models.ResNet50_Weights,
    "resnet101": models.ResNet101_Weights,
    "resnet152": models.ResNet152_Weights    
}

try:
    OBJECT_DETECTORS = {
        "fasterrcnn_resnet50_fpn_v2": models.detection.fasterrcnn_resnet50_fpn_v2,
        "fasterrcnn_resnet50_fpn": models.detection.fasterrcnn_resnet50_fpn
    }
except:
    # Cheeky little quick-fix
    OBJECT_DETECTORS = {
        "fasterrcnn_resnet50_fpn": models.detection.fasterrcnn_resnet50_fpn,
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

    IS_OBJECT_DETECTOR = False    

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

    IS_OBJECT_DETECTOR = False    


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


class ResNetConvLSTM(nn.Module):
    __name__ = "ResNetConvLSTM"

    IS_OBJECT_DETECTOR = False    

    DEFAULT_NUM_CHANNLES: int = 3
    DEFAULT_NUM_CLASSES: int = 2
    DEFAULT_DROPOUT: float = 0.0
    DEFAULT_LSTM_DROPOUT: float = 0.0
    DEFAULT_FREEZE_BACKBONE_PARAMS: bool = True
    DEFAULT_BACKBONE_NAME: str = "resnet152"
    DEFAULT_LSTM_LAYERS: int = 3
    DEFAULT_LSTM_HIDDEN_SIZE: int = 256    
    DEFAULT_FC1_OUT: int = 128


    def __init__(self):
        super().__init__()
        args = self.parse_args()
        num_channels: int = args["num_channels"]
        num_classes: int = args["num_classes"]
        dropout: float = args["dropout"]        
        lstm_dropout: float = args["lstm_dropout"]
        freeze_backbone_params: bool = arg_is_true(args["freeze_backbone_params"])
        backbone_name: str = args["backbone"]
        num_layers: int = args["lstm_layers"]
        lstm_hidden_size: int = args["lstm_hidden_size"]
        fc1_out: int = args["fc1_out"]

        assert num_channels == 3, f"Must have `num_channels == 3` for model {self.__name__}."

        self.args = args 

        resnet = BACKBONES[backbone_name](pretrained=True)
        resnet_fc_in_features: int  = resnet.fc.in_features
        resnet.fc = nn.Sequential()
        self.resnet = resnet

        if freeze_backbone_params:
            self.resnet.requires_grad_(False)
     
        self.lstm = nn.LSTM(
            input_size=resnet_fc_in_features, hidden_size=lstm_hidden_size, 
            num_layers=num_layers, dropout=lstm_dropout
        )          
        self.fc1 = nn.Linear(lstm_hidden_size, fc1_out)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(fc1_out, num_classes)


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
            "--lstm-dropout",
            default=self.DEFAULT_LSTM_DROPOUT,
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
        parser.add_argument(
            "--fc1-out",
            default=self.DEFAULT_FC1_OUT,
            type=int
        )        
        args = parse_args(parser=parser)
        return args

       
    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x: torch.Tensor = self.resnet(x_3d[:, t, :, :, :])
                x = x.view(x.size(0), -1) # Flatten output of ResNet
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResNet(nn.Module):
    __name__ = "ResNet"

    IS_OBJECT_DETECTOR = False

    DEFAULT_NUM_CHANNLES: int = 3
    DEFAULT_NUM_CLASSES: int = 2
    DEFAULT_DROPOUT: float = 0.0
    DEFAULT_FREEZE_BACKBONE_PARAMS: bool = True
    DEFAULT_BACKBONE_NAME: str = "resnet152"
    DEFAULT_FC1_OUT: int = 512


    def __init__(self):
        super().__init__()
        args = self.parse_args()
        num_channels: int = args["num_channels"]
        num_classes: int = args["num_classes"]
        dropout: float = args["dropout"]        
        freeze_backbone_params: bool = arg_is_true(args["freeze_backbone_params"])
        backbone_name: str = args["backbone"]
        fc1_out: int = args["fc1_out"]

        assert num_channels == 3, f"Must have `num_channels == 3` for model {self.__name__}."

        self.args = args 

        resnet = BACKBONES[backbone_name](pretrained=True)
        resnet_fc_in_features: int  = resnet.fc.in_features
        resnet.fc = nn.Sequential()
        self.resnet = resnet

        if freeze_backbone_params:
            self.resnet.requires_grad_(False)

        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(resnet_fc_in_features, fc1_out)
        self.fc2 = nn.Linear(fc1_out, num_classes)


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
            "--fc1-out",
            default=self.DEFAULT_FC1_OUT,
            type=int
        )
        args = parse_args(parser=parser)
        return args

       
    def forward(self, x):
        with torch.no_grad():
            x: torch.Tensor = self.resnet(x)

        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)

        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
           
        return x             


class ResNetOneDConv(nn.Module):
    __name__ = "ResNetOneDConv"

    IS_OBJECT_DETECTOR = False    

    DEFAULT_NUM_CHANNLES: int = 3
    DEFAULT_NUM_CLASSES: int = 2
    DEFAULT_DROPOUT: float = 0.0
    DEFAULT_FREEZE_BACKBONE_PARAMS: bool = True
    DEFAULT_BACKBONE_NAME: str = "resnet152"
    DEFAULT_KERNEL_SIZE: int = 3
    DEFAULT_STRIDE: int = 1
    DEFAULT_CONV_OUT_CHANNELS: int = 256
    DEFAULT_FC1_OUT: int = 128


    def __init__(self):
        super().__init__()
        args = self.parse_args()
        num_channels: int = args["num_channels"]
        num_classes: int = args["num_classes"]
        dropout: float = args["dropout"]        
        freeze_backbone_params: bool = arg_is_true(args["freeze_backbone_params"])
        backbone_name: str = args["backbone"]
        conv_out_channels: int = args["conv_out_channels"]
        kernel_size: int = args["kernel_size"]
        stride: int = args["stride"]
        sequence_length: int = args["sequence_length"]
        fc1_out: int = args["fc1_out"]

        assert num_channels == 3, f"Must have `num_channels == 3` for model {self.__name__}."

        self.args = args 

        resnet = BACKBONES[backbone_name](pretrained=True)
        resnet_fc_in_features: int  = resnet.fc.in_features
        resnet.fc = nn.Sequential()
        self.resnet = resnet

        if freeze_backbone_params:
            self.resnet.requires_grad_(False)

        self.conv1d = nn.Conv1d(
            in_channels=resnet_fc_in_features, out_channels=conv_out_channels,
            kernel_size=kernel_size, stride=stride
        )

        max_pool_kernel_size: int = math.floor(1 + (sequence_length - kernel_size) / stride)

        self.maxpool = nn.MaxPool1d(kernel_size=max_pool_kernel_size)
     
        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(conv_out_channels, fc1_out)
        self.fc2 = nn.Linear(fc1_out, num_classes)


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
            "--conv-out-channels",
            default=self.DEFAULT_CONV_OUT_CHANNELS,
            type=int
        )
        parser.add_argument(
            "--kernel-size",
            default=self.DEFAULT_KERNEL_SIZE,
            type=int
        )
        parser.add_argument(
            "--fc1-out",
            default=self.DEFAULT_FC1_OUT,
            type=int
        )
        parser.add_argument(
            "--stride",
            default=self.DEFAULT_STRIDE,
            type=int
        )
        parser.add_argument(
            "--sequence-length",
            type=int,
            required=True
        )
        args = parse_args(parser=parser)
        return args

       
    def forward(self, x_3d):
        features = list()
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x: torch.Tensor = self.resnet(x_3d[:, t, :, :, :])
                x = x.unsqueeze(-1)
                features.append(x)

        x = torch.concat(features, dim=-1)

        x = self.conv1d(x)
        x = self.maxpool(x)
        x = x.squeeze(-1)

        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)

        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
           
        return x


class FasterRCNN(nn.Module):
    __name__ = "FasterRCNN"

    IS_OBJECT_DETECTOR = True
    MODEL_NAME = "fasterrcnn_resnet50_fpn"

    DEFAULT_NUM_CHANNLES = 3
    DEFAULT_NUM_CLASSES = 2
    DEFAULT_TRAINABLE_LAYERS = 3
    DEFAULT_BACKBONE_NAME: str = "resnet152"


    def __init__(self, **kwargs):
        super().__init__()
        args = self.parse_args()
        num_channels = int(args["num_channels"])
        assert num_channels == 3, f"Must have `num_channels == 3` for model {self.__name__}."        
        
        num_classes = int(args["num_classes"])
        trainable_layers = int(args["trainable_layers"])
        backbone_name: str = args["backbone"]        
        backbone_weights = BACKBONE_WEIGHTS[backbone_name]
        self.args = {**args, **kwargs}

        model =  OBJECT_DETECTORS[self.MODEL_NAME](
            num_classes=num_classes, trainable_backbone_layers=trainable_layers, 
            backbone_weights=backbone_weights, **kwargs
        )
        self.model = model


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
            "--trainable-layers",
            default=self.DEFAULT_TRAINABLE_LAYERS,
            type=int
        )
        parser.add_argument(
            "--backbone",
            default=self.DEFAULT_BACKBONE_NAME
        )        
        args = parse_args(parser=parser)
        return args

    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
