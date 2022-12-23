import torch
import torch.nn as nn
import torch.nn.init as init

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


    def __init__(
        self, num_channels: int = 4, num_classes: int = 10, 
        dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
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
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
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


    def __init__(
        self, num_channels: int = 4, num_classes: int = 10, 
        dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
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
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1)
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
