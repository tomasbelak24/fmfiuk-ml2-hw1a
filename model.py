import torch
import torch.nn as nn

class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)


class DilatedConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
    

class AvgMaxPoolHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        avg = self.avgpool(x).squeeze(-1)
        mx = self.maxpool(x).squeeze(-1)
        return torch.cat([avg, mx], dim=1)


def create_model():
    # No sigmoid at the end, evaluation handles this by itself
    dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]#, 32]#, 64, 128, 256]#, 512, 1024]

    layers = [

        LambdaLayer(lambda x: x.unsqueeze(1)),
        nn.Conv1d(1, 32, kernel_size=7, padding=3),
        nn.ReLU(),
    ]

    for d in dilations:
        layers.append(DilatedConvBlock(32, d))

    layers.extend(
        [
            nn.AdaptiveMaxPool1d(1),
            LambdaLayer(lambda x: x.view(x.size(0), -1)),
            nn.Linear(32, 1),
        ]
    )

    return nn.Sequential(*layers)