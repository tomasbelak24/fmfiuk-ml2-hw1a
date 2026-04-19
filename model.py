import torch
import torch.nn as nn
import torch.nn.functional as F

class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)


class DilatedConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        out = self.dropout(out)
        return self.act(x + out)
    

class AvgMaxAttentionPoolHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        # x: [B, C, T]
        avg = x.mean(dim=-1)                 # [B, C]
        mx = x.amax(dim=-1)                  # [B, C]

        attn_logits = self.score(x)          # [B, 1, T]
        attn = F.softmax(attn_logits, dim=-1)
        attn_pool = (x * attn).sum(dim=-1)   # [B, C]

        return torch.cat([avg, mx, attn_pool], dim=1)   # [B, 3C]



def create_model():
    # No sigmoid at the end, evaluation handles this by itself

    dilations = [1, 2, 4, 8, 16, 32, 64]

    layers = [
        LambdaLayer(lambda x: (x*2).unsqueeze(1)),
        nn.Conv1d(1, 24, kernel_size=7, padding=3),
        nn.ReLU(),
        nn.Conv1d(24, 12, kernel_size=5, padding=2),
        nn.ReLU(),
    ]

    for d in dilations:
        layers.append(DilatedConvBlock(12, d, dropout=0.2))

    layers.extend([
    nn.Conv1d(12, 8, kernel_size=1),
    nn.ReLU(),
    AvgMaxAttentionPoolHead(8),  # [B, 16, T] -> [B, 48]
    nn.Linear(24, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
    ])

    return nn.Sequential(*layers)
