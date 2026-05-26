from __future__ import annotations

import torch
import torch.nn as nn

from modules.components import ConvBlock, DSCBlock
from utils.constants import (
    ENCODER_HEIGHT_REDUCTION,
    ENCODER_OUTPUT_CHANNELS,
    ENCODER_WIDTH_REDUCTION,
)


class ScoreEncoder(nn.Module):
    height_reduction = ENCODER_HEIGHT_REDUCTION
    width_reduction = ENCODER_WIDTH_REDUCTION
    out_channels = ENCODER_OUTPUT_CHANNELS

    def __init__(self, in_channels: int, dropout: float = 0.4) -> None:
        super().__init__()
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_channels, 32, stride=(1, 1), dropout=dropout),
                ConvBlock(32, 64, stride=(2, 2), dropout=dropout),
                ConvBlock(64, 128, stride=(2, 2), dropout=dropout),
                ConvBlock(128, 256, stride=(2, 2), dropout=dropout),
                ConvBlock(256, 512, stride=(2, 1), dropout=dropout),
            ]
        )
        self.dsc_blocks = nn.ModuleList(
            [
                DSCBlock(512, 512, stride=(1, 1), dropout=dropout),
                DSCBlock(512, 512, stride=(1, 1), dropout=dropout),
                DSCBlock(512, 512, stride=(1, 1), dropout=dropout),
                DSCBlock(512, 512, stride=(1, 1), dropout=dropout),
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.conv_blocks:
            x = layer(x)

        for layer in self.dsc_blocks:
            residual = layer(x)
            x = x + residual if x.size() == residual.size() else residual

        return x
