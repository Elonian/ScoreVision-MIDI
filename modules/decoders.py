from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.components import PositionalEncoding1D
from utils.constants import ENCODER_OUTPUT_CHANNELS


class PageDecoder(nn.Module):
    def __init__(self, out_categories: int) -> None:
        super().__init__()
        self.dec_conv = nn.Conv2d(
            in_channels=ENCODER_OUTPUT_CHANNELS,
            out_channels=out_categories,
            kernel_size=(5, 5),
            padding=(2, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.dec_conv(inputs)
        x = F.log_softmax(x, dim=1)
        batch, channels, height, width = x.size()
        x = x.reshape(batch, channels, height * width)
        return x.permute(2, 0, 1)


class RecurrentScoreUnfolding(nn.Module):
    def __init__(self, out_categories: int) -> None:
        super().__init__()
        self.dec_lstm = nn.LSTM(
            input_size=ENCODER_OUTPUT_CHANNELS,
            hidden_size=256,
            bidirectional=True,
            batch_first=True,
        )
        self.out_dense = nn.Linear(in_features=ENCODER_OUTPUT_CHANNELS, out_features=out_categories)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.size()
        x = inputs.reshape(batch, channels, height * width)
        x = x.permute(0, 2, 1)
        x, _ = self.dec_lstm(x)
        x = self.out_dense(x)
        x = x.permute(1, 0, 2)
        return F.log_softmax(x, dim=2)


class TransformerScoreUnfolding(nn.Module):
    def __init__(self, out_categories: int, max_len: int) -> None:
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.pos_encoding = PositionalEncoding1D(
            dim=ENCODER_OUTPUT_CHANNELS,
            len_max=max_len,
            device=self.dummy_param.device,
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=ENCODER_OUTPUT_CHANNELS,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True,
        )
        self.dec_transf = nn.TransformerEncoder(transformer_layer, num_layers=1)
        self.out_dense = nn.Linear(in_features=ENCODER_OUTPUT_CHANNELS, out_features=out_categories)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.size()
        x = inputs.reshape(batch, channels, height * width)
        x = self.pos_encoding(x)
        x = x.permute(0, 2, 1)
        x = self.dec_transf(x)
        x = self.out_dense(x)
        x = x.permute(1, 0, 2)
        return F.log_softmax(x, dim=2)


class StaveRNNDecoder(nn.Module):
    def __init__(
        self,
        img_height: int,
        height_reduction: int,
        out_channels: int,
        out_categories: int,
    ) -> None:
        super().__init__()
        features = (img_height // height_reduction) * out_channels
        self.reshape_features = features
        self.dec_lstm = nn.LSTM(
            input_size=features,
            hidden_size=256,
            bidirectional=True,
            batch_first=True,
        )
        self.out_dense = nn.Linear(in_features=ENCODER_OUTPUT_CHANNELS, out_features=out_categories)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.size(0)
        x = inputs.permute(0, 3, 2, 1)
        x = x.reshape(batch, -1, self.reshape_features)
        x, _ = self.dec_lstm(x)
        x = self.out_dense(x)
        x = x.permute(1, 0, 2)
        return F.log_softmax(x, dim=-1)


class StaveTransformerDecoder(nn.Module):
    def __init__(
        self,
        img_height: int,
        height_reduction: int,
        out_channels: int,
        out_categories: int,
        max_len: int,
    ) -> None:
        super().__init__()
        features = (img_height // height_reduction) * out_channels
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.reshape_features = features
        self.projection_layer = nn.Linear(features, ENCODER_OUTPUT_CHANNELS)
        self.pos_encoding = PositionalEncoding1D(
            dim=ENCODER_OUTPUT_CHANNELS,
            len_max=max_len,
            device=self.dummy_param.device,
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=ENCODER_OUTPUT_CHANNELS,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True,
        )
        self.dec_transf = nn.TransformerEncoder(transformer_layer, num_layers=1)
        self.out_dense = nn.Linear(in_features=ENCODER_OUTPUT_CHANNELS, out_features=out_categories)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.size(0)
        x = inputs.permute(0, 3, 2, 1).contiguous()
        x = x.reshape(batch, -1, self.reshape_features)
        x = self.projection_layer(x)
        x = self.pos_encoding(x.permute(0, 2, 1).contiguous())
        x = x.permute(0, 2, 1).contiguous()
        x = self.dec_transf(x)
        x = self.out_dense(x)
        x = x.permute(1, 0, 2).contiguous()
        return F.log_softmax(x, dim=-1)
