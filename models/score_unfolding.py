from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from modules.decoders import (
    PageDecoder,
    RecurrentScoreUnfolding,
    StaveRNNDecoder,
    StaveTransformerDecoder,
    TransformerScoreUnfolding,
)
from modules.encoder import ScoreEncoder


def compute_unfolded_length(height: int, width: int) -> int:
    encoded_height, encoded_width = compute_encoder_output_hw(height, width)
    return encoded_height * encoded_width


def compute_encoder_output_hw(height: int, width: int) -> tuple[int, int]:
    encoded_height = int(height)
    encoded_width = int(width)
    for stride_h, stride_w in ((1, 1), (2, 2), (2, 2), (2, 2), (2, 1)):
        encoded_height = _conv_stride_output(encoded_height, stride_h)
        encoded_width = _conv_stride_output(encoded_width, stride_w)
    return encoded_height, encoded_width


def _conv_stride_output(size: int, stride: int) -> int:
    return (int(size) + int(stride) - 1) // int(stride)


class ScoreUnfoldingFCN(nn.Module):
    def __init__(self, in_channels: int, out_categories: int, dropout: float = 0.4) -> None:
        super().__init__()
        self.encoder = ScoreEncoder(in_channels=in_channels, dropout=dropout)
        self.decoder = PageDecoder(out_categories=out_categories)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


class ScoreUnfoldingCRNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_categories: int,
        dropout: float = 0.4,
        pretrain_path: str | None = None,
    ) -> None:
        super().__init__()
        self.encoder = ScoreEncoder(in_channels=in_channels, dropout=dropout)
        _load_encoder_weights(self.encoder, pretrain_path)
        self.decoder = RecurrentScoreUnfolding(out_categories=out_categories)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


class ScoreUnfoldingCNNT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_categories: int,
        max_len: int,
        dropout: float = 0.4,
        pretrain_path: str | None = None,
    ) -> None:
        super().__init__()
        self.encoder = ScoreEncoder(in_channels=in_channels, dropout=dropout)
        _load_encoder_weights(self.encoder, pretrain_path)
        self.decoder = TransformerScoreUnfolding(
            out_categories=out_categories,
            max_len=max_len,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


class StaveUnfoldingCRNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_categories: int,
        img_height: int,
        dropout: float = 0.4,
        pretrain_path: str | None = None,
    ) -> None:
        super().__init__()
        self.encoder = ScoreEncoder(in_channels=in_channels, dropout=dropout)
        _load_encoder_weights(self.encoder, pretrain_path)
        self.decoder = StaveRNNDecoder(
            img_height=img_height,
            height_reduction=ScoreEncoder.height_reduction,
            out_channels=ScoreEncoder.out_channels,
            out_categories=out_categories,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


class StaveUnfoldingCNNT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_categories: int,
        img_height: int,
        max_len: int,
        dropout: float = 0.4,
        pretrain_path: str | None = None,
    ) -> None:
        super().__init__()
        self.encoder = ScoreEncoder(in_channels=in_channels, dropout=dropout)
        _load_encoder_weights(self.encoder, pretrain_path)
        self.decoder = StaveTransformerDecoder(
            img_height=img_height,
            height_reduction=ScoreEncoder.height_reduction,
            out_channels=ScoreEncoder.out_channels,
            out_categories=out_categories,
            max_len=max_len,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


def build_model(
    model_name: str,
    max_width: int,
    max_height: int,
    in_channels: int,
    out_size: int,
    dropout: float = 0.4,
    max_len: int | None = None,
    pretrain_path: str | None = None,
) -> nn.Module:
    model_key = model_name.upper()
    inferred_max_len = max_len or compute_unfolded_length(max_height, max_width)

    if model_key == "FCN":
        return ScoreUnfoldingFCN(
            in_channels=in_channels,
            out_categories=out_size,
            dropout=dropout,
        )
    if model_key == "CRNN":
        return ScoreUnfoldingCRNN(
            in_channels=in_channels,
            out_categories=out_size,
            dropout=dropout,
            pretrain_path=pretrain_path,
        )
    if model_key == "CNNT":
        return ScoreUnfoldingCNNT(
            in_channels=in_channels,
            out_categories=out_size,
            max_len=inferred_max_len,
            dropout=dropout,
            pretrain_path=pretrain_path,
        )
    if model_key == "STAVE_CRNN":
        return StaveUnfoldingCRNN(
            in_channels=in_channels,
            out_categories=out_size,
            img_height=max_height,
            dropout=dropout,
            pretrain_path=pretrain_path,
        )
    if model_key == "STAVE_CNNT":
        _, encoded_width = compute_encoder_output_hw(max_height, max_width)
        return StaveUnfoldingCNNT(
            in_channels=in_channels,
            out_categories=out_size,
            img_height=max_height,
            max_len=encoded_width,
            dropout=dropout,
            pretrain_path=pretrain_path,
        )

    supported = "FCN, CRNN, CNNT, STAVE_CRNN, STAVE_CNNT"
    raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")


def _load_encoder_weights(encoder: nn.Module, pretrain_path: str | None) -> None:
    if not pretrain_path:
        return

    weights_path = Path(pretrain_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Encoder pretrain weights not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu")
    encoder.load_state_dict(state_dict, strict=True)
