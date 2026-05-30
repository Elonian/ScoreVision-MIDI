from __future__ import annotations

import torch
import torch.nn as nn

from models.score_unfolding import compute_encoder_output_hw
from modules.autoregressive_decoder import AutoregressiveTransformerDecoder, PositionalEncoding2D
from modules.encoder import ScoreEncoder


class ScoreAutoregressive(nn.Module):
    def __init__(
        self,
        in_channels: int,
        vocab_size: int,
        padding_idx: int,
        max_height: int,
        max_width: int,
        encoder_dropout: float = 0.2,
        decoder_dropout: float = 0.1,
        d_model: int = 256,
        num_decoder_layers: int = 6,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.encoder = ScoreEncoder(in_channels=in_channels, dropout=encoder_dropout)
        self.feature_projection = nn.Conv2d(ScoreEncoder.out_channels, d_model, kernel_size=1)
        encoded_height, encoded_width = compute_encoder_output_hw(max_height, max_width)
        self.position_encoding = PositionalEncoding2D(d_model, encoded_height, encoded_width)
        self.decoder = AutoregressiveTransformerDecoder(
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=decoder_dropout,
            max_seq_len=max_seq_len,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        features = self.feature_projection(features)
        features = self.position_encoding(features)
        return features.flatten(2).transpose(1, 2).contiguous()

    def forward(self, images: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        memory = self.encode(images)
        return self.decoder(decoder_input, memory)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_length: int,
    ) -> torch.Tensor:
        self.eval()
        memory = self.encode(images)
        generated = torch.full(
            (images.size(0), 1),
            fill_value=int(bos_idx),
            dtype=torch.long,
            device=images.device,
        )
        for _ in range(max(int(max_length) - 1, 0)):
            logits = self.decoder(generated, memory)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token.squeeze(1).eq(int(eos_idx))):
                break
        return generated


def build_score_autoregressive(
    in_channels: int,
    vocab_size: int,
    padding_idx: int,
    max_height: int,
    max_width: int,
    encoder_dropout: float = 0.2,
    decoder_dropout: float = 0.1,
    d_model: int = 256,
    num_decoder_layers: int = 6,
    num_heads: int = 4,
    dim_feedforward: int = 1024,
    max_seq_len: int = 4096,
) -> ScoreAutoregressive:
    return ScoreAutoregressive(
        in_channels=in_channels,
        vocab_size=vocab_size,
        padding_idx=padding_idx,
        max_height=max_height,
        max_width=max_width,
        encoder_dropout=encoder_dropout,
        decoder_dropout=decoder_dropout,
        d_model=d_model,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        max_seq_len=max_seq_len,
    )
