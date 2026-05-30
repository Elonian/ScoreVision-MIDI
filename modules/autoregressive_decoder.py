from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding1D(nn.Module):
    def __init__(self, dim: int, max_len: int) -> None:
        super().__init__()
        if max_len <= 0:
            raise ValueError("max_len must be positive")

        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.pe[:, : inputs.size(1)].to(inputs.device)


class PositionalEncoding2D(nn.Module):
    def __init__(self, dim: int, max_height: int, max_width: int) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("2D positional encoding requires dim divisible by 4")
        if max_height <= 0 or max_width <= 0:
            raise ValueError("max_height and max_width must be positive")

        pe = torch.zeros(dim, max_height, max_width, dtype=torch.float32)
        half = dim // 2
        y_positions = torch.arange(max_height, dtype=torch.float32).unsqueeze(1)
        x_positions = torch.arange(max_width, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, half, 2, dtype=torch.float32) * (-math.log(10000.0) / half))

        pe[0:half:2, :, :] = torch.sin(y_positions * div_term).T.unsqueeze(2).repeat(1, 1, max_width)
        pe[1:half:2, :, :] = torch.cos(y_positions * div_term).T.unsqueeze(2).repeat(1, 1, max_width)
        pe[half::2, :, :] = torch.sin(x_positions * div_term).T.unsqueeze(1).repeat(1, max_height, 1)
        pe[half + 1 :: 2, :, :] = torch.cos(x_positions * div_term).T.unsqueeze(1).repeat(1, max_height, 1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, _, height, width = inputs.shape
        if height > self.pe.size(1) or width > self.pe.size(2):
            raise ValueError(
                f"Encoded image shape {(height, width)} exceeds configured positional encoding "
                f"{(self.pe.size(1), self.pe.size(2))}"
            )
        return inputs + self.pe[:, :height, :width].unsqueeze(0).to(inputs.device)


class AutoregressiveTransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.padding_idx = int(padding_idx)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.padding_idx)
        self.position_encoding = PositionalEncoding1D(d_model, max_seq_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, decoder_input: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        target_padding_mask = decoder_input.eq(self.padding_idx)
        target = self.embedding(decoder_input)
        target = self.position_encoding(target)
        causal_mask = torch.triu(
            torch.ones(decoder_input.size(1), decoder_input.size(1), device=decoder_input.device, dtype=torch.bool),
            diagonal=1,
        )
        decoded = self.decoder(
            tgt=target,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
        )
        return self.output_projection(decoded)
