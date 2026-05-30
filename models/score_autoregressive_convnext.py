from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import ConvNextConfig, ConvNextModel

from modules.autoregressive_decoder import AutoregressiveTransformerDecoder, PositionalEncoding2D


class ConvNextScoreAutoregressive(nn.Module):
    def __init__(
        self,
        in_channels: int,
        vocab_size: int,
        padding_idx: int,
        max_height: int,
        max_width: int,
        encoder_source: str = "scratch",
        pretrained_model_name: str = "facebook/convnext-tiny-224",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        freeze_encoder: bool = False,
        scratch_hidden_sizes: list[int] | tuple[int, ...] = (64, 128, 256),
        scratch_depths: list[int] | tuple[int, ...] = (3, 3, 9),
        d_model: int = 256,
        num_decoder_layers: int = 6,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        decoder_dropout: float = 0.1,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.encoder_source = encoder_source.lower()
        if self.encoder_source not in {"scratch", "pretrained"}:
            raise ValueError("encoder_source must be 'scratch' or 'pretrained'")

        self.use_imagenet_normalization = self.encoder_source == "pretrained"
        if self.encoder_source == "pretrained":
            self.encoder = ConvNextModel.from_pretrained(
                pretrained_model_name,
                cache_dir=str(cache_dir) if cache_dir else None,
                local_files_only=bool(local_files_only),
            )
        else:
            encoder_config = ConvNextConfig(
                num_channels=int(in_channels),
                num_stages=len(scratch_hidden_sizes),
                hidden_sizes=list(scratch_hidden_sizes),
                depths=list(scratch_depths),
            )
            self.encoder = ConvNextModel(encoder_config)

        # The pooled-output layernorm is not used because transcription consumes the
        # spatial last_hidden_state. Keeping it trainable breaks DDP unused-parameter checks.
        if hasattr(self.encoder, "layernorm"):
            for parameter in self.encoder.layernorm.parameters():
                parameter.requires_grad = False

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        encoder_channels = int(self.encoder.config.hidden_sizes[-1])
        self.feature_projection = nn.Conv2d(encoder_channels, d_model, kernel_size=1)

        reduction = _convnext_reduction(self.encoder.config)
        encoded_height = _convnext_output_size(int(max_height), self.encoder.config)
        encoded_width = _convnext_output_size(int(max_width), self.encoder.config)
        if encoded_height <= 0 or encoded_width <= 0:
            raise ValueError(
                f"Invalid ConvNeXt encoded size {(encoded_height, encoded_width)} from "
                f"max image size {(max_height, max_width)} and reduction={reduction}"
            )

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
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        encoder_input = self._prepare_encoder_input(images)
        features = self.encoder(pixel_values=encoder_input).last_hidden_state
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

    def _prepare_encoder_input(self, images: torch.Tensor) -> torch.Tensor:
        if self.encoder_source == "pretrained":
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
            if images.size(1) != 3:
                raise ValueError("Pretrained ConvNeXt expects 1-channel grayscale or 3-channel RGB input.")
            images = (images - self.imagenet_mean.to(images.device)) / self.imagenet_std.to(images.device)
        return images


def build_convnext_score_autoregressive(
    in_channels: int,
    vocab_size: int,
    padding_idx: int,
    max_height: int,
    max_width: int,
    encoder_source: str = "scratch",
    pretrained_model_name: str = "facebook/convnext-tiny-224",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    freeze_encoder: bool = False,
    scratch_hidden_sizes: list[int] | tuple[int, ...] = (64, 128, 256),
    scratch_depths: list[int] | tuple[int, ...] = (3, 3, 9),
    d_model: int = 256,
    num_decoder_layers: int = 6,
    num_heads: int = 4,
    dim_feedforward: int = 1024,
    decoder_dropout: float = 0.1,
    max_seq_len: int = 4096,
) -> ConvNextScoreAutoregressive:
    return ConvNextScoreAutoregressive(
        in_channels=in_channels,
        vocab_size=vocab_size,
        padding_idx=padding_idx,
        max_height=max_height,
        max_width=max_width,
        encoder_source=encoder_source,
        pretrained_model_name=pretrained_model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        freeze_encoder=freeze_encoder,
        scratch_hidden_sizes=scratch_hidden_sizes,
        scratch_depths=scratch_depths,
        d_model=d_model,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        decoder_dropout=decoder_dropout,
        max_seq_len=max_seq_len,
    )


def _convnext_reduction(config: ConvNextConfig) -> int:
    patch_size = int(getattr(config, "patch_size", 4))
    return patch_size * (2 ** (int(config.num_stages) - 1))


def _convnext_output_size(size: int, config: ConvNextConfig) -> int:
    output = int(size) // int(getattr(config, "patch_size", 4))
    for _ in range(int(config.num_stages) - 1):
        output //= 2
    return output
