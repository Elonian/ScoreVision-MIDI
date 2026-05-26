from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthSepConv2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        activation: nn.Module | None = None,
        padding: bool | tuple[int, int] = True,
        stride: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.extra_padding: list[int] | None = None

        if padding:
            if padding is True:
                padding = tuple(int((k - 1) / 2) for k in kernel_size)
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.extra_padding = [
                        padding_h // 2,
                        padding_h - padding_h // 2,
                        padding_w // 2,
                        padding_w - padding_w // 2,
                    ]
                    padding = (0, 0)
        else:
            padding = (0, 0)

        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            dilation=dilation,
            kernel_size=(1, 1),
        )
        self.activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(inputs)
        if self.extra_padding:
            x = F.pad(x, self.extra_padding)
        if self.activation:
            x = self.activation(x)
        return self.point_conv(x)


class MixDropout(nn.Module):
    def __init__(self, dropout_prob: float = 0.4, dropout_2d_prob: float = 0.2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.dropout2d = nn.Dropout2d(dropout_2d_prob)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            return self.dropout(inputs)
        return self.dropout2d(inputs)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int] = (1, 1),
        kernel_size: int = 3,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.activation = activation()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=stride,
        )
        self.norm_layer = nn.InstanceNorm2d(
            num_features=out_channels,
            eps=0.001,
            momentum=0.99,
            track_running_stats=False,
        )
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dropout_position = random.randint(1, 3)

        x = self.activation(self.conv1(inputs))
        if dropout_position == 1:
            x = self.dropout(x)

        x = self.activation(self.conv2(x))
        if dropout_position == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.activation(self.conv3(x))
        if dropout_position == 3:
            x = self.dropout(x)

        return x


class DSCBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int] = (2, 1),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_channels, out_channels, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_channels, out_channels, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=stride,
        )
        self.norm_layer = nn.InstanceNorm2d(
            out_channels,
            eps=0.001,
            momentum=0.99,
            track_running_stats=False,
        )
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dropout_position = random.randint(1, 3)

        x = self.activation(self.conv1(inputs))
        if dropout_position == 1:
            x = self.dropout(x)

        x = self.activation(self.conv2(x))
        if dropout_position == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        if dropout_position == 3:
            x = self.dropout(x)

        return x


class PositionalEncoding1D(nn.Module):
    def __init__(self, dim: int, len_max: int, device: torch.device | None = None) -> None:
        super().__init__()
        if len_max is None or len_max <= 0:
            raise ValueError("len_max must be a positive integer for transformer decoding")

        device = device or torch.device("cpu")
        self.len_max = int(len_max)
        self.dim = int(dim)

        pe = torch.zeros((1, dim, self.len_max), device=device, requires_grad=False)
        div = torch.exp(
            -torch.arange(0.0, dim, 2, device=device)
            / dim
            * torch.log(torch.tensor(10000.0, device=device))
        ).unsqueeze(1)
        positions = torch.arange(0.0, self.len_max, device=device)
        pe[:, ::2, :] = torch.sin(positions * div).unsqueeze(0)
        pe[:, 1::2, :] = torch.cos(positions * div).unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, start: int | torch.Tensor = 0) -> torch.Tensor:
        if isinstance(start, int):
            return x + self.pe[:, :, start : start + x.size(2)].to(x.device)

        output = x.clone()
        for i in range(x.size(0)):
            offset = int(start[i])
            output[i] = output[i] + self.pe[0, :, offset : offset + x.size(2)].to(x.device)
        return output
