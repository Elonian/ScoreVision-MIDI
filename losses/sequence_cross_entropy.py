from __future__ import annotations

import torch
import torch.nn as nn


class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self, padding_idx: int) -> None:
        super().__init__()
        self.padding_idx = int(padding_idx)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(f"Expected logits with shape [batch, steps, vocab], got {tuple(logits.shape)}")
        if targets.ndim != 2:
            raise ValueError(f"Expected targets with shape [batch, steps], got {tuple(targets.shape)}")
        return self.loss(logits.transpose(1, 2).contiguous(), targets.long())
