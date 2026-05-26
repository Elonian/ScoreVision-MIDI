from __future__ import annotations

import torch
import torch.nn as nn


class CTCSequenceLoss(nn.Module):
    def __init__(self, blank_idx: int, zero_infinity: bool = False) -> None:
        super().__init__()
        self.blank_idx = int(blank_idx)
        self.loss = nn.CTCLoss(blank=self.blank_idx, zero_infinity=zero_infinity)

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss(
            log_probs,
            targets.long(),
            input_lengths.long(),
            target_lengths.long(),
        )
