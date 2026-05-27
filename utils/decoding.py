from __future__ import annotations

from itertools import groupby

import torch


def greedy_decode_ctc(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    i2w: dict[int, str],
    blank_idx: int,
) -> tuple[list[list[str]], list[list[str]]]:
    decoded_batch: list[list[str]] = []
    target_batch: list[list[str]] = []
    batch_size = log_probs.size(1)

    for batch_idx in range(batch_size):
        input_length = min(int(input_lengths[batch_idx]), int(log_probs.size(0)))
        sample_probs = log_probs[:input_length, batch_idx, :]
        best_path = torch.argmax(sample_probs, dim=1)
        collapsed = [token for token, _ in groupby(best_path.detach().cpu().tolist())]
        decoded_ids = [int(token) for token in collapsed if int(token) != int(blank_idx)]
        decoded_batch.append([i2w[token] for token in decoded_ids if token in i2w])

        target_length = int(target_lengths[batch_idx])
        target_ids = targets[batch_idx, :target_length].detach().cpu().tolist()
        target_batch.append([i2w[int(token)] for token in target_ids if int(token) in i2w])

    return decoded_batch, target_batch
