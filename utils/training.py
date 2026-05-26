from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from losses.ctc_loss import CTCSequenceLoss
from utils.decoding import greedy_decode_ctc
from utils.metrics import get_metrics
from utils.transcription import tokens_to_kern


def run_training(
    config: dict[str, Any],
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    blank_idx: int,
    i2w: dict[int, str],
    output_dir: str | Path,
    logger: logging.Logger,
) -> tuple[Path, tuple[float, float, float]]:
    output_dir = Path(output_dir)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weights_dir / f"{config['model']['name']}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device=%s", device)
    model.to(device)

    training_cfg = config["training"]
    early_cfg = training_cfg["early_stopping"]
    optimizer = optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))
    loss_fn = CTCSequenceLoss(
        blank_idx=blank_idx,
        zero_infinity=bool(config["loss"].get("zero_infinity", False)),
    ).to(device)

    best_metric = float("inf") if early_cfg["mode"] == "min" else -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    max_epochs = int(training_cfg["max_epochs"])

    for epoch in range(1, max_epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            logger=logger,
            log_every_n_steps=int(training_cfg.get("log_every_n_steps", 50)),
        )
        val_cer, val_ser, val_ler = _evaluate(model, val_loader, i2w, blank_idx, device)

        logger.info(
            "epoch=%s train_loss=%.6f val_CER=%.4f val_SER=%.4f val_LER=%.4f",
            epoch,
            train_loss,
            val_cer,
            val_ser,
            val_ler,
        )

        monitored = {"val_CER": val_cer, "val_SER": val_ser, "val_LER": val_ler}[early_cfg["monitor"]]
        if _is_improved(monitored, best_metric, float(early_cfg["min_delta"]), early_cfg["mode"]):
            best_metric = monitored
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "config": config,
                    "blank_idx": blank_idx,
                    "i2w": i2w,
                },
                checkpoint_path,
            )
            logger.info("Saved best checkpoint to %s", checkpoint_path)
        else:
            epochs_without_improvement += 1
            logger.info(
                "No %s improvement for %s/%s epochs",
                early_cfg["monitor"],
                epochs_without_improvement,
                int(early_cfg["patience"]),
            )

        if epochs_without_improvement >= int(early_cfg["patience"]):
            logger.info(
                "Early stopping at epoch=%s; best_epoch=%s best_metric=%.6f",
                epoch,
                best_epoch,
                best_metric,
            )
            break

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = _evaluate(
        model,
        test_loader,
        i2w,
        blank_idx,
        device,
        output_dir=output_dir,
        write_predictions=bool(config["output"].get("write_predictions", True)),
    )
    logger.info(
        "test_CER=%.4f test_SER=%.4f test_LER=%.4f",
        test_metrics[0],
        test_metrics[1],
        test_metrics[2],
    )
    return checkpoint_path, test_metrics


def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: CTCSequenceLoss,
    device: torch.device,
    logger: logging.Logger,
    log_every_n_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    for step, batch in enumerate(train_loader, start=1):
        images, targets, input_lengths, target_lengths = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = loss_fn(predictions, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        steps += 1
        if step % max(log_every_n_steps, 1) == 0:
            logger.info("train_step=%s loss=%.6f", step, total_loss / steps)

    return total_loss / max(steps, 1)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    i2w: dict[int, str],
    blank_idx: int,
    device: torch.device,
    output_dir: Path | None = None,
    write_predictions: bool = False,
) -> tuple[float, float, float]:
    model.eval()
    hypotheses: list[str] = []
    references: list[str] = []
    hyp_dir = output_dir / "hyp" if output_dir else None
    gt_dir = output_dir / "gt" if output_dir else None

    if write_predictions and hyp_dir is not None and gt_dir is not None:
        hyp_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(loader):
        images, targets, _, target_lengths = _move_batch(batch, device)
        predictions = model(images)
        decoded, target_tokens = greedy_decode_ctc(predictions, targets, target_lengths, i2w, blank_idx)

        for sample_idx, (decoded_tokens, gt_tokens) in enumerate(zip(decoded, target_tokens)):
            hypothesis = tokens_to_kern(decoded_tokens)
            reference = tokens_to_kern(gt_tokens)
            hypotheses.append(hypothesis)
            references.append(reference)

            if write_predictions and hyp_dir is not None and gt_dir is not None:
                filename = f"{batch_idx}.krn" if len(decoded) == 1 else f"{batch_idx}_{sample_idx}.krn"
                (hyp_dir / filename).write_text(hypothesis, encoding="utf-8")
                (gt_dir / filename).write_text(reference, encoding="utf-8")

    return get_metrics(hypotheses, references)


def _move_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images, targets, input_lengths, target_lengths = batch
    return (
        images.to(device),
        targets.to(device),
        input_lengths.to(device),
        target_lengths.to(device),
    )


def _is_improved(current: float, best: float, min_delta: float, mode: str) -> bool:
    if mode == "min":
        return current < best - min_delta
    if mode == "max":
        return current > best + min_delta
    raise ValueError(f"Unsupported early stopping mode: {mode}")
