from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
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
    distributed = config.get("distributed", {})
    is_distributed = bool(distributed.get("enabled", False))
    is_main_process = bool(distributed.get("is_main_process", True))
    if is_main_process:
        weights_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path: Path | None = None

    if is_distributed:
        device = torch.device(f"cuda:{int(distributed['local_rank'])}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        (
            "Using device=%s distributed=%s rank=%s world_size=%s "
            "batch_size_per_process=%s effective_global_batch_size=%s"
        ),
        device,
        is_distributed,
        distributed.get("rank", 0),
        distributed.get("world_size", 1),
        config["training"]["batch_size"],
        int(config["training"]["batch_size"]) * int(distributed.get("world_size", 1)),
    )
    model.to(device)
    if is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[int(distributed["local_rank"])],
            output_device=int(distributed["local_rank"]),
            broadcast_buffers=False,
        )

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
    training_started_at = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_started_at = time.perf_counter()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_loss = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            logger=logger,
            log_every_n_steps=int(training_cfg.get("log_every_n_steps", 50)),
            should_log=is_main_process,
        )
        train_loss = _reduce_mean(train_loss, device) if is_distributed else train_loss
        _barrier(distributed)
        if is_main_process:
            eval_model = _unwrap_model(model)
            val_cer, val_ser, val_ler = _evaluate(eval_model, val_loader, i2w, blank_idx, device)
        else:
            val_cer, val_ser, val_ler = 0.0, 0.0, 0.0
        _synchronize_device(device)

        should_stop = False
        if is_main_process:
            epoch_seconds = time.perf_counter() - epoch_started_at
            elapsed_seconds = time.perf_counter() - training_started_at
            average_epoch_seconds = elapsed_seconds / epoch
            eta_seconds = average_epoch_seconds * max(max_epochs - epoch, 0)

            logger.info(
                (
                    "epoch=%s train_loss=%.6f val_CER=%.4f val_SER=%.4f val_LER=%.4f "
                    "epoch_time=%s elapsed=%s eta_if_no_early_stop=%s"
                ),
                epoch,
                train_loss,
                val_cer,
                val_ser,
                val_ler,
                _format_duration(epoch_seconds),
                _format_duration(elapsed_seconds),
                _format_duration(eta_seconds),
            )

            monitored = {"val_CER": val_cer, "val_SER": val_ser, "val_LER": val_ler}[early_cfg["monitor"]]
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": _unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "blank_idx": blank_idx,
                "i2w": i2w,
                "train_loss": train_loss,
                "val_CER": val_cer,
                "val_SER": val_ser,
                "val_LER": val_ler,
                "monitor": early_cfg["monitor"],
                "monitored_metric": monitored,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
            }
            if bool(config.get("output", {}).get("save_epoch_checkpoints", True)):
                epoch_checkpoint_path = weights_dir / f"epoch_{epoch:04d}.pt"
                torch.save(checkpoint_payload, epoch_checkpoint_path)
                logger.info("Saved epoch checkpoint to %s", epoch_checkpoint_path)

            if _is_improved(monitored, best_metric, float(early_cfg["min_delta"]), early_cfg["mode"]):
                best_metric = monitored
                best_epoch = epoch
                epochs_without_improvement = 0
                checkpoint_payload["best_epoch"] = best_epoch
                checkpoint_payload["best_metric"] = best_metric
                best_checkpoint_path = weights_dir / "best.pt"
                torch.save(checkpoint_payload, best_checkpoint_path)
                logger.info("Saved best checkpoint to %s", best_checkpoint_path)
            else:
                epochs_without_improvement += 1
                logger.info(
                    "No %s improvement for %s/%s epochs",
                    early_cfg["monitor"],
                    epochs_without_improvement,
                    int(early_cfg["patience"]),
                )

            if epochs_without_improvement >= int(early_cfg["patience"]):
                should_stop = True
                logger.info(
                    "Early stopping at epoch=%s; best_epoch=%s best_metric=%.6f",
                    epoch,
                    best_epoch,
                    best_metric,
                )

        should_stop = _broadcast_bool(should_stop, device, distributed)
        if should_stop:
            break

    _barrier(distributed)
    if is_main_process and best_checkpoint_path is not None and best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])

    if is_main_process:
        test_metrics = _evaluate(
            _unwrap_model(model),
            test_loader,
            i2w,
            blank_idx,
            device,
            output_dir=output_dir,
            write_predictions=bool(config["output"].get("write_predictions", True)),
        )
        _synchronize_device(device)
        total_seconds = time.perf_counter() - training_started_at
        logger.info(
            "test_CER=%.4f test_SER=%.4f test_LER=%.4f total_time=%s",
            test_metrics[0],
            test_metrics[1],
            test_metrics[2],
            _format_duration(total_seconds),
        )
    else:
        test_metrics = (0.0, 0.0, 0.0)
    _barrier(distributed)
    return best_checkpoint_path or (weights_dir / "best.pt"), test_metrics


def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: CTCSequenceLoss,
    device: torch.device,
    logger: logging.Logger,
    log_every_n_steps: int,
    should_log: bool = True,
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
        if should_log and step % max(log_every_n_steps, 1) == 0:
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
        images, targets, input_lengths, target_lengths = _move_batch(batch, device)
        predictions = model(images)
        decoded, target_tokens = greedy_decode_ctc(
            predictions,
            targets,
            input_lengths,
            target_lengths,
            i2w,
            blank_idx,
        )

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
        input_lengths.cpu(),
        target_lengths.cpu(),
    )


def _is_improved(current: float, best: float, min_delta: float, mode: str) -> bool:
    if mode == "min":
        return current < best - min_delta
    if mode == "max":
        return current > best + min_delta
    raise ValueError(f"Unsupported early stopping mode: {mode}")


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _barrier(distributed: dict[str, Any]) -> None:
    if distributed.get("enabled") and dist.is_initialized():
        if distributed.get("backend") == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[int(distributed["local_rank"])])
        else:
            dist.barrier()


def _reduce_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def _broadcast_bool(value: bool, device: torch.device, distributed: dict[str, Any]) -> bool:
    if not distributed.get("enabled"):
        return bool(value)
    tensor = torch.tensor(1 if value else 0, device=device, dtype=torch.int64)
    dist.broadcast(tensor, src=0)
    return bool(tensor.item())


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"
