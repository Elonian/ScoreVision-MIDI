from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from losses.sequence_cross_entropy import SequenceCrossEntropyLoss
from utils.metrics import get_metrics
from utils.autoregressive_data import EOS_TOKEN, BOS_TOKEN, PAD_TOKEN, decoder_tokens_to_kern


def run_autoregressive_training(
    config: dict[str, Any],
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    w2i: dict[str, int],
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

    if is_distributed:
        device = torch.device("cuda", int(distributed["local_rank"]))
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
    logger.info("Model moved to %s", device)
    if is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[int(distributed["local_rank"])],
            output_device=int(distributed["local_rank"]),
            broadcast_buffers=False,
        )
        logger.info("DistributedDataParallel wrapper initialized")

    training_cfg = config["training"]
    early_cfg = training_cfg["early_stopping"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))
    loss_fn = SequenceCrossEntropyLoss(padding_idx=w2i[PAD_TOKEN]).to(device)
    logger.info("Optimizer and loss initialized")
    teacher_noise = float(training_cfg.get("teacher_forcing_error_rate", 0.0))
    validation_every = max(int(training_cfg.get("validation_every_n_epochs", 1)), 1)

    best_metric = float("inf") if early_cfg["mode"] == "min" else -float("inf")
    best_checkpoint_path = weights_dir / "best.pt"
    best_epoch = 0
    epochs_without_improvement = 0
    started_at = time.perf_counter()

    max_epochs = int(training_cfg["max_epochs"])
    for epoch in range(1, max_epochs + 1):
        epoch_started_at = time.perf_counter()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_loss = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            padding_idx=w2i[PAD_TOKEN],
            vocab_size=len(w2i),
            teacher_noise=teacher_noise,
            log_every_n_steps=int(training_cfg.get("log_every_n_steps", 50)),
            logger=logger,
            should_log=is_main_process,
        )
        train_loss = _reduce_mean(train_loss, device) if is_distributed else train_loss
        should_validate = epoch % validation_every == 0 or epoch == max_epochs
        epoch_seconds = time.perf_counter() - epoch_started_at
        elapsed_seconds = time.perf_counter() - started_at
        if not should_validate:
            if is_main_process:
                logger.info(
                    "epoch=%s train_loss=%.6f epoch_time=%s elapsed=%s",
                    epoch,
                    train_loss,
                    _format_duration(epoch_seconds),
                    _format_duration(elapsed_seconds),
                )
            continue

        _barrier(distributed)
        if is_main_process:
            val_cer, val_ser, val_ler = evaluate_autoregressive(
                model=_unwrap_model(model),
                loader=val_loader,
                i2w=i2w,
                bos_idx=w2i[BOS_TOKEN],
                eos_idx=w2i[EOS_TOKEN],
                max_length=int(config["decoding"]["max_length"]),
                device=device,
            )
            logger.info(
                "epoch=%s train_loss=%.6f val_CER=%.4f val_SER=%.4f val_LER=%.4f epoch_time=%s elapsed=%s",
                epoch,
                train_loss,
                val_cer,
                val_ser,
                val_ler,
                _format_duration(epoch_seconds),
                _format_duration(elapsed_seconds),
            )
        else:
            val_cer, val_ser, val_ler = 0.0, 0.0, 0.0

        should_stop = False
        if is_main_process:
            monitored = {"val_CER": val_cer, "val_SER": val_ser, "val_LER": val_ler}[early_cfg["monitor"]]
            payload = {
                "epoch": epoch,
                "model_state_dict": _unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "w2i": w2i,
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
                epoch_path = weights_dir / f"epoch_{epoch:04d}.pt"
                torch.save(payload, epoch_path)
                logger.info("Saved epoch checkpoint to %s", epoch_path)

            if _is_improved(monitored, best_metric, float(early_cfg["min_delta"]), early_cfg["mode"]):
                best_metric = monitored
                best_epoch = epoch
                epochs_without_improvement = 0
                payload["best_epoch"] = best_epoch
                payload["best_metric"] = best_metric
                torch.save(payload, best_checkpoint_path)
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
                logger.info("Early stopping at epoch=%s; best_epoch=%s best_metric=%.6f", epoch, best_epoch, best_metric)
        should_stop = _broadcast_bool(should_stop, device, distributed)
        if should_stop:
            break

    _barrier(distributed)
    if is_main_process and best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])

    if is_main_process:
        test_metrics = evaluate_autoregressive(
            model=_unwrap_model(model),
            loader=test_loader,
            i2w=i2w,
            bos_idx=w2i[BOS_TOKEN],
            eos_idx=w2i[EOS_TOKEN],
            max_length=int(config["decoding"]["max_length"]),
            device=device,
            output_dir=output_dir,
            write_predictions=bool(config["output"].get("write_predictions", True)),
        )
        logger.info("test_CER=%.4f test_SER=%.4f test_LER=%.4f", test_metrics[0], test_metrics[1], test_metrics[2])
    else:
        test_metrics = (0.0, 0.0, 0.0)
    _barrier(distributed)
    return best_checkpoint_path, test_metrics


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: SequenceCrossEntropyLoss,
    device: torch.device,
    padding_idx: int,
    vocab_size: int,
    teacher_noise: float,
    log_every_n_steps: int,
    logger: logging.Logger,
    should_log: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for step, batch in enumerate(loader, start=1):
        images, decoder_input, labels, _ = _move_batch(batch, device)
        decoder_input = _apply_teacher_forcing_noise(decoder_input, padding_idx, vocab_size, teacher_noise)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images, decoder_input)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        steps += 1
        if should_log and step % max(log_every_n_steps, 1) == 0:
            logger.info("train_step=%s loss=%.6f", step, total_loss / steps)
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate_autoregressive(
    model: nn.Module,
    loader: DataLoader,
    i2w: dict[int, str],
    bos_idx: int,
    eos_idx: int,
    max_length: int,
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
        images, _, labels, lengths = _move_batch(batch, device)
        generated = model.generate(images, bos_idx=bos_idx, eos_idx=eos_idx, max_length=max_length)
        for sample_idx in range(images.size(0)):
            pred_tokens = _ids_to_tokens(generated[sample_idx].detach().cpu().tolist(), i2w)
            label_length = int(lengths[sample_idx].detach().cpu())
            ref_tokens = _ids_to_tokens(labels[sample_idx, :label_length].detach().cpu().tolist(), i2w)
            hypothesis = decoder_tokens_to_kern(pred_tokens)
            reference = decoder_tokens_to_kern(ref_tokens)
            hypotheses.append(hypothesis)
            references.append(reference)
            if write_predictions and hyp_dir is not None and gt_dir is not None:
                filename = f"{batch_idx}.krn" if images.size(0) == 1 else f"{batch_idx}_{sample_idx}.krn"
                (hyp_dir / filename).write_text(hypothesis, encoding="utf-8")
                (gt_dir / filename).write_text(reference, encoding="utf-8")
    return get_metrics(hypotheses, references)


def _ids_to_tokens(ids: list[int], i2w: dict[int, str]) -> list[str]:
    tokens = []
    for token_id in ids:
        token = i2w.get(int(token_id))
        if token is None:
            continue
        if token == EOS_TOKEN:
            break
        if token != BOS_TOKEN:
            tokens.append(token)
    return tokens


def _move_batch(batch, device: torch.device):
    images, decoder_input, labels, lengths = batch
    return images.to(device), decoder_input.to(device), labels.to(device), lengths.to(device)


def _apply_teacher_forcing_noise(
    decoder_input: torch.Tensor,
    padding_idx: int,
    vocab_size: int,
    probability: float,
) -> torch.Tensor:
    if probability <= 0:
        return decoder_input
    noisy = decoder_input.clone()
    mask = torch.rand_like(noisy, dtype=torch.float32) < float(probability)
    mask &= noisy.ne(int(padding_idx))
    mask[:, 0] = False
    random_tokens = torch.randint(0, int(vocab_size), noisy.shape, device=noisy.device)
    noisy[mask] = random_tokens[mask]
    return noisy


def _is_improved(current: float, best: float, min_delta: float, mode: str) -> bool:
    if mode == "min":
        return current < best - min_delta
    if mode == "max":
        return current > best + min_delta
    raise ValueError(f"Unsupported early stopping mode: {mode}")


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
