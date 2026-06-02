from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Callable

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
    max_epochs = int(training_cfg["max_epochs"])
    start_epoch = 1
    resume_step = 0
    resume_loss_sum = 0.0
    resume_steps = 0

    resume_checkpoint_path = _select_resume_checkpoint(training_cfg, weights_dir)
    if resume_checkpoint_path is not None:
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint {resume_checkpoint_path} does not contain optimizer_state_dict")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state_to_device(optimizer, device)

        resume_epoch = int(checkpoint["epoch"])
        checkpoint_kind = str(checkpoint.get("checkpoint_kind", "epoch"))
        if checkpoint_kind == "step":
            resume_step = int(checkpoint.get("step", 0))
            resume_steps = int(checkpoint.get("train_steps", resume_step))
            train_loss_value = float(checkpoint.get("train_loss", 0.0))
            resume_loss_sum = float(checkpoint.get("train_loss_sum", train_loss_value * max(resume_steps, 1)))
            start_epoch = resume_epoch
        else:
            start_epoch = resume_epoch + 1
        best_metadata, _ = _best_metadata_for_resume(
            checkpoint=checkpoint,
            best_checkpoint_path=best_checkpoint_path,
            resume_epoch=resume_epoch,
        )
        best_epoch = int(best_metadata.get("best_epoch") or best_metadata.get("epoch") or 0)
        best_metric_value = best_metadata.get("best_metric", best_metadata.get("monitored_metric", best_metric))
        best_metric = float(best_metric if best_metric_value is None else best_metric_value)
        epochs_without_improvement = _resume_patience_counter(
            checkpoint=checkpoint,
            resume_epoch=resume_epoch,
            best_epoch=best_epoch,
            validation_every=validation_every,
        )
        if is_main_process:
            logger.info(
                (
                    "Resuming autoregressive training from %s at epoch=%s step=%s; next_epoch=%s "
                    "best_epoch=%s best_metric=%.6f epochs_without_improvement=%s/%s"
                ),
                resume_checkpoint_path,
                resume_epoch,
                resume_step,
                start_epoch,
                best_epoch,
                best_metric,
                epochs_without_improvement,
                int(early_cfg["patience"]),
            )
    elif bool((training_cfg.get("resume") or {}).get("auto", False)) and is_main_process:
        logger.info("Auto-resume requested, but no checkpoint was found in %s", weights_dir)

    started_at = time.perf_counter()

    for epoch in range(start_epoch, max_epochs + 1):
        epoch_started_at = time.perf_counter()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        resume_this_epoch = epoch == start_epoch and resume_step > 0
        step_checkpoint_callback = None
        if is_main_process and bool(config.get("output", {}).get("save_latest_checkpoint", True)):
            step_checkpoint_callback = _make_step_checkpoint_callback(
                config=config,
                weights_dir=weights_dir,
                model=model,
                optimizer=optimizer,
                w2i=w2i,
                i2w=i2w,
                epoch=epoch,
                monitor=early_cfg["monitor"],
                best_epoch=best_epoch,
                best_metric=best_metric,
                epochs_without_improvement=epochs_without_improvement,
                logger=logger,
            )
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
            start_step=resume_step if resume_this_epoch else 0,
            initial_loss_sum=resume_loss_sum if resume_this_epoch else 0.0,
            initial_steps=resume_steps if resume_this_epoch else 0,
            step_checkpoint_callback=step_checkpoint_callback,
        )
        resume_step = 0
        resume_loss_sum = 0.0
        resume_steps = 0
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
                payload = _build_checkpoint_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    w2i=w2i,
                    i2w=i2w,
                    train_loss=train_loss,
                    val_cer=None,
                    val_ser=None,
                    val_ler=None,
                    monitor=early_cfg["monitor"],
                    monitored_metric=None,
                    best_epoch=best_epoch,
                    best_metric=best_metric,
                    epochs_without_improvement=epochs_without_improvement,
                    validated=False,
                )
                _save_epoch_checkpoint(config, weights_dir, payload, logger)
                _save_latest_checkpoint(config, weights_dir, payload, logger, quiet=True)
            continue

        _barrier(distributed)
        if is_main_process:
            validation_started_at = time.perf_counter()
            logger.info(
                "Starting autoregressive validation epoch=%s batches=%s max_length=%s length_margin=%s",
                epoch,
                len(val_loader),
                int(config["decoding"]["max_length"]),
                int(config["decoding"].get("length_margin", 32)),
            )
            val_cer, val_ser, val_ler = evaluate_autoregressive(
                model=_unwrap_model(model),
                loader=val_loader,
                i2w=i2w,
                bos_idx=w2i[BOS_TOKEN],
                eos_idx=w2i[EOS_TOKEN],
                max_length=int(config["decoding"]["max_length"]),
                length_margin=int(config["decoding"].get("length_margin", 32)),
                device=device,
            )
            validation_seconds = time.perf_counter() - validation_started_at
            logger.info(
                (
                    "epoch=%s train_loss=%.6f val_CER=%.4f val_SER=%.4f val_LER=%.4f "
                    "validation_time=%s epoch_time=%s elapsed=%s"
                ),
                epoch,
                train_loss,
                val_cer,
                val_ser,
                val_ler,
                _format_duration(validation_seconds),
                _format_duration(epoch_seconds),
                _format_duration(elapsed_seconds),
            )
        else:
            logger.info("Waiting for rank 0 autoregressive validation at epoch=%s", epoch)
            val_cer, val_ser, val_ler = 0.0, 0.0, 0.0

        should_stop = False
        if is_main_process:
            monitored = {"val_CER": val_cer, "val_SER": val_ser, "val_LER": val_ler}[early_cfg["monitor"]]
            improved = _is_improved(monitored, best_metric, float(early_cfg["min_delta"]), early_cfg["mode"])
            if improved:
                best_metric = monitored
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            payload = _build_checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                config=config,
                w2i=w2i,
                i2w=i2w,
                train_loss=train_loss,
                val_cer=val_cer,
                val_ser=val_ser,
                val_ler=val_ler,
                monitor=early_cfg["monitor"],
                monitored_metric=monitored,
                best_epoch=best_epoch,
                best_metric=best_metric,
                epochs_without_improvement=epochs_without_improvement,
                validated=True,
            )
            _save_epoch_checkpoint(config, weights_dir, payload, logger)
            _save_latest_checkpoint(config, weights_dir, payload, logger, quiet=True)

            if improved:
                _atomic_torch_save(payload, best_checkpoint_path)
                logger.info("Saved best checkpoint to %s", best_checkpoint_path)
            else:
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
            length_margin=int(config["decoding"].get("length_margin", 32)),
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
    start_step: int = 0,
    initial_loss_sum: float = 0.0,
    initial_steps: int = 0,
    step_checkpoint_callback: Callable[[int, float, int], None] | None = None,
) -> float:
    model.train()
    total_loss = float(initial_loss_sum)
    steps = int(initial_steps)
    if should_log and start_step > 0:
        logger.info("Skipping %s already-trained batches from resumed epoch", start_step)
    for step, batch in enumerate(loader, start=1):
        if step <= start_step:
            continue
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
        if step_checkpoint_callback is not None:
            step_checkpoint_callback(step, total_loss, steps)
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate_autoregressive(
    model: nn.Module,
    loader: DataLoader,
    i2w: dict[int, str],
    bos_idx: int,
    eos_idx: int,
    max_length: int,
    length_margin: int,
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
        reference_max_length = int(lengths.detach().max().cpu())
        batch_max_length = min(int(max_length), reference_max_length + max(int(length_margin), 0))
        generated = model.generate(images, bos_idx=bos_idx, eos_idx=eos_idx, max_length=batch_max_length)
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


def _make_step_checkpoint_callback(
    *,
    config: dict[str, Any],
    weights_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    w2i: dict[str, int],
    i2w: dict[int, str],
    epoch: int,
    monitor: str,
    best_epoch: int,
    best_metric: float,
    epochs_without_improvement: int,
    logger: logging.Logger,
) -> Callable[[int, float, int], None]:
    interval = max(int(config.get("output", {}).get("latest_checkpoint_every_n_steps", 100)), 1)

    def _save(step: int, train_loss_sum: float, train_steps: int) -> None:
        if step % interval != 0:
            return
        payload = _build_checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            config=config,
            w2i=w2i,
            i2w=i2w,
            train_loss=train_loss_sum / max(train_steps, 1),
            val_cer=None,
            val_ser=None,
            val_ler=None,
            monitor=monitor,
            monitored_metric=None,
            best_epoch=best_epoch,
            best_metric=best_metric,
            epochs_without_improvement=epochs_without_improvement,
            validated=False,
            checkpoint_kind="step",
            step=step,
            train_loss_sum=train_loss_sum,
            train_steps=train_steps,
        )
        _save_latest_checkpoint(config, weights_dir, payload, logger)

    return _save


def _build_checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    w2i: dict[str, int],
    i2w: dict[int, str],
    train_loss: float,
    val_cer: float | None,
    val_ser: float | None,
    val_ler: float | None,
    monitor: str,
    monitored_metric: float | None,
    best_epoch: int,
    best_metric: float,
    epochs_without_improvement: int,
    validated: bool,
    checkpoint_kind: str = "epoch",
    step: int = 0,
    train_loss_sum: float | None = None,
    train_steps: int | None = None,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "checkpoint_kind": checkpoint_kind,
        "step": step,
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "w2i": w2i,
        "i2w": i2w,
        "train_loss": train_loss,
        "train_loss_sum": train_loss if train_loss_sum is None else train_loss_sum,
        "train_steps": step if train_steps is None else train_steps,
        "val_CER": val_cer,
        "val_SER": val_ser,
        "val_LER": val_ler,
        "monitor": monitor,
        "monitored_metric": monitored_metric,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "epochs_without_improvement": epochs_without_improvement,
        "validated": validated,
    }


def _save_epoch_checkpoint(
    config: dict[str, Any],
    weights_dir: Path,
    payload: dict[str, Any],
    logger: logging.Logger,
) -> Path | None:
    if not bool(config.get("output", {}).get("save_epoch_checkpoints", True)):
        return None
    epoch_path = weights_dir / f"epoch_{int(payload['epoch']):04d}.pt"
    _atomic_torch_save(payload, epoch_path)
    logger.info("Saved epoch checkpoint to %s", epoch_path)
    return epoch_path


def _save_latest_checkpoint(
    config: dict[str, Any],
    weights_dir: Path,
    payload: dict[str, Any],
    logger: logging.Logger,
    *,
    quiet: bool = False,
) -> Path | None:
    if not bool(config.get("output", {}).get("save_latest_checkpoint", True)):
        return None
    latest_path = weights_dir / "latest.pt"
    _atomic_torch_save(payload, latest_path)
    if not quiet:
        logger.info(
            "Saved latest autoregressive checkpoint to %s epoch=%s step=%s",
            latest_path,
            payload.get("epoch"),
            payload.get("step", 0),
        )
    return latest_path


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _select_resume_checkpoint(training_cfg: dict[str, Any], weights_dir: Path) -> Path | None:
    resume_cfg = training_cfg.get("resume") or {}
    explicit_checkpoint = resume_cfg.get("checkpoint") or training_cfg.get("resume_checkpoint")
    if explicit_checkpoint:
        checkpoint_path = Path(str(explicit_checkpoint)).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint is not a file: {checkpoint_path}")
        return checkpoint_path

    auto_resume = bool(resume_cfg.get("auto", False) or training_cfg.get("auto_resume", False))
    if auto_resume:
        return _latest_resume_checkpoint(weights_dir)
    return None


def _latest_resume_checkpoint(weights_dir: Path) -> Path | None:
    candidates = []
    latest_path = weights_dir / "latest.pt"
    if latest_path.is_file():
        candidates.append(latest_path)
    epoch_path = _latest_epoch_checkpoint(weights_dir)
    if epoch_path is not None:
        candidates.append(epoch_path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_epoch_checkpoint(weights_dir: Path) -> Path | None:
    latest_epoch = -1
    latest_path: Path | None = None
    for path in weights_dir.glob("epoch_*.pt"):
        try:
            epoch = int(path.stem.rsplit("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_path = path
    return latest_path


def _best_metadata_for_resume(
    checkpoint: dict[str, Any],
    best_checkpoint_path: Path,
    resume_epoch: int,
) -> tuple[dict[str, Any], Path | None]:
    if best_checkpoint_path.exists():
        best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        best_epoch = int(best_checkpoint.get("best_epoch") or best_checkpoint.get("epoch") or 0)
        if 0 < best_epoch <= resume_epoch:
            return best_checkpoint, best_checkpoint_path
    return checkpoint, None


def _resume_patience_counter(
    checkpoint: dict[str, Any],
    resume_epoch: int,
    best_epoch: int,
    validation_every: int,
) -> int:
    if "epochs_without_improvement" in checkpoint:
        return int(checkpoint["epochs_without_improvement"])
    if best_epoch > 0 and resume_epoch >= best_epoch:
        return max((resume_epoch - best_epoch) // max(validation_every, 1), 0)
    return 0


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


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
