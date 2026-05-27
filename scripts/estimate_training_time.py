from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.ctc_loss import CTCSequenceLoss
from models.score_unfolding import build_model
from scripts.train_score_unfolding import apply_cli_overrides, build_datasets, seed_everything
from utils.config import load_yaml_config, resolve_path
from utils.vocabulary import make_vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate score unfolding training time.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-name", default=None, help="Override project.run_name.")
    parser.add_argument("--model-name", default=None, choices=["FCN", "CRNN", "CNNT"], help="Override model.name.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override training.max_epochs.")
    parser.add_argument("--max-samples", type=int, default=256, help="Samples per split to load for the estimate.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override training.num_workers.")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Warmup optimization steps.")
    parser.add_argument("--measure-steps", type=int, default=20, help="Optimization steps to time.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(args.config)
    apply_cli_overrides(config, args)
    config["data"]["max_samples"] = args.max_samples

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("estimate_training_time")

    seed_everything(int(config["project"].get("seed", 42)))
    train_dataset, val_dataset, test_dataset = build_datasets(config, project_root, logger)
    w2i, i2w = make_vocabulary([train_dataset.get_gt(), val_dataset.get_gt(), test_dataset.get_gt()])
    for dataset in (train_dataset, val_dataset, test_dataset):
        dataset.set_dictionaries(w2i, i2w)

    training_cfg = config["training"]
    batch_size = int(training_cfg["batch_size"])
    from utils.data import ctc_collate

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(training_cfg["num_workers"]),
        collate_fn=ctc_collate,
        persistent_workers=int(training_cfg["num_workers"]) > 0,
    )

    max_height, max_width = train_dataset.get_max_hw()
    blank_idx = len(i2w)
    model = build_model(
        model_name=config["model"]["name"],
        max_width=max_width,
        max_height=max_height,
        in_channels=int(config["model"]["in_channels"]),
        out_size=train_dataset.vocab_size() + 1,
        dropout=float(config["model"].get("dropout", 0.4)),
        max_len=config["model"].get("max_len"),
        pretrain_path=config["model"].get("pretrain_path"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))
    loss_fn = CTCSequenceLoss(blank_idx=blank_idx, zero_infinity=bool(config["loss"].get("zero_infinity", False)))

    logger.info("Benchmarking on device=%s with max_samples=%s", device, args.max_samples)
    seconds_per_step = benchmark_steps(
        model=model,
        loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        warmup_steps=max(args.warmup_steps, 0),
        measure_steps=max(args.measure_steps, 1),
    )

    full_train_samples = count_partition_entries(resolve_path(config["data"]["train_partition"], project_root))
    steps_per_epoch = math.ceil(full_train_samples / batch_size)
    train_seconds_per_epoch = seconds_per_step * steps_per_epoch
    max_epochs = int(training_cfg["max_epochs"])

    print()
    print(f"device: {device}")
    print(f"measured seconds/step: {seconds_per_step:.3f}")
    print(f"full train samples: {full_train_samples}")
    print(f"steps/epoch at batch_size={batch_size}: {steps_per_epoch}")
    print(f"estimated train-only epoch time: {format_duration(train_seconds_per_epoch)}")
    print(f"estimated train-only time for {max_epochs} epochs: {format_duration(train_seconds_per_epoch * max_epochs)}")
    print("validation, checkpointing, and early stopping are not included in this estimate.")


def benchmark_steps(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CTCSequenceLoss,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
) -> float:
    model.train()
    batches = cycle(loader)

    for _ in range(warmup_steps):
        run_step(model, next(batches), optimizer, loss_fn, device)

    synchronize(device)
    timings: list[float] = []
    for _ in range(measure_steps):
        started_at = time.perf_counter()
        run_step(model, next(batches), optimizer, loss_fn, device)
        synchronize(device)
        timings.append(time.perf_counter() - started_at)

    return float(np.mean(timings))


def run_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    loss_fn: CTCSequenceLoss,
    device: torch.device,
) -> None:
    images, targets, input_lengths, target_lengths = batch
    optimizer.zero_grad(set_to_none=True)
    predictions = model(images.to(device))
    loss = loss_fn(predictions, targets.to(device), input_lengths.cpu(), target_lengths.cpu())
    loss.backward()
    optimizer.step()


def count_partition_entries(partition_file: Path | None) -> int:
    if partition_file is None:
        return 0
    return sum(1 for line in partition_file.read_text(encoding="utf-8").splitlines() if line.strip())


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_duration(seconds: float) -> str:
    total_seconds = int(round(max(float(seconds), 0.0)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()
