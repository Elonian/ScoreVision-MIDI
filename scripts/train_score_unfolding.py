from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.score_unfolding import build_model
from utils.config import load_yaml_config, resolve_path
from utils.logging import log_environment, setup_logging
from utils.training import run_training
from utils.vocabulary import load_or_create_vocabulary, vocabulary_paths
from torchinfo import summary as torch_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train score unfolding OMR with CTC.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-name", default=None, help="Override project.run_name.")
    parser.add_argument("--model-name", default=None, choices=["FCN", "CRNN", "CNNT"], help="Override model.name.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override training.max_epochs.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per split for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size per process.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override training.num_workers.")
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume from a checkpoint path. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from the latest epoch_*.pt checkpoint in the run output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed = init_distributed()
    project_root = PROJECT_ROOT
    config = load_yaml_config(args.config)
    apply_cli_overrides(config, args, project_root)
    config["distributed"] = distributed

    run_name = config["project"]["run_name"]
    log_dir = resolve_path(config["logging"]["log_dir"], project_root)
    output_root = resolve_path(config["output"]["root"], project_root)
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if distributed["is_main_process"]:
        (output_dir / "hyp").mkdir(parents=True, exist_ok=True)
        (output_dir / "gt").mkdir(parents=True, exist_ok=True)

    log_run_name = run_name if distributed["is_main_process"] else f"{run_name}_rank{distributed['rank']}"
    logger, log_path = setup_logging(log_dir=log_dir, run_name=log_run_name, level=config["logging"]["level"])
    log_environment(logger, config, output_dir=output_dir)
    logger.info("Training output directory: %s", output_dir)

    seed_everything(int(config["project"].get("seed", 42)))

    train_dataset, val_dataset, test_dataset = build_datasets(config, project_root, logger)
    vocab_name = config["vocab"]["name"]
    if config["data"].get("max_samples") is not None:
        vocab_name = f"{vocab_name}_sample{int(config['data']['max_samples'])}"
    w2i, i2w = load_or_create_vocabulary_distributed(
        datasets=[train_dataset, val_dataset, test_dataset],
        config=config,
        vocab_name=vocab_name,
        project_root=project_root,
        logger=logger,
    )
    for dataset in (train_dataset, val_dataset, test_dataset):
        dataset.set_dictionaries(w2i, i2w)

    train_loader, val_loader, test_loader = build_dataloaders(
        config,
        train_dataset,
        val_dataset,
        test_dataset,
        distributed,
        logger,
    )
    model_name = config["model"]["name"].upper()
    needs_max_hw = model_name in {"CNNT", "STAVE_CNNT"}
    configured_max_height = config["data"].get("max_height")
    configured_max_width = config["data"].get("max_width")
    has_configured_max_hw = configured_max_height is not None and configured_max_width is not None
    if needs_max_hw and has_configured_max_hw:
        max_height, max_width = int(configured_max_height), int(configured_max_width)
    elif needs_max_hw and distributed.get("enabled") and not distributed.get("is_main_process"):
        _wait_for_dataset_hw_caches([train_dataset], logger)
        max_height, max_width = train_dataset.get_max_hw()
    else:
        max_height, max_width = train_dataset.get_max_hw() if needs_max_hw else train_dataset.get_sample_hw(0)
    blank_idx = len(i2w)
    out_size = train_dataset.vocab_size() + 1
    if distributed["is_main_process"]:
        logger.info(
            "Dataset sizes: train=%s val=%s test=%s vocab=%s blank_idx=%s %s=(%s,%s)",
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
            train_dataset.vocab_size(),
            blank_idx,
            "max_hw" if needs_max_hw else "reference_hw",
            max_height,
            max_width,
        )

    network = build_model(
        model_name=config["model"]["name"],
        max_width=max_width,
        max_height=max_height,
        in_channels=int(config["model"]["in_channels"]),
        out_size=out_size,
        dropout=float(config["model"].get("dropout", 0.4)),
        max_len=config["model"].get("max_len"),
        pretrain_path=config["model"].get("pretrain_path"),
    )
    if distributed["is_main_process"]:
        maybe_log_model_summary(network, config, max_height, max_width, logger)

    checkpoint_path, test_metrics = run_training(
        config=config,
        model=network,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        blank_idx=blank_idx,
        i2w=i2w,
        output_dir=output_dir,
        logger=logger,
    )

    if distributed["is_main_process"]:
        logger.info("Best checkpoint: %s", checkpoint_path)
        logger.info(
            "Finished training: test_CER=%.4f test_SER=%.4f test_LER=%.4f",
            test_metrics[0],
            test_metrics[1],
            test_metrics[2],
        )
    logger.info("Log file: %s", log_path)
    cleanup_distributed(distributed)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace, project_root: Path) -> None:
    if args.run_name:
        config["project"]["run_name"] = args.run_name
    if args.model_name:
        config["model"]["name"] = args.model_name
        config["project"]["run_name"] = replace_model_suffix(config["project"]["run_name"], args.model_name)
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    if getattr(args, "max_samples", None) is not None:
        config["data"]["max_samples"] = args.max_samples
    if getattr(args, "batch_size", None) is not None:
        config["training"]["batch_size"] = args.batch_size
    if getattr(args, "num_workers", None) is not None:
        config["training"]["num_workers"] = args.num_workers
    resume_cfg = config["training"].setdefault("resume", {})
    if getattr(args, "resume_checkpoint", None) is not None:
        resume_cfg["checkpoint"] = str(resolve_path(args.resume_checkpoint, project_root))
    if getattr(args, "auto_resume", False):
        resume_cfg["auto"] = True


def replace_model_suffix(run_name: str, model_name: str) -> str:
    parts = run_name.split("_")
    if parts and parts[-1].upper() in {"FCN", "CRNN", "CNNT"}:
        parts[-1] = model_name.lower()
        return "_".join(parts)
    return f"{run_name}_{model_name.lower()}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = {
        "enabled": world_size > 1,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main_process": rank == 0,
        "backend": os.environ.get("DIST_BACKEND", "nccl"),
    }
    if distributed["enabled"]:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training was requested but CUDA is not available.")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        if distributed["backend"] == "nccl":
            dist.init_process_group(backend="nccl", device_id=device)
        else:
            dist.init_process_group(backend=distributed["backend"])
    return distributed


def cleanup_distributed(distributed: dict[str, Any]) -> None:
    if distributed.get("enabled") and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(distributed: dict[str, Any]) -> None:
    if distributed.get("enabled") and dist.is_initialized():
        if distributed.get("backend") == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[int(distributed["local_rank"])])
        else:
            dist.barrier()


def build_datasets(
    config: dict[str, Any],
    project_root: Path,
    logger: logging.Logger,
) -> tuple[GrandStaffDataset, GrandStaffDataset, GrandStaffDataset]:
    from utils.data import GrandStaffDataset

    data_cfg = config["data"]
    data_root = resolve_path(data_cfg["data_root"], project_root)
    common = {
        "data_root": data_root,
        "resize_ratio": float(data_cfg["resize_ratio"]),
        "load_distorted": bool(data_cfg["load_distorted"]),
        "extension": data_cfg["extension"],
        "max_samples": data_cfg.get("max_samples"),
        "preload_images": bool(data_cfg.get("preload_images", False)),
        "image_cache_dir": resolve_path(data_cfg.get("image_cache_dir"), project_root),
        "metadata_cache_dir": resolve_path(data_cfg.get("metadata_cache_dir"), project_root),
        "logger": logger,
    }

    train_dataset = GrandStaffDataset(
        partition_file=resolve_path(data_cfg["train_partition"], project_root),
        **common,
    )
    val_dataset = GrandStaffDataset(
        partition_file=resolve_path(data_cfg["val_partition"], project_root),
        **common,
    )
    test_dataset = GrandStaffDataset(
        partition_file=resolve_path(data_cfg["test_partition"], project_root),
        **common,
    )
    return train_dataset, val_dataset, test_dataset


def load_or_create_vocabulary_distributed(
    datasets: list[GrandStaffDataset],
    config: dict[str, Any],
    vocab_name: str,
    project_root: Path,
    logger: logging.Logger,
) -> tuple[dict[str, int], dict[int, str]]:
    distributed = config["distributed"]
    vocab_dir = resolve_path(config["vocab"]["directory"], project_root)
    sort_tokens = bool(config["vocab"].get("sort_tokens", False))

    if distributed["is_main_process"]:
        try:
            w2i, i2w = load_or_create_vocabulary(
                None,
                vocab_dir=vocab_dir,
                name=vocab_name,
                sort_tokens=sort_tokens,
                logger=logger,
            )
        except FileNotFoundError:
            logger.info("Vocabulary is missing; reading all transcriptions to create it")
            w2i, i2w = load_or_create_vocabulary(
                [dataset.get_gt() for dataset in datasets],
                vocab_dir=vocab_dir,
                name=vocab_name,
                sort_tokens=sort_tokens,
                logger=logger,
            )
        logger.info("Vocabulary ready: size=%s", len(w2i))
        return w2i, i2w

    w2i_path, i2w_path = vocabulary_paths(vocab_dir, vocab_name)
    _wait_for_paths([w2i_path, i2w_path], logger=logger, description="vocabulary files")
    w2i, i2w = load_or_create_vocabulary(
        None,
        vocab_dir=vocab_dir,
        name=vocab_name,
        sort_tokens=sort_tokens,
        logger=logger,
    )
    return w2i, i2w


def build_dataloaders(
    config: dict[str, Any],
    train_dataset: GrandStaffDataset,
    val_dataset: GrandStaffDataset,
    test_dataset: GrandStaffDataset,
    distributed: dict[str, Any],
    logger: logging.Logger,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    from utils.data import LengthBucketBatchSampler, ctc_collate

    training_cfg = config["training"]
    num_workers = int(training_cfg["num_workers"])
    batch_size = int(training_cfg["batch_size"])
    use_bucketing = bool(training_cfg.get("bucket_by_length", False))
    shuffle = bool(training_cfg.get("shuffle", False))
    seed = int(config["project"].get("seed", 42))
    common_loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": ctc_collate,
        "persistent_workers": num_workers > 0,
        "pin_memory": torch.cuda.is_available(),
    }
    if use_bucketing:
        rank = int(distributed.get("rank", 0)) if distributed.get("enabled") else 0
        world_size = int(distributed.get("world_size", 1)) if distributed.get("enabled") else 1
        if distributed.get("enabled") and not distributed.get("is_main_process"):
            _wait_for_dataset_hw_caches([train_dataset, val_dataset, test_dataset], logger)
        logger.info("Collecting sample heights for length-bucketed batches")
        train_heights = train_dataset.get_sample_heights()
        val_heights = val_dataset.get_sample_heights()
        test_heights = test_dataset.get_sample_heights()
        train_batch_sampler = LengthBucketBatchSampler(
            lengths=train_heights,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            shuffle=shuffle,
            seed=seed,
        )
        val_batch_sampler = LengthBucketBatchSampler(
            lengths=val_heights,
            batch_size=batch_size,
        )
        test_batch_sampler = LengthBucketBatchSampler(
            lengths=test_heights,
            batch_size=batch_size,
        )
        logger.info(
            (
                "Using length-bucketed batches: train_batches=%s val_batches=%s test_batches=%s "
                "batch_size=%s rank=%s world_size=%s"
            ),
            len(train_batch_sampler),
            len(val_batch_sampler),
            len(test_batch_sampler),
            batch_size,
            rank,
            world_size,
        )
        return (
            DataLoader(train_dataset, batch_sampler=train_batch_sampler, **common_loader_kwargs),
            DataLoader(val_dataset, batch_sampler=val_batch_sampler, **common_loader_kwargs),
            DataLoader(test_dataset, batch_sampler=test_batch_sampler, **common_loader_kwargs),
        )

    train_sampler = None
    if distributed.get("enabled"):
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=int(distributed["world_size"]),
            rank=int(distributed["rank"]),
            shuffle=shuffle,
            drop_last=False,
        )
    loader_kwargs = {
        "batch_size": batch_size,
        **common_loader_kwargs,
    }
    return (
        DataLoader(
            train_dataset,
            sampler=train_sampler,
            shuffle=train_sampler is None and bool(training_cfg.get("shuffle", False)),
            **loader_kwargs,
        ),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    )


def _wait_for_dataset_hw_caches(datasets: list[GrandStaffDataset], logger: logging.Logger) -> None:
    cache_paths = []
    for dataset in datasets:
        cache_path = dataset.hw_cache_path()
        if cache_path is not None:
            cache_paths.append(cache_path)
    if cache_paths:
        _wait_for_paths(cache_paths, logger=logger, description="image-shape cache files")


def _wait_for_paths(
    paths: list[Path],
    logger: logging.Logger,
    description: str,
    timeout_seconds: int = 3600,
    poll_seconds: float = 2.0,
) -> None:
    started_at = time.monotonic()
    missing = [path for path in paths if not path.exists()]
    if missing:
        logger.info("Waiting for %s: %s", description, ", ".join(str(path) for path in missing))
    while missing:
        if time.monotonic() - started_at > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for {description}: {', '.join(str(path) for path in missing)}"
            )
        time.sleep(poll_seconds)
        missing = [path for path in paths if not path.exists()]


def maybe_log_model_summary(
    network: torch.nn.Module,
    config: dict[str, Any],
    max_height: int,
    max_width: int,
    logger: logging.Logger,
) -> None:
    if not config["model"].get("print_summary", True):
        return

    input_size = (1, int(config["model"]["in_channels"]), max_height, max_width)
    logger.info("Model summary:\n%s", torch_summary(network, input_size=input_size, dtypes=[torch.float]))


if __name__ == "__main__":
    main()
