from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.score_autoregressive import build_score_autoregressive
from models.score_autoregressive_convnext import build_convnext_score_autoregressive
from utils.config import load_yaml_config, resolve_path
from utils.logging import log_environment, setup_logging
from utils.autoregressive_data import (
    LocalGrandStaffAutoregressiveDataset,
    autoregressive_collate,
    load_or_create_autoregressive_vocabulary,
)
from utils.autoregressive_training import run_autoregressive_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ScoreVision autoregressive OMR.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-name", default=None, help="Override project.run_name.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override training.max_epochs.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per split for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override training.num_workers.")
    parser.add_argument("--max-decode-length", type=int, default=None, help="Override decoding.max_length.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed = init_distributed()
    config = load_yaml_config(args.config)
    apply_cli_overrides(config, args)
    config["distributed"] = distributed

    project_root = PROJECT_ROOT
    run_name = config["project"]["run_name"]
    log_dir = resolve_path(config["logging"]["log_dir"], project_root)
    output_root = resolve_path(config["output"]["root"], project_root)
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_run_name = run_name if distributed["is_main_process"] else f"{run_name}_rank{distributed['rank']}"
    logger, log_path = setup_logging(log_dir=log_dir, run_name=log_run_name, level=config["logging"]["level"])
    log_environment(logger, config, output_dir=output_dir)
    seed_everything(int(config["project"].get("seed", 42)))

    train_dataset, val_dataset, test_dataset = build_datasets(config, project_root, logger)
    vocab_dir = resolve_path(config["vocab"]["directory"], project_root)
    w2i, i2w = load_or_create_autoregressive_vocabulary(
        [train_dataset, val_dataset, test_dataset],
        vocab_dir=vocab_dir,
        name=config["vocab"]["name"],
        sort_tokens=bool(config["vocab"].get("sort_tokens", False)),
        logger=logger,
    )
    for dataset in (train_dataset, val_dataset, test_dataset):
        dataset.set_vocabulary(w2i, i2w)

    train_loader, val_loader, test_loader = build_dataloaders(
        config,
        train_dataset,
        val_dataset,
        test_dataset,
        w2i,
        distributed,
    )
    if distributed["is_main_process"]:
        logger.info(
            "Dataset sizes: train=%s val=%s test=%s vocab=%s max_hw=(%s,%s)",
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
            len(w2i),
            int(config["data"]["max_height"]),
            int(config["data"]["max_width"]),
        )

    model = build_model(config, w2i=w2i, project_root=project_root, logger=logger)
    checkpoint_path, test_metrics = run_autoregressive_training(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        w2i=w2i,
        i2w=i2w,
        output_dir=output_dir,
        logger=logger,
    )
    logger.info("Best checkpoint: %s", checkpoint_path)
    logger.info(
        "Finished autoregressive training: test_CER=%.4f test_SER=%.4f test_LER=%.4f",
        test_metrics[0],
        test_metrics[1],
        test_metrics[2],
    )
    logger.info("Log file: %s", log_path)
    cleanup_distributed(distributed)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    if args.run_name:
        config["project"]["run_name"] = args.run_name
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.max_decode_length is not None:
        config["decoding"]["max_length"] = args.max_decode_length


def build_datasets(config: dict, project_root: Path, logger: logging.Logger):
    data_cfg = config["data"]
    data_root = resolve_path(data_cfg["data_root"], project_root)
    common = {
        "data_root": data_root,
        "extension": data_cfg.get("extension", ".bekrn"),
        "image_cache_dir": resolve_path(data_cfg.get("image_cache_dir"), project_root),
        "resize_ratio": float(data_cfg.get("resize_ratio", 1.0)),
        "max_samples": data_cfg.get("max_samples"),
        "logger": logger,
    }
    return (
        LocalGrandStaffAutoregressiveDataset(resolve_path(data_cfg["train_partition"], project_root), **common),
        LocalGrandStaffAutoregressiveDataset(resolve_path(data_cfg["val_partition"], project_root), **common),
        LocalGrandStaffAutoregressiveDataset(resolve_path(data_cfg["test_partition"], project_root), **common),
    )


def build_dataloaders(config: dict, train_dataset, val_dataset, test_dataset, w2i: dict[str, int], distributed: dict):
    training_cfg = config["training"]
    collate = partial(autoregressive_collate, padding_idx=w2i["<pad>"])
    loader_kwargs = {
        "batch_size": int(training_cfg["batch_size"]),
        "num_workers": int(training_cfg["num_workers"]),
        "collate_fn": collate,
        "persistent_workers": int(training_cfg["num_workers"]) > 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_sampler = None
    if distributed.get("enabled"):
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=int(distributed["world_size"]),
            rank=int(distributed["rank"]),
            shuffle=True,
            drop_last=False,
        )
    return (
        DataLoader(train_dataset, sampler=train_sampler, shuffle=train_sampler is None, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    )


def build_model(config: dict, w2i: dict[str, int], project_root: Path, logger: logging.Logger):
    data_cfg = config["data"]
    model_cfg = config["model"]
    model_name = model_cfg.get("name", "ScoreAutoregressive")
    common = {
        "in_channels": int(model_cfg["in_channels"]),
        "vocab_size": len(w2i),
        "padding_idx": w2i["<pad>"],
        "max_height": int(data_cfg["max_height"]),
        "max_width": int(data_cfg["max_width"]),
        "d_model": int(model_cfg.get("d_model", 256)),
        "num_decoder_layers": int(model_cfg.get("num_decoder_layers", 6)),
        "num_heads": int(model_cfg.get("num_heads", 4)),
        "dim_feedforward": int(model_cfg.get("dim_feedforward", 1024)),
        "max_seq_len": int(model_cfg.get("max_seq_len", 4096)),
    }

    if model_name == "ScoreAutoregressive":
        logger.info("Building ScoreAutoregressive with existing ScoreVision encoder")
        return build_score_autoregressive(
            **common,
            encoder_dropout=float(model_cfg.get("encoder_dropout", 0.2)),
            decoder_dropout=float(model_cfg.get("decoder_dropout", 0.1)),
        )

    if model_name == "ConvNextScoreAutoregressive":
        encoder_cfg = model_cfg.get("encoder", {})
        cache_dir = resolve_path(encoder_cfg.get("cache_dir"), project_root)
        logger.info(
            "Building ConvNextScoreAutoregressive encoder_source=%s cache_dir=%s",
            encoder_cfg.get("source", "scratch"),
            cache_dir,
        )
        return build_convnext_score_autoregressive(
            **common,
            encoder_source=encoder_cfg.get("source", "scratch"),
            pretrained_model_name=encoder_cfg.get("pretrained_model_name", "facebook/convnext-tiny-224"),
            cache_dir=cache_dir,
            local_files_only=bool(encoder_cfg.get("local_files_only", False)),
            freeze_encoder=bool(encoder_cfg.get("freeze_encoder", False)),
            scratch_hidden_sizes=encoder_cfg.get("scratch_hidden_sizes", [64, 128, 256]),
            scratch_depths=encoder_cfg.get("scratch_depths", [3, 3, 9]),
            decoder_dropout=float(model_cfg.get("decoder_dropout", 0.1)),
        )

    raise ValueError(f"Unsupported model.name: {model_name}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed() -> dict:
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


def cleanup_distributed(distributed: dict) -> None:
    if distributed.get("enabled") and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
