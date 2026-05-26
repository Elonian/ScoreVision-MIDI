from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchinfo import summary as torch_summary
from torch.utils.data import DataLoader

from models.score_unfolding import build_model
from utils.config import load_yaml_config, resolve_path
from utils.data import GrandStaffDataset, ctc_collate
from utils.logging import log_environment, setup_logging
from utils.training import run_training
from utils.vocabulary import load_or_create_vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train score unfolding OMR with CTC.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-name", default=None, help="Override project.run_name.")
    parser.add_argument("--model-name", default=None, choices=["FCN", "CRNN", "CNNT"], help="Override model.name.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override training.max_epochs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(args.config)
    apply_cli_overrides(config, args)

    run_name = config["project"]["run_name"]
    log_dir = resolve_path(config["logging"]["log_dir"], project_root)
    output_root = resolve_path(config["output"]["root"], project_root)
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hyp").mkdir(parents=True, exist_ok=True)
    (output_dir / "gt").mkdir(parents=True, exist_ok=True)

    logger, log_path = setup_logging(log_dir=log_dir, run_name=run_name, level=config["logging"]["level"])
    log_environment(logger, config, output_dir=output_dir)
    logger.info("Training output directory: %s", output_dir)

    seed_everything(int(config["project"].get("seed", 42)))

    train_dataset, val_dataset, test_dataset = build_datasets(config, project_root, logger)
    w2i, i2w = load_or_create_vocabulary(
        [train_dataset.get_gt(), val_dataset.get_gt(), test_dataset.get_gt()],
        vocab_dir=resolve_path(config["vocab"]["directory"], project_root),
        name=config["vocab"]["name"],
        sort_tokens=bool(config["vocab"].get("sort_tokens", False)),
        logger=logger,
    )
    for dataset in (train_dataset, val_dataset, test_dataset):
        dataset.set_dictionaries(w2i, i2w)

    train_loader, val_loader, test_loader = build_dataloaders(config, train_dataset, val_dataset, test_dataset)
    max_height, max_width = train_dataset.get_max_hw()
    blank_idx = len(i2w)
    out_size = train_dataset.vocab_size() + 1
    logger.info(
        "Dataset sizes: train=%s val=%s test=%s vocab=%s blank_idx=%s max_hw=(%s,%s)",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        train_dataset.vocab_size(),
        blank_idx,
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

    logger.info("Best checkpoint: %s", checkpoint_path)
    logger.info(
        "Finished training: test_CER=%.4f test_SER=%.4f test_LER=%.4f",
        test_metrics[0],
        test_metrics[1],
        test_metrics[2],
    )
    logger.info("Log file: %s", log_path)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.run_name:
        config["project"]["run_name"] = args.run_name
    if args.model_name:
        config["model"]["name"] = args.model_name
        config["project"]["run_name"] = replace_model_suffix(config["project"]["run_name"], args.model_name)
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs


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


def build_datasets(
    config: dict[str, Any],
    project_root: Path,
    logger: logging.Logger,
) -> tuple[GrandStaffDataset, GrandStaffDataset, GrandStaffDataset]:
    data_cfg = config["data"]
    data_root = resolve_path(data_cfg["data_root"], project_root)
    common = {
        "data_root": data_root,
        "resize_ratio": float(data_cfg["resize_ratio"]),
        "load_distorted": bool(data_cfg["load_distorted"]),
        "extension": data_cfg["extension"],
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


def build_dataloaders(
    config: dict[str, Any],
    train_dataset: GrandStaffDataset,
    val_dataset: GrandStaffDataset,
    test_dataset: GrandStaffDataset,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    training_cfg = config["training"]
    num_workers = int(training_cfg["num_workers"])
    loader_kwargs = {
        "batch_size": int(training_cfg["batch_size"]),
        "num_workers": num_workers,
        "collate_fn": ctc_collate,
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    )


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
