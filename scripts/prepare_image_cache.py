from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_config, resolve_path
from utils.data import (
    _load_score_image,
    _read_transcription_tokens,
    _resolve_transcription_path,
    image_cache_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cached score images for training.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--cache-dir", default=None, help="Override data.image_cache_dir.")
    parser.add_argument("--data-root", default=None, help="Override data.data_root.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cached images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    project_root = PROJECT_ROOT

    data_cfg = config["data"]
    data_root = Path(args.data_root) if args.data_root else resolve_path(data_cfg["data_root"], project_root)
    cache_dir = Path(args.cache_dir) if args.cache_dir else resolve_path(data_cfg.get("image_cache_dir"), project_root)
    if cache_dir is None:
        raise ValueError("Set data.image_cache_dir in the config or pass --cache-dir.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("prepare_image_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    partition_files = [
        resolve_path(data_cfg["train_partition"], project_root),
        resolve_path(data_cfg["val_partition"], project_root),
        resolve_path(data_cfg["test_partition"], project_root),
    ]
    partition_entries = load_partition_entries(partition_files)

    started_at = time.perf_counter()
    prepared = 0
    skipped_existing = 0
    skipped_missing = 0

    logger.info("Preparing %s cached images into %s", len(partition_entries), cache_dir)
    for index, relative_path in enumerate(partition_entries, start=1):
        transcription_path = _resolve_transcription_path(relative_path, data_root, data_cfg["extension"])
        if not transcription_path.exists():
            skipped_missing += 1
            logger.warning("Missing transcription: %s", transcription_path)
            continue

        cache_path = image_cache_path(
            transcription_path=transcription_path,
            data_root=data_root,
            cache_dir=cache_dir,
        )
        if cache_path.exists() and not args.force:
            skipped_existing += 1
        else:
            tokens = _read_transcription_tokens(transcription_path)
            image = _load_score_image(
                transcription_path=transcription_path,
                resize_ratio=float(data_cfg["resize_ratio"]),
                load_distorted=bool(data_cfg["load_distorted"]),
                target_length=len(tokens),
                logger=logger,
            )
            if image is None:
                skipped_missing += 1
                continue

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, image, allow_pickle=False)
            prepared += 1

        if index % 1000 == 0:
            elapsed = time.perf_counter() - started_at
            rate = index / max(elapsed, 1e-9)
            remaining = (len(partition_entries) - index) / max(rate, 1e-9)
            logger.info(
                "cache_progress=%s/%s prepared=%s existing=%s missing=%s elapsed=%s eta=%s",
                index,
                len(partition_entries),
                prepared,
                skipped_existing,
                skipped_missing,
                format_duration(elapsed),
                format_duration(remaining),
            )

    elapsed = time.perf_counter() - started_at
    logger.info(
        "cache_done total=%s prepared=%s existing=%s missing=%s elapsed=%s",
        len(partition_entries),
        prepared,
        skipped_existing,
        skipped_missing,
        format_duration(elapsed),
    )


def load_partition_entries(partition_files: list[Path | None]) -> list[str]:
    entries: list[str] = []
    seen: set[str] = set()
    for partition_file in partition_files:
        if partition_file is None:
            continue
        with partition_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                relative_path = line.strip()
                if relative_path and relative_path not in seen:
                    entries.append(relative_path)
                    seen.add(relative_path)
    return entries


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
