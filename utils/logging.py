from __future__ import annotations

import json
import logging
import os
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def setup_logging(
    log_dir: str | Path,
    run_name: str,
    level: str = "INFO",
) -> tuple[logging.Logger, Path]:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{run_name}_{timestamp}.log"

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("Logging to %s", log_path)
    return logger, log_path


def log_environment(logger: logging.Logger, config: dict[str, Any], output_dir: str | Path) -> None:
    environment = {
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "pid": os.getpid(),
        "output_dir": str(output_dir),
        "config_path": config.get("_config_path"),
    }

    environment.update(
        {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        }
    )

    logger.info("Runtime environment:\n%s", json.dumps(environment, indent=2, sort_keys=True))
    safe_config = {key: value for key, value in config.items() if not key.startswith("_")}
    logger.info("Resolved config:\n%s", json.dumps(safe_config, indent=2, sort_keys=True, default=str))
