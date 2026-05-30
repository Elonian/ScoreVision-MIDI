from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from utils.constants import PAD_TOKEN


def load_or_create_vocabulary(
    sequence_groups: list[list[list[str]]] | None,
    vocab_dir: str | Path,
    name: str,
    sort_tokens: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, int], dict[int, str]]:
    logger = logger or logging.getLogger(__name__)
    vocab_dir = Path(vocab_dir)
    vocab_dir.mkdir(parents=True, exist_ok=True)

    w2i_path, i2w_path = vocabulary_paths(vocab_dir, name)

    if w2i_path.exists() and i2w_path.exists():
        return load_vocabulary(vocab_dir, name, logger=logger)

    if sequence_groups is None:
        raise FileNotFoundError(f"Vocabulary files not found: {w2i_path} and {i2w_path}")

    logger.info("Creating vocabulary at %s", vocab_dir)
    w2i, i2w = make_vocabulary(sequence_groups, sort_tokens=sort_tokens)
    np.save(w2i_path, w2i)
    np.save(i2w_path, i2w)
    logger.info("Vocabulary size: %s", len(w2i))
    return w2i, i2w


def vocabulary_paths(vocab_dir: str | Path, name: str) -> tuple[Path, Path]:
    vocab_dir = Path(vocab_dir)
    return vocab_dir / f"{name}w2i.npy", vocab_dir / f"{name}i2w.npy"


def load_vocabulary(
    vocab_dir: str | Path,
    name: str,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, int], dict[int, str]]:
    logger = logger or logging.getLogger(__name__)
    w2i_path, i2w_path = vocabulary_paths(vocab_dir, name)
    logger.info("Loading vocabulary from %s and %s", w2i_path, i2w_path)
    w2i = np.load(w2i_path, allow_pickle=True).item()
    i2w = np.load(i2w_path, allow_pickle=True).item()
    return {str(key): int(value) for key, value in w2i.items()}, {
        int(key): str(value) for key, value in i2w.items()
    }


def make_vocabulary(
    sequence_groups: list[list[list[str]]],
    sort_tokens: bool = False,
) -> tuple[dict[str, int], dict[int, str]]:
    vocabulary = set()
    for samples in sequence_groups:
        for element in samples:
            vocabulary.update(element)

    symbols = sorted(vocabulary) if sort_tokens else list(vocabulary)
    w2i = {symbol: index + 1 for index, symbol in enumerate(symbols)}
    i2w = {index + 1: symbol for index, symbol in enumerate(symbols)}

    w2i[PAD_TOKEN] = 0
    i2w[0] = PAD_TOKEN
    return w2i, i2w
