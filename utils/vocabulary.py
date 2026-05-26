from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from utils.constants import PAD_TOKEN


def load_or_create_vocabulary(
    sequence_groups: list[list[list[str]]],
    vocab_dir: str | Path,
    name: str,
    sort_tokens: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, int], dict[int, str]]:
    logger = logger or logging.getLogger(__name__)
    vocab_dir = Path(vocab_dir)
    vocab_dir.mkdir(parents=True, exist_ok=True)

    w2i_path = vocab_dir / f"{name}w2i.npy"
    i2w_path = vocab_dir / f"{name}i2w.npy"

    if w2i_path.exists() and i2w_path.exists():
        logger.info("Loading vocabulary from %s and %s", w2i_path, i2w_path)
        w2i = np.load(w2i_path, allow_pickle=True).item()
        i2w = np.load(i2w_path, allow_pickle=True).item()
        return {str(key): int(value) for key, value in w2i.items()}, {
            int(key): str(value) for key, value in i2w.items()
        }

    logger.info("Creating vocabulary at %s", vocab_dir)
    w2i, i2w = make_vocabulary(sequence_groups, sort_tokens=sort_tokens)
    np.save(w2i_path, w2i)
    np.save(i2w_path, i2w)
    logger.info("Vocabulary size: %s", len(w2i))
    return w2i, i2w


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
