from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.constants import BEKRN_EXTENSION, BREAK_TOKEN, PAD_TOKEN, SPACE_TOKEN, TAB_TOKEN

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

try:
    _PIL_ROTATE_CLOCKWISE = Image.Transpose.ROTATE_270
    _PIL_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _PIL_ROTATE_CLOCKWISE = Image.ROTATE_270
    _PIL_BILINEAR = Image.BILINEAR


def decoder_special_tokens() -> tuple[str, str, str]:
    return BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


def bekern_text_to_decoder_tokens(content: str) -> list[str]:
    content = content.strip("\n ")
    content = re.sub(r"(?<=\=)\d+", "", content)
    content = content.replace(" ", f" {SPACE_TOKEN} ")
    content = content.replace("·", "")
    content = content.replace("\t", f" {TAB_TOKEN} ")
    content = content.replace("\n", f" {BREAK_TOKEN} ")
    return [BOS_TOKEN, *content.split(" "), EOS_TOKEN]


def decoder_tokens_to_kern(tokens: list[str]) -> str:
    filtered = [token for token in tokens if token not in {BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}]
    transcription = "".join(filtered)
    transcription = transcription.replace(TAB_TOKEN, "\t")
    transcription = transcription.replace(BREAK_TOKEN, "\n")
    transcription = transcription.replace(SPACE_TOKEN, " ")
    return transcription


def load_or_create_autoregressive_vocabulary(
    datasets: list["LocalGrandStaffAutoregressiveDataset"],
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
        logger.info("Loading autoregressive vocabulary from %s and %s", w2i_path, i2w_path)
        w2i = np.load(w2i_path, allow_pickle=True).item()
        i2w = np.load(i2w_path, allow_pickle=True).item()
        return {str(key): int(value) for key, value in w2i.items()}, {
            int(key): str(value) for key, value in i2w.items()
        }

    logger.info("Creating autoregressive vocabulary at %s", vocab_dir)
    vocabulary = set()
    for dataset in datasets:
        for index in range(len(dataset)):
            vocabulary.update(dataset.get_tokens(index))
            if (index + 1) % 10000 == 0:
                logger.info("Read autoregressive tokens %s/%s from %s", index + 1, len(dataset), dataset.partition_file)

    symbols = sorted(vocabulary) if sort_tokens else list(vocabulary)
    symbols = [symbol for symbol in symbols if symbol not in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}]
    w2i = {PAD_TOKEN: 0, BOS_TOKEN: 1, EOS_TOKEN: 2}
    for symbol in symbols:
        w2i.setdefault(symbol, len(w2i))
    i2w = {index: symbol for symbol, index in w2i.items()}

    np.save(w2i_path, w2i)
    np.save(i2w_path, i2w)
    logger.info("Autoregressive vocabulary size: %s", len(w2i))
    return w2i, i2w


class LocalGrandStaffAutoregressiveDataset(Dataset):
    def __init__(
        self,
        partition_file: str | Path,
        data_root: str | Path,
        extension: str = BEKRN_EXTENSION,
        image_cache_dir: str | Path | None = None,
        resize_ratio: float = 1.0,
        max_samples: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.partition_file = Path(partition_file)
        self.data_root = Path(data_root)
        self.extension = extension
        self.image_cache_dir = Path(image_cache_dir) if image_cache_dir else None
        self.resize_ratio = float(resize_ratio)
        self.samples = self._load_partition(max_samples=max_samples)
        self._tokens: list[list[str] | None] = [None] * len(self.samples)
        self.w2i: dict[str, int] | None = None
        self.i2w: dict[int, str] | None = None

        if not self.samples:
            raise RuntimeError(f"No samples loaded from partition file: {self.partition_file}")

    def _load_partition(self, max_samples: int | None) -> list[Path]:
        if not self.partition_file.exists():
            raise FileNotFoundError(f"Partition file not found: {self.partition_file}")
        samples = []
        with self.partition_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                relative_path = line.strip()
                if not relative_path:
                    continue
                path = _resolve_transcription_path(relative_path, self.data_root, self.extension)
                samples.append(path)
                if max_samples is not None and len(samples) >= int(max_samples):
                    break
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def set_vocabulary(self, w2i: dict[str, int], i2w: dict[int, str]) -> None:
        self.w2i = w2i
        self.i2w = i2w

    def get_tokens(self, index: int) -> list[str]:
        tokens = self._tokens[index]
        if tokens is None:
            content = self.samples[index].read_text(encoding="utf-8")
            tokens = bekern_text_to_decoder_tokens(content)
            self._tokens[index] = tokens
        return tokens

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.w2i is None:
            raise RuntimeError("Dataset vocabulary is not set. Call set_vocabulary first.")
        transcription_path = self.samples[index]
        image = _image_to_tensor(self._load_image(transcription_path))
        tokens = self.get_tokens(index)
        target = torch.tensor([self.w2i[token] for token in tokens], dtype=torch.long)
        return image, target, len(target)

    def _load_image(self, transcription_path: Path) -> np.ndarray:
        if self.image_cache_dir is not None:
            cache_path = image_cache_path(transcription_path, self.data_root, self.image_cache_dir)
            if cache_path.exists():
                image = np.load(cache_path, allow_pickle=False)
                if self.resize_ratio != 1.0:
                    image = _resize_grayscale(image, self.resize_ratio)
                return image

        image_path = transcription_path.with_suffix(".jpg")
        if not image_path.exists():
            raise FileNotFoundError(f"Could not read image for sample: {image_path}")
        with Image.open(image_path) as pil_image:
            pil_image = pil_image.convert("L")
            image = np.asarray(pil_image, dtype=np.uint8)
        if self.resize_ratio != 1.0:
            image = _resize_grayscale(image, self.resize_ratio)
        return np.asarray(Image.fromarray(image).transpose(_PIL_ROTATE_CLOCKWISE))


def autoregressive_collate(batch: list[tuple[torch.Tensor, torch.Tensor, int]], padding_idx: int):
    images = [sample[0] for sample in batch]
    targets = [sample[1] for sample in batch]
    lengths = torch.tensor([sample[2] for sample in batch], dtype=torch.long)

    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)
    batch_images = torch.ones((len(images), 1, max_height, max_width), dtype=torch.float32)
    for index, image in enumerate(images):
        _, height, width = image.shape
        batch_images[index, :, :height, :width] = image

    max_length = max(target.numel() for target in targets)
    batch_targets = torch.full((len(targets), max_length), int(padding_idx), dtype=torch.long)
    for index, target in enumerate(targets):
        batch_targets[index, : target.numel()] = target

    decoder_input = batch_targets[:, :-1].contiguous()
    labels = batch_targets[:, 1:].contiguous()
    return batch_images, decoder_input, labels, lengths - 1


def image_cache_path(transcription_path: Path, data_root: Path, cache_dir: Path) -> Path:
    transcription_path = Path(transcription_path)
    data_root = Path(data_root)
    if transcription_path.is_relative_to(data_root):
        relative_path = transcription_path.relative_to(data_root)
    else:
        relative_path = Path(transcription_path.name)
    return Path(cache_dir) / relative_path.with_suffix(".npy")


def _resolve_transcription_path(relative_path: str, data_root: Path, extension: str) -> Path:
    normalized = relative_path.replace("\\", "/")
    if normalized.startswith("Data/") or normalized.startswith("data/"):
        normalized = normalized.split("/", 1)[1]
    path = Path(normalized)
    if not path.is_absolute():
        path = data_root / path
    if extension != BEKRN_EXTENSION:
        path = path.with_suffix(extension)
    return path


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image with 2 dimensions, got shape={image.shape}")
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)


def _resize_grayscale(image: np.ndarray, resize_ratio: float) -> np.ndarray:
    width = int(np.ceil(image.shape[1] * resize_ratio))
    height = int(np.ceil(image.shape[0] * resize_ratio))
    resized = Image.fromarray(image).resize((width, height), resample=_PIL_BILINEAR)
    return np.asarray(resized)
