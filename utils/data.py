from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.constants import (
    BEKRN_EXTENSION,
    DISTORTED_IMAGE_HEIGHT,
    DISTORTED_IMAGE_SUFFIX,
    ENCODER_HEIGHT_REDUCTION,
    ENCODER_WIDTH_REDUCTION,
    JPG_EXTENSION,
    PAD_TOKEN,
)
from utils.transcription import bekern_text_to_tokens


def ctc_collate(data: list[tuple[torch.Tensor, torch.Tensor, int, int]]):
    images = [sample[0] for sample in data]
    targets = [sample[1] for sample in data]
    input_lengths = torch.tensor([sample[2] for sample in data], dtype=torch.long)
    target_lengths = torch.tensor([sample[3] for sample in data], dtype=torch.long)

    max_image_width = max(img.shape[2] for img in images)
    max_image_height = max(img.shape[1] for img in images)
    batch_images = torch.ones(
        size=(len(images), 1, max_image_height, max_image_width),
        dtype=torch.float32,
    )

    for index, image in enumerate(images):
        _, height, width = image.size()
        batch_images[index, :, :height, :width] = image

    max_target_length = max(len(target) for target in targets)
    batch_targets = torch.zeros(size=(len(targets), max_target_length), dtype=torch.long)
    for index, target in enumerate(targets):
        batch_targets[index, : len(target)] = target.long()

    return batch_images, batch_targets, input_lengths, target_lengths


class GrandStaffDataset(Dataset):
    def __init__(
        self,
        partition_file: str | Path,
        data_root: str | Path,
        resize_ratio: float = 1.0,
        load_distorted: bool = False,
        extension: str = BEKRN_EXTENSION,
        max_samples: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.partition_file = Path(partition_file)
        self.data_root = Path(data_root)
        self.resize_ratio = float(resize_ratio)
        self.load_distorted = bool(load_distorted)
        self.extension = extension
        self.x, self.y = load_data(
            partition_file=self.partition_file,
            data_root=self.data_root,
            resize_ratio=self.resize_ratio,
            load_distorted=self.load_distorted,
            extension=self.extension,
            max_samples=max_samples,
            logger=self.logger,
        )
        self.w2i: dict[str, int] | None = None
        self.i2w: dict[int, str] | None = None
        self.padding_token: int | None = None

        if not self.x:
            raise RuntimeError(f"No samples loaded from partition file: {self.partition_file}")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if self.w2i is None:
            raise RuntimeError("Dataset vocabulary is not set. Call set_dictionaries first.")

        image = _image_to_tensor(self.x[index])
        target = torch.tensor([self.w2i[token] for token in self.y[index]], dtype=torch.long)
        input_length = (image.shape[2] // ENCODER_WIDTH_REDUCTION) * (
            image.shape[1] // ENCODER_HEIGHT_REDUCTION
        )
        return image, target, input_length, len(target)

    def get_max_hw(self) -> tuple[int, int]:
        max_width = int(np.max([img.shape[1] for img in self.x]))
        max_height = int(np.max([img.shape[0] for img in self.x]))
        return max_height, max_width

    def get_max_seqlen(self) -> int:
        return int(np.max([len(seq) for seq in self.y]))

    def vocab_size(self) -> int:
        if self.w2i is None:
            raise RuntimeError("Dataset vocabulary is not set. Call set_dictionaries first.")
        return len(self.w2i)

    def get_gt(self) -> list[list[str]]:
        return self.y

    def set_dictionaries(self, w2i: dict[str, int], i2w: dict[int, str]) -> None:
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i[PAD_TOKEN]

    def get_dictionaries(self) -> tuple[dict[str, int], dict[int, str]]:
        if self.w2i is None or self.i2w is None:
            raise RuntimeError("Dataset vocabulary is not set. Call set_dictionaries first.")
        return self.w2i, self.i2w


def load_data(
    partition_file: str | Path,
    data_root: str | Path,
    resize_ratio: float,
    load_distorted: bool = False,
    extension: str = BEKRN_EXTENSION,
    max_samples: int | None = None,
    logger: logging.Logger | None = None,
) -> tuple[list[np.ndarray], list[list[str]]]:
    logger = logger or logging.getLogger(__name__)
    partition_file = Path(partition_file)
    data_root = Path(data_root)

    if not partition_file.exists():
        raise FileNotFoundError(f"Partition file not found: {partition_file}")

    image_samples: list[np.ndarray] = []
    token_samples: list[list[str]] = []
    skipped = 0

    with partition_file.open("r", encoding="utf-8") as partfile:
        part_lines = [line.strip() for line in partfile.read().splitlines() if line.strip()]

    if max_samples is not None:
        part_lines = part_lines[: int(max_samples)]

    logger.info(
        "Loading %s samples from %s (extension=%s, distorted=%s, resize_ratio=%s)",
        len(part_lines),
        partition_file,
        extension,
        load_distorted,
        resize_ratio,
    )

    for index, relative_path in enumerate(part_lines, start=1):
        transcription_path = _resolve_transcription_path(relative_path, data_root, extension)
        if not transcription_path.exists():
            skipped += 1
            logger.warning("Skipping missing transcription: %s", transcription_path)
            continue

        tokens = _read_transcription_tokens(transcription_path)
        image = _load_score_image(
            transcription_path=transcription_path,
            resize_ratio=resize_ratio,
            load_distorted=load_distorted,
            target_length=len(tokens),
            logger=logger,
        )
        if image is None:
            skipped += 1
            continue

        image_samples.append(image)
        token_samples.append(tokens)

        if index % 1000 == 0:
            logger.info("Loaded %s/%s partition entries", index, len(part_lines))

    logger.info("Finished loading %s samples from %s; skipped=%s", len(image_samples), partition_file, skipped)
    return image_samples, token_samples


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


def _read_transcription_tokens(transcription_path: Path) -> list[str]:
    raw_text = transcription_path.read_text(encoding="utf-8")
    return bekern_text_to_tokens(raw_text)


def _load_score_image(
    transcription_path: Path,
    resize_ratio: float,
    load_distorted: bool,
    target_length: int,
    logger: logging.Logger,
) -> np.ndarray | None:
    base_path = transcription_path.with_suffix("")
    image_path = (
        base_path.with_name(f"{base_path.name}{DISTORTED_IMAGE_SUFFIX}{JPG_EXTENSION}")
        if load_distorted
        else base_path.with_suffix(JPG_EXTENSION)
    )

    if not image_path.exists():
        logger.warning("Skipping missing image: %s", image_path)
        return None

    image = _read_grayscale_image(image_path)
    if image is None:
        logger.warning("Skipping unreadable image: %s", image_path)
        return None

    if load_distorted:
        height = DISTORTED_IMAGE_HEIGHT
        width = int(float(height * image.shape[1]) / image.shape[0])
        image = _resize_grayscale(image, width=width, height=height)
        input_length = (height // ENCODER_WIDTH_REDUCTION) * (width // ENCODER_HEIGHT_REDUCTION)
        if input_length <= target_length:
            logger.warning(
                "Skipping sample with short CTC input length: image=%s input_length=%s target_length=%s",
                image_path,
                input_length,
                target_length,
            )
            return None

    width = int(np.ceil(image.shape[1] * resize_ratio))
    height = int(np.ceil(image.shape[0] * resize_ratio))
    image = _resize_grayscale(image, width=width, height=height)
    return _rotate_clockwise(image)


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image with 2 dimensions, got shape={image.shape}")
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0)


def _read_grayscale_image(image_path: Path) -> np.ndarray | None:
    return cv2.imread(str(image_path), 0)


def _resize_grayscale(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _rotate_clockwise(image: np.ndarray) -> np.ndarray:
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
