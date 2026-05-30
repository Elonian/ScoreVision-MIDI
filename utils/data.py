from __future__ import annotations

import logging
import math
import random
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

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
        preload_images: bool = False,
        image_cache_dir: str | Path | None = None,
        metadata_cache_dir: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.partition_file = Path(partition_file)
        self.data_root = Path(data_root)
        self.resize_ratio = float(resize_ratio)
        self.load_distorted = bool(load_distorted)
        self.extension = extension
        self.preload_images = bool(preload_images)
        self.image_cache_dir = Path(image_cache_dir) if image_cache_dir else None
        self.metadata_cache_dir = Path(metadata_cache_dir) if metadata_cache_dir else None
        self.max_samples = max_samples
        self.samples: list[tuple[Path, list[str] | None]] = []
        self.x: list[np.ndarray] = []
        self.y: list[list[str] | None] = []
        self._sample_hw_cache: list[tuple[int, int]] | None = None

        if self.preload_images:
            self.x, self.y = load_data(
                partition_file=self.partition_file,
                data_root=self.data_root,
                resize_ratio=self.resize_ratio,
                load_distorted=self.load_distorted,
                extension=self.extension,
                max_samples=max_samples,
                logger=self.logger,
            )
        else:
            self.samples = load_transcription_samples(
                partition_file=self.partition_file,
                data_root=self.data_root,
                extension=self.extension,
                max_samples=max_samples,
                read_tokens=False,
                logger=self.logger,
            )
            self.y = [tokens for _, tokens in self.samples]

        self.w2i: dict[str, int] | None = None
        self.i2w: dict[int, str] | None = None
        self.padding_token: int | None = None

        if len(self) == 0:
            raise RuntimeError(f"No samples loaded from partition file: {self.partition_file}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if self.w2i is None:
            raise RuntimeError("Dataset vocabulary is not set. Call set_dictionaries first.")

        tokens = self._get_tokens(index)
        if self.preload_images:
            image_array = self.x[index]
        else:
            transcription_path, _ = self.samples[index]
            image_array = self._load_image_for_sample(transcription_path, len(tokens))

        image = _image_to_tensor(image_array)
        target = torch.tensor([self.w2i[token] for token in tokens], dtype=torch.long)
        input_length = (image.shape[2] // ENCODER_WIDTH_REDUCTION) * (
            image.shape[1] // ENCODER_HEIGHT_REDUCTION
        )
        return image, target, input_length, len(target)

    def get_max_hw(self) -> tuple[int, int]:
        if self.preload_images:
            max_width = int(np.max([img.shape[1] for img in self.x]))
            max_height = int(np.max([img.shape[0] for img in self.x]))
            return max_height, max_width

        sample_hw_cache = self._ensure_sample_hw_cache()
        max_height = max(height for height, _ in sample_hw_cache)
        max_width = max(width for _, width in sample_hw_cache)
        return max_height, max_width

    def get_sample_hw(self, index: int = 0) -> tuple[int, int]:
        if self._sample_hw_cache is not None:
            return self._sample_hw_cache[index]
        return self._read_sample_hw(index)

    def get_sample_heights(self) -> list[int]:
        return [height for height, _ in self._ensure_sample_hw_cache()]

    def _ensure_sample_hw_cache(self) -> list[tuple[int, int]]:
        if self._sample_hw_cache is None:
            cached_hw = self._load_hw_cache()
            if cached_hw is not None:
                self._sample_hw_cache = cached_hw
                return self._sample_hw_cache

            total = len(self)
            self.logger.info(
                "Reading cached image shapes for %s samples from %s",
                total,
                self.partition_file,
            )
            sample_hw_cache = []
            for index in range(total):
                sample_hw_cache.append(self._read_sample_hw(index))
                if (index + 1) % 10000 == 0:
                    self.logger.info(
                        "Read cached image shapes %s/%s from %s",
                        index + 1,
                        total,
                        self.partition_file,
                    )
            self._sample_hw_cache = sample_hw_cache
            self._save_hw_cache(sample_hw_cache)
        return self._sample_hw_cache

    def _read_sample_hw(self, index: int) -> tuple[int, int]:
        if self.preload_images:
            image = self.x[index]
            return int(image.shape[0]), int(image.shape[1])

        transcription_path, _ = self.samples[index]
        if self.image_cache_dir is not None:
            cache_path = image_cache_path(
                transcription_path=transcription_path,
                data_root=self.data_root,
                cache_dir=self.image_cache_dir,
            )
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Cached image not found: {cache_path}. "
                    "Run scripts/prepare_image_cache.py before training or unset data.image_cache_dir."
                )
            image = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            return int(image.shape[0]), int(image.shape[1])

        tokens = self._get_tokens(index)
        image = self._load_image_for_sample(transcription_path, len(tokens))
        return int(image.shape[0]), int(image.shape[1])

    def _get_tokens(self, index: int) -> list[str]:
        tokens = self.y[index]
        if tokens is None:
            transcription_path, _ = self.samples[index]
            tokens = _read_transcription_tokens(transcription_path)
            self.y[index] = tokens
            self.samples[index] = (transcription_path, tokens)
        return tokens

    def _ensure_all_tokens(self) -> list[list[str]]:
        total = len(self)
        token_samples = []
        for index in range(total):
            token_samples.append(self._get_tokens(index))
            if (index + 1) % 10000 == 0:
                self.logger.info(
                    "Loaded tokens %s/%s from %s",
                    index + 1,
                    total,
                    self.partition_file,
                )
        return token_samples

    def _hw_cache_path(self) -> Path | None:
        if self.metadata_cache_dir is None:
            return None
        cache_parts = [
            str(self.partition_file.resolve()),
            str(self.data_root.resolve()),
            str(self.image_cache_dir.resolve() if self.image_cache_dir else None),
            self.extension,
            str(self.resize_ratio),
            str(self.load_distorted),
            str(self.max_samples),
        ]
        digest = hashlib.sha1("|".join(cache_parts).encode("utf-8")).hexdigest()[:16]
        return self.metadata_cache_dir / f"{self.partition_file.stem}_{digest}_hw.npy"

    def hw_cache_path(self) -> Path | None:
        return self._hw_cache_path()

    def _load_hw_cache(self) -> list[tuple[int, int]] | None:
        cache_path = self._hw_cache_path()
        if cache_path is None or not cache_path.exists():
            return None
        hw_array = np.load(cache_path, allow_pickle=False)
        if int(hw_array.shape[0]) != len(self):
            self.logger.warning("Ignoring stale image-shape cache: %s", cache_path)
            return None
        self.logger.info("Loaded image-shape cache from %s", cache_path)
        return [(int(height), int(width)) for height, width in hw_array.tolist()]

    def _save_hw_cache(self, sample_hw_cache: list[tuple[int, int]]) -> None:
        cache_path = self._hw_cache_path()
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, np.asarray(sample_hw_cache, dtype=np.int32), allow_pickle=False)
        self.logger.info("Saved image-shape cache to %s", cache_path)

    def _load_image_for_sample(self, transcription_path: Path, target_length: int) -> np.ndarray:
        if self.image_cache_dir is not None:
            cache_path = image_cache_path(
                transcription_path=transcription_path,
                data_root=self.data_root,
                cache_dir=self.image_cache_dir,
            )
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Cached image not found: {cache_path}. "
                    "Run scripts/prepare_image_cache.py before training or unset data.image_cache_dir."
                )
            return np.load(cache_path, allow_pickle=False)

        image = _load_score_image(
            transcription_path=transcription_path,
            resize_ratio=self.resize_ratio,
            load_distorted=self.load_distorted,
            target_length=target_length,
            logger=self.logger,
        )
        if image is None:
            raise RuntimeError(f"Could not load image for sample: {transcription_path}")
        return image

    def get_max_seqlen(self) -> int:
        return int(np.max([len(seq) for seq in self._ensure_all_tokens()]))

    def vocab_size(self) -> int:
        if self.w2i is None:
            raise RuntimeError("Dataset vocabulary is not set. Call set_dictionaries first.")
        return len(self.w2i)

    def get_gt(self) -> list[list[str]]:
        return self._ensure_all_tokens()

    def set_dictionaries(self, w2i: dict[str, int], i2w: dict[int, str]) -> None:
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i[PAD_TOKEN]

    def get_dictionaries(self) -> tuple[dict[str, int], dict[int, str]]:
        if self.w2i is None or self.i2w is None:
            raise RuntimeError("Dataset vocabulary is not set. Call set_dictionaries first.")
        return self.w2i, self.i2w


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.num_samples = self._num_samples()

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)

        if self.drop_last:
            total_size = (len(indices) // self.world_size) * self.world_size
            indices = indices[:total_size]
        else:
            total_size = self.num_samples * self.world_size
            padding = total_size - len(indices)
            if padding > 0:
                indices += indices[:padding]

        rank_indices = indices[self.rank:total_size:self.world_size]
        rank_indices.sort(key=lambda index: (self.lengths[index], index))
        batches = [
            rank_indices[start : start + self.batch_size]
            for start in range(0, len(rank_indices), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        return iter(batches)

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _num_samples(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.world_size
        return math.ceil(len(self.lengths) / self.world_size)


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


def load_transcription_samples(
    partition_file: str | Path,
    data_root: str | Path,
    extension: str = BEKRN_EXTENSION,
    max_samples: int | None = None,
    read_tokens: bool = True,
    validate_files: bool | None = None,
    logger: logging.Logger | None = None,
) -> list[tuple[Path, list[str] | None]]:
    logger = logger or logging.getLogger(__name__)
    partition_file = Path(partition_file)
    data_root = Path(data_root)
    validate_files = read_tokens if validate_files is None else bool(validate_files)

    if not partition_file.exists():
        raise FileNotFoundError(f"Partition file not found: {partition_file}")

    with partition_file.open("r", encoding="utf-8") as partfile:
        part_lines = [line.strip() for line in partfile.read().splitlines() if line.strip()]

    if max_samples is not None:
        part_lines = part_lines[: int(max_samples)]

    action = "Loading" if read_tokens else "Indexing"
    logger.info(
        "%s %s transcriptions from %s (extension=%s, lazy_images=True, lazy_tokens=%s)",
        action,
        len(part_lines),
        partition_file,
        extension,
        not read_tokens,
    )

    samples: list[tuple[Path, list[str] | None]] = []
    skipped = 0
    for index, relative_path in enumerate(part_lines, start=1):
        transcription_path = _resolve_transcription_path(relative_path, data_root, extension)
        if validate_files and not transcription_path.exists():
            skipped += 1
            logger.warning("Skipping missing transcription: %s", transcription_path)
            continue

        tokens = _read_transcription_tokens(transcription_path) if read_tokens else None
        samples.append((transcription_path, tokens))

        if read_tokens and index % 10000 == 0:
            logger.info("Loaded %s/%s transcriptions", index, len(part_lines))

    logger.info(
        "Finished %s %s transcriptions from %s; skipped=%s",
        action.lower(),
        len(samples),
        partition_file,
        skipped,
    )
    return samples


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
