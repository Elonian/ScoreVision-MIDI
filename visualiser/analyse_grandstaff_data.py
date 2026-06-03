from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from numpy.lib import format as np_format

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.constants import (  # noqa: E402
    BEKRN_EXTENSION,
    BREAK_TOKEN,
    DISTORTED_IMAGE_SUFFIX,
    ENCODER_HEIGHT_REDUCTION,
    ENCODER_WIDTH_REDUCTION,
    JPG_EXTENSION,
    SPACE_TOKEN,
    TAB_TOKEN,
)
from utils.data import _resolve_transcription_path, image_cache_path  # noqa: E402
from utils.transcription import bekern_text_to_tokens  # noqa: E402


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs" / "image_cache" / "grandstaff_bekrn_plain"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualiser" / "data_preprocessing"
PARTITIONS = {
    "train": PROJECT_ROOT / "data" / "grandstaff_dataset" / "partitions" / "train.txt",
    "val": PROJECT_ROOT / "data" / "grandstaff_dataset" / "partitions" / "val.txt",
    "test": PROJECT_ROOT / "data" / "grandstaff_dataset" / "partitions" / "test.txt",
}
SPLIT_ORDER = ["train", "val", "test"]
SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}
SPLIT_COLORS = {"train": "#2f6f9f", "val": "#d98b2b", "test": "#4f8f46"}


@dataclass(frozen=True)
class SampleRecord:
    split: str
    relative_path: str
    transcription_path: Path
    raw_image_path: Path
    distorted_image_path: Path
    cache_path: Path
    composer: str
    transcription_exists: bool
    raw_image_exists: bool
    distorted_image_exists: bool
    cache_exists: bool
    token_count: int | None
    line_break_count: int | None
    character_count: int | None
    cache_height: int | None
    cache_width: int | None
    ctc_input_length: int | None
    ctc_margin: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GrandStaff data preprocessing statistics and slide visuals."
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Root that contains grandstaff_dataset.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Training image cache directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where plots, CSV, JSON, and Markdown are written.",
    )
    parser.add_argument(
        "--plot-sample-limit",
        type=int,
        default=5000,
        help="Maximum points per split for scatter plots only. Exact tables use all samples.",
    )
    parser.add_argument(
        "--top-tokens",
        type=int,
        default=25,
        help="Number of frequent tokens shown in the top-token plot and report table.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel file readers for exact transcription/cache statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_split: dict[str, list[SampleRecord]] = {}
    token_counts_by_split: dict[str, Counter[str]] = {}
    for split in SPLIT_ORDER:
        records, token_counts = analyse_split(
            split=split,
            partition_file=PARTITIONS[split],
            data_root=data_root,
            cache_dir=cache_dir,
            workers=max(1, int(args.workers)),
        )
        records_by_split[split] = records
        token_counts_by_split[split] = token_counts

    all_records = [record for split in SPLIT_ORDER for record in records_by_split[split]]
    overall_token_counts = Counter()
    for counter in token_counts_by_split.values():
        overall_token_counts.update(counter)

    split_stats = {
        split: build_split_stats(records_by_split[split], token_counts_by_split[split])
        for split in SPLIT_ORDER
    }
    total_stats = build_split_stats(all_records, overall_token_counts)
    composer_counts = build_composer_counts(records_by_split)
    examples = choose_examples(records_by_split)

    write_split_statistics_csv(split_stats, total_stats, output_dir / "split_statistics.csv")
    write_top_tokens_csv(
        overall_token_counts,
        token_counts_by_split,
        output_dir / "top_tokens.csv",
    )
    write_composer_counts_csv(composer_counts, output_dir / "composer_counts.csv")
    write_sample_manifest_csv(examples, output_dir / "preprocessing_sample_manifest.csv")
    write_summary_json(
        output_path=output_dir / "dataset_summary.json",
        data_root=data_root,
        cache_dir=cache_dir,
        output_dir=output_dir,
        split_stats=split_stats,
        total_stats=total_stats,
        composer_counts=composer_counts,
        token_counts=overall_token_counts,
        token_counts_by_split=token_counts_by_split,
        top_tokens=args.top_tokens,
    )

    make_split_count_plot(split_stats, total_stats, output_dir)
    make_token_length_plot(records_by_split, output_dir)
    make_image_shape_plot(records_by_split, int(args.plot_sample_limit), output_dir)
    make_top_token_plot(overall_token_counts, int(args.top_tokens), output_dir)
    make_composer_plot(composer_counts, output_dir)
    make_preprocessing_examples_plot(examples, output_dir)
    make_pipeline_plot(output_dir)
    make_dashboard(split_stats, total_stats, overall_token_counts, records_by_split, output_dir)

    write_report(
        output_path=output_dir / "data_preprocessing_report.md",
        data_root=data_root,
        cache_dir=cache_dir,
        split_stats=split_stats,
        total_stats=total_stats,
        composer_counts=composer_counts,
        token_counts=overall_token_counts,
        token_counts_by_split=token_counts_by_split,
        top_tokens=args.top_tokens,
    )

    print(f"Wrote GrandStaff data preprocessing outputs to {output_dir}")


def analyse_split(
    split: str,
    partition_file: Path,
    data_root: Path,
    cache_dir: Path,
    workers: int,
) -> tuple[list[SampleRecord], Counter[str]]:
    entries = read_partition_entries(partition_file)
    records: list[SampleRecord] = []
    token_counts: Counter[str] = Counter()

    print(
        f"Analysing {split}: {len(entries)} partition entries with {workers} workers",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(analyse_entry, split, relative_path, data_root, cache_dir)
            for relative_path in entries
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            record, tokens = future.result()
            records.append(record)
            token_counts.update(tokens)
            if index % 5000 == 0 or index == len(entries):
                print(f"  {split}: processed {index}/{len(entries)}", flush=True)

    records.sort(key=lambda record: record.relative_path)
    return records, token_counts


def analyse_entry(
    split: str,
    relative_path: str,
    data_root: Path,
    cache_dir: Path,
) -> tuple[SampleRecord, list[str]]:
    transcription_path = _resolve_transcription_path(relative_path, data_root, BEKRN_EXTENSION)
    base_path = transcription_path.with_suffix("")
    raw_image_path = base_path.with_suffix(JPG_EXTENSION)
    distorted_image_path = base_path.with_name(
        f"{base_path.name}{DISTORTED_IMAGE_SUFFIX}{JPG_EXTENSION}"
    )
    cache_path = image_cache_path(transcription_path, data_root, cache_dir)

    transcription_exists = transcription_path.exists()
    tokens: list[str] = []
    token_count: int | None = None
    line_break_count: int | None = None
    character_count: int | None = None
    if transcription_exists:
        content = transcription_path.read_text(encoding="utf-8", errors="replace")
        tokens = bekern_text_to_tokens(content)
        token_count = len(tokens)
        line_break_count = tokens.count(BREAK_TOKEN)
        character_count = len(content)

    cache_shape = read_cached_image_shape(cache_path)
    cache_height: int | None = None
    cache_width: int | None = None
    ctc_input_length: int | None = None
    ctc_margin: int | None = None
    if cache_shape is not None:
        cache_height, cache_width = cache_shape
        ctc_input_length = (cache_width // ENCODER_WIDTH_REDUCTION) * (
            cache_height // ENCODER_HEIGHT_REDUCTION
        )
        if token_count is not None:
            ctc_margin = ctc_input_length - token_count

    return (
        SampleRecord(
            split=split,
            relative_path=relative_path,
            transcription_path=transcription_path,
            raw_image_path=raw_image_path,
            distorted_image_path=distorted_image_path,
            cache_path=cache_path,
            composer=extract_composer(relative_path),
            transcription_exists=transcription_exists,
            raw_image_exists=raw_image_path.exists(),
            distorted_image_exists=distorted_image_path.exists(),
            cache_exists=cache_path.exists(),
            token_count=token_count,
            line_break_count=line_break_count,
            character_count=character_count,
            cache_height=cache_height,
            cache_width=cache_width,
            ctc_input_length=ctc_input_length,
            ctc_margin=ctc_margin,
        ),
        tokens,
    )


def read_partition_entries(partition_file: Path) -> list[str]:
    return [
        line.strip()
        for line in partition_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_cached_image_shape(cache_path: Path) -> tuple[int, int] | None:
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as handle:
        version = np_format.read_magic(handle)
        if version == (1, 0):
            shape, _, _ = np_format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _, _ = np_format.read_array_header_2_0(handle)
        else:
            shape, _, _ = np_format._read_array_header(handle, version)
    if len(shape) < 2:
        return None
    return int(shape[0]), int(shape[1])


def extract_composer(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) >= 3 and parts[0] == "grandstaff_dataset" and parts[1] == "grandstaff":
        return parts[2]
    return "unknown"


def build_split_stats(records: list[SampleRecord], token_counts: Counter[str]) -> dict[str, Any]:
    token_lengths = [record.token_count for record in records if record.token_count is not None]
    line_counts = [record.line_break_count for record in records if record.line_break_count is not None]
    character_counts = [
        record.character_count for record in records if record.character_count is not None
    ]
    cache_heights = [record.cache_height for record in records if record.cache_height is not None]
    cache_widths = [record.cache_width for record in records if record.cache_width is not None]
    ctc_lengths = [
        record.ctc_input_length for record in records if record.ctc_input_length is not None
    ]
    ctc_margins = [record.ctc_margin for record in records if record.ctc_margin is not None]

    return {
        "samples": len(records),
        "transcriptions": sum(record.transcription_exists for record in records),
        "clean_jpg": sum(record.raw_image_exists for record in records),
        "distorted_jpg": sum(record.distorted_image_exists for record in records),
        "cached_npy": sum(record.cache_exists for record in records),
        "missing_transcriptions": sum(not record.transcription_exists for record in records),
        "missing_clean_jpg": sum(not record.raw_image_exists for record in records),
        "missing_cache": sum(not record.cache_exists for record in records),
        "vocab_size": len(token_counts),
        "total_tokens": int(sum(token_counts.values())),
        "token_length": numeric_summary(token_lengths),
        "line_breaks": numeric_summary(line_counts),
        "characters": numeric_summary(character_counts),
        "cache_height": numeric_summary(cache_heights),
        "cache_width": numeric_summary(cache_widths),
        "ctc_input_length": numeric_summary(ctc_lengths),
        "ctc_margin": numeric_summary(ctc_margins),
    }


def numeric_summary(values: list[int | float | None]) -> dict[str, Any]:
    clean_values = [float(value) for value in values if value is not None]
    if not clean_values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(clean_values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": round_number(np.min(array)),
        "p25": round_number(np.percentile(array, 25)),
        "median": round_number(np.percentile(array, 50)),
        "mean": round_number(np.mean(array)),
        "p75": round_number(np.percentile(array, 75)),
        "p95": round_number(np.percentile(array, 95)),
        "max": round_number(np.max(array)),
    }


def round_number(value: float) -> int | float:
    if float(value).is_integer():
        return int(value)
    return round(float(value), 2)


def build_composer_counts(
    records_by_split: dict[str, list[SampleRecord]]
) -> dict[str, dict[str, int]]:
    composers = sorted(
        {
            record.composer
            for records in records_by_split.values()
            for record in records
        }
    )
    result = {composer: {split: 0 for split in SPLIT_ORDER} for composer in composers}
    for split, records in records_by_split.items():
        counts = Counter(record.composer for record in records)
        for composer, count in counts.items():
            result[composer][split] = int(count)
    return result


def choose_examples(
    records_by_split: dict[str, list[SampleRecord]]
) -> list[SampleRecord]:
    examples: list[SampleRecord] = []
    for split in SPLIT_ORDER:
        candidates = [
            record
            for record in records_by_split[split]
            if record.raw_image_exists and record.cache_exists and record.token_count is not None
        ]
        if not candidates:
            continue
        median_tokens = float(np.median([record.token_count for record in candidates]))
        chosen = min(
            candidates,
            key=lambda record: (abs(float(record.token_count) - median_tokens), record.relative_path),
        )
        examples.append(chosen)
    return examples


def write_split_statistics_csv(
    split_stats: dict[str, dict[str, Any]],
    total_stats: dict[str, Any],
    output_path: Path,
) -> None:
    fields = [
        "split",
        "samples",
        "transcriptions",
        "clean_jpg",
        "distorted_jpg",
        "cached_npy",
        "vocab_size",
        "total_tokens",
        "token_median",
        "token_mean",
        "token_p95",
        "token_max",
        "height_median",
        "height_p95",
        "height_max",
        "width_median",
        "width_p95",
        "width_max",
        "ctc_margin_min",
        "ctc_margin_median",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in [*SPLIT_ORDER, "total"]:
            stats = total_stats if split == "total" else split_stats[split]
            writer.writerow(
                {
                    "split": split,
                    "samples": stats["samples"],
                    "transcriptions": stats["transcriptions"],
                    "clean_jpg": stats["clean_jpg"],
                    "distorted_jpg": stats["distorted_jpg"],
                    "cached_npy": stats["cached_npy"],
                    "vocab_size": stats["vocab_size"],
                    "total_tokens": stats["total_tokens"],
                    "token_median": stats["token_length"]["median"],
                    "token_mean": stats["token_length"]["mean"],
                    "token_p95": stats["token_length"]["p95"],
                    "token_max": stats["token_length"]["max"],
                    "height_median": stats["cache_height"]["median"],
                    "height_p95": stats["cache_height"]["p95"],
                    "height_max": stats["cache_height"]["max"],
                    "width_median": stats["cache_width"]["median"],
                    "width_p95": stats["cache_width"]["p95"],
                    "width_max": stats["cache_width"]["max"],
                    "ctc_margin_min": stats["ctc_margin"]["min"],
                    "ctc_margin_median": stats["ctc_margin"]["median"],
                }
            )


def write_top_tokens_csv(
    token_counts: Counter[str],
    token_counts_by_split: dict[str, Counter[str]],
    output_path: Path,
) -> None:
    total = max(sum(token_counts.values()), 1)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "token",
                "display_token",
                "count",
                "percent",
                "train_count",
                "val_count",
                "test_count",
            ],
        )
        writer.writeheader()
        for rank, (token, count) in enumerate(token_counts.most_common(), start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "token": token,
                    "display_token": display_token(token),
                    "count": int(count),
                    "percent": f"{(count / total) * 100:.4f}",
                    "train_count": int(token_counts_by_split["train"][token]),
                    "val_count": int(token_counts_by_split["val"][token]),
                    "test_count": int(token_counts_by_split["test"][token]),
                }
            )


def write_composer_counts_csv(
    composer_counts: dict[str, dict[str, int]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["composer", *SPLIT_ORDER, "total"])
        writer.writeheader()
        for composer, counts in sorted(composer_counts.items()):
            writer.writerow(
                {
                    "composer": composer,
                    **{split: counts[split] for split in SPLIT_ORDER},
                    "total": sum(counts.values()),
                }
            )


def write_sample_manifest_csv(examples: list[SampleRecord], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "relative_path",
                "composer",
                "tokens",
                "cache_height",
                "cache_width",
                "raw_image",
                "cached_image",
            ],
        )
        writer.writeheader()
        for record in examples:
            writer.writerow(
                {
                    "split": record.split,
                    "relative_path": record.relative_path,
                    "composer": record.composer,
                    "tokens": record.token_count,
                    "cache_height": record.cache_height,
                    "cache_width": record.cache_width,
                    "raw_image": relative_to_project(record.raw_image_path),
                    "cached_image": relative_to_project(record.cache_path),
                }
            )


def write_summary_json(
    output_path: Path,
    data_root: Path,
    cache_dir: Path,
    output_dir: Path,
    split_stats: dict[str, dict[str, Any]],
    total_stats: dict[str, Any],
    composer_counts: dict[str, dict[str, int]],
    token_counts: Counter[str],
    token_counts_by_split: dict[str, Counter[str]],
    top_tokens: int,
) -> None:
    payload = {
        "data_root": str(data_root),
        "partition_files": {split: str(path) for split, path in PARTITIONS.items()},
        "image_cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "preprocessing": {
            "transcription_extension": BEKRN_EXTENSION,
            "image_source": "clean .jpg files",
            "load_distorted": False,
            "resize_ratio": 1.0,
            "grayscale": True,
            "rotation": "90 degrees clockwise",
            "tensor_normalization": "float32 image / 255.0",
            "ctc_input_length": (
                f"(cached_width // {ENCODER_WIDTH_REDUCTION}) * "
                f"(cached_height // {ENCODER_HEIGHT_REDUCTION})"
            ),
            "tokens": {
                "space": SPACE_TOKEN,
                "tab": TAB_TOKEN,
                "line_break": BREAK_TOKEN,
            },
        },
        "split_statistics": split_stats,
        "total_statistics": total_stats,
        "composer_counts": composer_counts,
        "top_tokens": [
            {
                "rank": rank,
                "token": token,
                "display_token": display_token(token),
                "count": int(count),
                "train_count": int(token_counts_by_split["train"][token]),
                "val_count": int(token_counts_by_split["val"][token]),
                "test_count": int(token_counts_by_split["test"][token]),
            }
            for rank, (token, count) in enumerate(token_counts.most_common(top_tokens), start=1)
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_split_count_plot(
    split_stats: dict[str, dict[str, Any]],
    total_stats: dict[str, Any],
    output_dir: Path,
) -> None:
    labels = [SPLIT_LABELS[split] for split in SPLIT_ORDER]
    values = [split_stats[split]["samples"] for split in SPLIT_ORDER]
    total = max(total_stats["samples"], 1)

    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160)
    bars = ax.bar(labels, values, color=[SPLIT_COLORS[split] for split in SPLIT_ORDER], width=0.6)
    ax.set_title("GrandStaff Split Counts", fontsize=16, fontweight="bold")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n{(value / total) * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    save_figure(fig, output_dir / "dataset_split_counts")


def make_token_length_plot(
    records_by_split: dict[str, list[SampleRecord]],
    output_dir: Path,
) -> None:
    values_by_split = [
        [record.token_count for record in records_by_split[split] if record.token_count is not None]
        for split in SPLIT_ORDER
    ]
    all_values = [value for values in values_by_split for value in values]
    histogram_limit = float(np.percentile(all_values, 99)) if all_values else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=160)
    ax_box, ax_hist = axes

    box = ax_box.boxplot(
        values_by_split,
        tick_labels=[SPLIT_LABELS[split] for split in SPLIT_ORDER],
        patch_artist=True,
        showfliers=False,
    )
    for patch, split in zip(box["boxes"], SPLIT_ORDER):
        patch.set_facecolor(SPLIT_COLORS[split])
        patch.set_alpha(0.75)
    ax_box.set_title("Target Sequence Lengths", fontweight="bold")
    ax_box.set_ylabel("BEKRN tokens per sample")
    ax_box.grid(axis="y", alpha=0.25)

    bins = np.linspace(0, histogram_limit, 50)
    for split, values in zip(SPLIT_ORDER, values_by_split):
        clipped = np.clip(np.asarray(values, dtype=np.float64), 0, histogram_limit)
        ax_hist.hist(
            clipped,
            bins=bins,
            density=True,
            alpha=0.35,
            color=SPLIT_COLORS[split],
            label=SPLIT_LABELS[split],
        )
    ax_hist.set_title("Length Distribution", fontweight="bold")
    ax_hist.set_xlabel("Tokens per sample (clipped at p99)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(frameon=False)
    ax_hist.grid(axis="y", alpha=0.25)

    fig.suptitle("GrandStaff BEKRN Target Lengths", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir / "token_length_distribution")


def make_image_shape_plot(
    records_by_split: dict[str, list[SampleRecord]],
    plot_sample_limit: int,
    output_dir: Path,
) -> None:
    heights_by_split = [
        [record.cache_height for record in records_by_split[split] if record.cache_height is not None]
        for split in SPLIT_ORDER
    ]
    widths_by_split = [
        [record.cache_width for record in records_by_split[split] if record.cache_width is not None]
        for split in SPLIT_ORDER
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), dpi=160)
    ax_h, ax_w, ax_scatter = axes
    plot_box(ax_h, heights_by_split, "Cached Height After Rotation", "Pixels")
    plot_box(ax_w, widths_by_split, "Cached Width After Rotation", "Pixels")

    for split in SPLIT_ORDER:
        records = [
            record
            for record in records_by_split[split]
            if record.cache_width is not None and record.cache_height is not None
        ]
        sampled = deterministic_sample(records, max(0, plot_sample_limit))
        ax_scatter.scatter(
            [record.cache_width for record in sampled],
            [record.cache_height for record in sampled],
            s=8,
            alpha=0.25,
            color=SPLIT_COLORS[split],
            label=SPLIT_LABELS[split],
            linewidths=0,
        )
    ax_scatter.set_title("Shape Scatter", fontweight="bold")
    ax_scatter.set_xlabel("Cached width")
    ax_scatter.set_ylabel("Cached height")
    ax_scatter.legend(frameon=False, markerscale=2)
    ax_scatter.grid(alpha=0.2)

    fig.suptitle(
        "Preprocessed Image Shapes Used by CRNN/CNNT",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir / "preprocessed_image_shapes")


def plot_box(ax: plt.Axes, values_by_split: list[list[int]], title: str, ylabel: str) -> None:
    box = ax.boxplot(
        values_by_split,
        tick_labels=[SPLIT_LABELS[split] for split in SPLIT_ORDER],
        patch_artist=True,
        showfliers=False,
    )
    for patch, split in zip(box["boxes"], SPLIT_ORDER):
        patch.set_facecolor(SPLIT_COLORS[split])
        patch.set_alpha(0.75)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def deterministic_sample(records: list[SampleRecord], limit: int) -> list[SampleRecord]:
    if limit <= 0 or len(records) <= limit:
        return records
    indices = np.linspace(0, len(records) - 1, limit, dtype=int)
    return [records[int(index)] for index in indices]


def make_top_token_plot(
    token_counts: Counter[str],
    top_tokens: int,
    output_dir: Path,
) -> None:
    rows = token_counts.most_common(top_tokens)
    labels = [display_token(token) for token, _ in rows][::-1]
    values = [count for _, count in rows][::-1]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    ax.barh(labels, values, color="#527a8f")
    ax.set_title(f"Top {len(rows)} BEKRN/CTC Tokens", fontsize=16, fontweight="bold")
    ax.set_xlabel("Token frequency")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output_dir / "top_bekrn_tokens")


def make_composer_plot(
    composer_counts: dict[str, dict[str, int]],
    output_dir: Path,
) -> None:
    composers = sorted(
        composer_counts,
        key=lambda composer: sum(composer_counts[composer].values()),
        reverse=True,
    )
    bottoms = np.zeros(len(composers), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    x = np.arange(len(composers))
    for split in SPLIT_ORDER:
        values = np.asarray([composer_counts[composer][split] for composer in composers])
        ax.bar(
            x,
            values,
            bottom=bottoms,
            color=SPLIT_COLORS[split],
            label=SPLIT_LABELS[split],
        )
        bottoms += values
    ax.set_title("Composer Coverage by Split", fontsize=16, fontweight="bold")
    ax.set_ylabel("Samples")
    ax.set_xticks(x)
    ax.set_xticklabels(composers, rotation=25, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_dir / "composer_distribution")


def make_preprocessing_examples_plot(
    examples: list[SampleRecord],
    output_dir: Path,
) -> None:
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 2, figsize=(13, 4 * len(examples)), dpi=160)
    if len(examples) == 1:
        axes = np.asarray([axes])

    for row_index, record in enumerate(examples):
        raw_image = cv2.imread(str(record.raw_image_path), 0)
        cached_image = np.load(record.cache_path, allow_pickle=False)

        ax_raw, ax_cached = axes[row_index]
        ax_raw.imshow(raw_image, cmap="gray", aspect="auto")
        ax_raw.set_axis_off()
        ax_raw.set_title(
            f"{SPLIT_LABELS[record.split]} raw .jpg\n"
            f"{record.composer} | {raw_image.shape[0]}x{raw_image.shape[1]}",
            fontsize=10,
        )

        ax_cached.imshow(cached_image, cmap="gray", aspect="auto")
        ax_cached.set_axis_off()
        ax_cached.set_title(
            "Cached model input\n"
            f"{record.cache_height}x{record.cache_width} | {record.token_count} tokens",
            fontsize=10,
        )

    fig.suptitle("Raw Score Image to Cached Model Input", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_dir / "preprocessing_examples", svg=False)


def make_pipeline_plot(output_dir: Path) -> None:
    steps = [
        ("Partition", "train/val/test txt\nrelative .bekrn paths"),
        ("Transcription", ".bekrn text\nUTF-8 read"),
        ("Tokenization", f"space={SPACE_TOKEN}\ntab={TAB_TOKEN}\nline={BREAK_TOKEN}"),
        ("Image", "clean .jpg\ngrayscale read"),
        ("Cache", "resize ratio 1.0\nrotate clockwise\nsave .npy"),
        ("Batch", "normalize /255\npad images and targets\nCTC lengths"),
    ]

    fig, ax = plt.subplots(figsize=(15, 4), dpi=160)
    ax.set_axis_off()
    x_positions = np.linspace(0.09, 0.91, len(steps))
    for index, ((title, body), x_pos) in enumerate(zip(steps, x_positions)):
        box = FancyBboxPatch(
            (x_pos - 0.07, 0.34),
            0.14,
            0.34,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.2,
            edgecolor="#385568",
            facecolor="#f4f7f9",
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(
            x_pos,
            0.59,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#203340",
        )
        ax.text(
            x_pos,
            0.45,
            body,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="#203340",
        )
        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 0.08, 0.51),
                xytext=(x_pos + 0.08, 0.51),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#385568"},
            )
    ax.set_title("GrandStaff Preprocessing Pipeline", fontsize=16, fontweight="bold", y=0.92)
    save_figure(fig, output_dir / "preprocessing_pipeline")


def make_dashboard(
    split_stats: dict[str, dict[str, Any]],
    total_stats: dict[str, Any],
    token_counts: Counter[str],
    records_by_split: dict[str, list[SampleRecord]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=160)
    ax_counts, ax_tokens, ax_shapes, ax_notes = axes.ravel()

    labels = [SPLIT_LABELS[split] for split in SPLIT_ORDER]
    values = [split_stats[split]["samples"] for split in SPLIT_ORDER]
    ax_counts.bar(labels, values, color=[SPLIT_COLORS[split] for split in SPLIT_ORDER])
    ax_counts.set_title("Split Sizes", fontweight="bold")
    ax_counts.set_ylabel("Samples")
    ax_counts.grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        ax_counts.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=9)

    token_lengths = [
        [record.token_count for record in records_by_split[split] if record.token_count is not None]
        for split in SPLIT_ORDER
    ]
    box = ax_tokens.boxplot(
        token_lengths,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
    )
    for patch, split in zip(box["boxes"], SPLIT_ORDER):
        patch.set_facecolor(SPLIT_COLORS[split])
        patch.set_alpha(0.75)
    ax_tokens.set_title("Target Lengths", fontweight="bold")
    ax_tokens.set_ylabel("BEKRN tokens")
    ax_tokens.grid(axis="y", alpha=0.25)

    top_rows = token_counts.most_common(10)
    ax_shapes.barh(
        [display_token(token) for token, _ in top_rows][::-1],
        [count for _, count in top_rows][::-1],
        color="#527a8f",
    )
    ax_shapes.set_title("Most Frequent Tokens", fontweight="bold")
    ax_shapes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax_shapes.grid(axis="x", alpha=0.25)

    ax_notes.axis("off")
    notes = [
        f"Total samples: {total_stats['samples']:,}",
        f"Vocabulary size: {total_stats['vocab_size']:,}",
        f"Total target tokens: {total_stats['total_tokens']:,}",
        f"Cached images: {total_stats['cached_npy']:,}",
        "Image source: clean .jpg, not distorted",
        "Preprocess: grayscale, resize 1.0, rotate clockwise",
        f"CTC input length: W//{ENCODER_WIDTH_REDUCTION} * H//{ENCODER_HEIGHT_REDUCTION}",
    ]
    ax_notes.text(
        0.04,
        0.96,
        "Preprocessing Facts",
        transform=ax_notes.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )
    ax_notes.text(
        0.04,
        0.84,
        "\n".join(notes),
        transform=ax_notes.transAxes,
        fontsize=11,
        va="top",
        linespacing=1.7,
    )

    fig.suptitle("GrandStaff Data and Preprocessing Summary", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_dir / "data_preprocessing_dashboard")


def write_report(
    output_path: Path,
    data_root: Path,
    cache_dir: Path,
    split_stats: dict[str, dict[str, Any]],
    total_stats: dict[str, Any],
    composer_counts: dict[str, dict[str, int]],
    token_counts: Counter[str],
    token_counts_by_split: dict[str, Counter[str]],
    top_tokens: int,
) -> None:
    lines = [
        "# GrandStaff Data Preprocessing and Statistics",
        "",
        "This report was generated from the local GrandStaff dataset used by the CRNN and CNNT runs.",
        "",
        "## Data Source",
        "",
        f"- Data root: `{relative_to_project(data_root)}`",
        "- Partitions: `data/grandstaff_dataset/partitions/train.txt`, `val.txt`, `test.txt`",
        f"- Training image cache: `{relative_to_project(cache_dir)}`",
        f"- Total partition entries: `{total_stats['samples']:,}`",
        f"- Cached model-input images found: `{total_stats['cached_npy']:,}`",
        "",
        "## Preprocessing Pipeline",
        "",
        "1. Read each partition file and resolve the relative path to a `.bekrn` transcription.",
        "2. Convert BEKRN text to CTC target tokens with the same tokenizer used by training.",
        f"3. Encode spaces as `{SPACE_TOKEN}`, tabs as `{TAB_TOKEN}`, and line breaks as `{BREAK_TOKEN}`.",
        "4. Read the clean `.jpg` score image as grayscale. The reported CRNN/CNNT configs set `load_distorted: false`.",
        "5. Resize with ratio `1.0`, rotate the image 90 degrees clockwise, and store it as `.npy` cache.",
        "6. During training/evaluation, convert to float tensor with `image / 255.0`, pad each batch, and use CTC loss/decoding.",
        f"7. CTC input length is `(cached_width // {ENCODER_WIDTH_REDUCTION}) * (cached_height // {ENCODER_HEIGHT_REDUCTION})`.",
        "",
        "## Split Statistics",
        "",
        "| Split | Samples | Clean JPG | Distorted JPG Present | Cached NPY | Vocab | Token median | Token p95 | Token max | Height median | Width median | Min CTC margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for split in [*SPLIT_ORDER, "total"]:
        stats = total_stats if split == "total" else split_stats[split]
        label = "Total" if split == "total" else SPLIT_LABELS[split]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    fmt_int(stats["samples"]),
                    fmt_int(stats["clean_jpg"]),
                    fmt_int(stats["distorted_jpg"]),
                    fmt_int(stats["cached_npy"]),
                    fmt_int(stats["vocab_size"]),
                    fmt_value(stats["token_length"]["median"]),
                    fmt_value(stats["token_length"]["p95"]),
                    fmt_value(stats["token_length"]["max"]),
                    fmt_value(stats["cache_height"]["median"]),
                    fmt_value(stats["cache_width"]["median"]),
                    fmt_value(stats["ctc_margin"]["min"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Composer Coverage",
            "",
            "| Composer | Train | Validation | Test | Total |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for composer, counts in sorted(
        composer_counts.items(),
        key=lambda item: sum(item[1].values()),
        reverse=True,
    ):
        lines.append(
            f"| {composer} | {fmt_int(counts['train'])} | {fmt_int(counts['val'])} | "
            f"{fmt_int(counts['test'])} | {fmt_int(sum(counts.values()))} |"
        )

    total_tokens = max(sum(token_counts.values()), 1)
    lines.extend(
        [
            "",
            f"## Top {top_tokens} Tokens",
            "",
            "| Rank | Token | Count | Percent | Train | Validation | Test |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, (token, count) in enumerate(token_counts.most_common(top_tokens), start=1):
        lines.append(
            f"| {rank} | `{display_token(token)}` | {fmt_int(count)} | "
            f"{(count / total_tokens) * 100:.2f}% | "
            f"{fmt_int(token_counts_by_split['train'][token])} | "
            f"{fmt_int(token_counts_by_split['val'][token])} | "
            f"{fmt_int(token_counts_by_split['test'][token])} |"
        )

    lines.extend(
        [
            "",
            "## Slide-Ready Outputs",
            "",
            "- `data_preprocessing_dashboard.png`: compact summary for one slide.",
            "- `dataset_split_counts.png`: train/validation/test counts.",
            "- `token_length_distribution.png`: target sequence length distribution.",
            "- `preprocessed_image_shapes.png`: exact cached image shape statistics with sampled scatter.",
            "- `top_bekrn_tokens.png`: most frequent tokens.",
            "- `composer_distribution.png`: composer composition by split.",
            "- `preprocessing_examples.png`: raw score image beside cached model input.",
            "- `preprocessing_pipeline.png`: pipeline diagram.",
            "- `split_statistics.csv`, `top_tokens.csv`, `composer_counts.csv`: tables for slides/report.",
            "",
            "## Presentation Notes",
            "",
            "- The model task is image-conditioned symbolic transcription: score image in, BEKRN/KERN token sequence out.",
            "- All dataset statistics above use the exact local partition files, not the partial CameraPrIMuS folder.",
            "- The CRNN/CNNT runs used clean GrandStaff images. Distorted image files exist locally, but the active configs disabled them.",
            "- The cache count equals the partition total, so the image-shape statistics describe every model input sample.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, output_base: Path, svg: bool = True) -> None:
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    if svg:
        fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def display_token(token: str) -> str:
    if token == "":
        return "<empty>"
    if token == " ":
        return "<space>"
    return token


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fmt_int(value: int | float | None) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def fmt_value(value: int | float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.2f}"
    return f"{int(value):,}"


if __name__ == "__main__":
    main()
