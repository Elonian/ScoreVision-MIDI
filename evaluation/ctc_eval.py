from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_config, resolve_path
from paper_metrics import get_metrics

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from utils.data import GrandStaffDataset


METRIC_DEFINITIONS = {
    "CER": "Character Error Rate: normalized Levenshtein distance after character-level parsing.",
    "SER": "Symbol Error Rate: normalized Levenshtein distance after token/symbol parsing.",
    "LER": "Line Error Rate: normalized Levenshtein distance after line-level parsing.",
}


@dataclass
class EvalResult:
    model: str
    run_name: str
    source: str
    split: str
    samples: int
    cer: float
    ser: float
    ler: float
    checkpoint: str | None
    checkpoint_epoch: int | None
    checkpoint_best_epoch: int | None
    checkpoint_best_metric: float | None
    hyp_dir: str
    gt_dir: str
    metrics_json: str | None = None
    elapsed_seconds: float | None = None


def evaluate_saved_predictions(
    run_name: str,
    hyp_dir: str | Path,
    gt_dir: str | Path,
    split: str = "test",
    model: str | None = None,
) -> EvalResult:
    hyp_dir = Path(hyp_dir)
    gt_dir = Path(gt_dir)
    hyp_files = _files_by_name(hyp_dir)
    gt_files = _files_by_name(gt_dir)
    _validate_matching_prediction_files(hyp_files, gt_files, hyp_dir, gt_dir)

    hypotheses = [path.read_text(encoding="utf-8") for _, path in sorted(hyp_files.items())]
    references = [path.read_text(encoding="utf-8") for _, path in sorted(gt_files.items())]
    cer, ser, ler = get_metrics(hypotheses, references)
    inferred_model = model or _infer_model_name(run_name)
    return EvalResult(
        model=inferred_model,
        run_name=run_name,
        source="saved_predictions",
        split=split,
        samples=len(hypotheses),
        cer=cer,
        ser=ser,
        ler=ler,
        checkpoint=None,
        checkpoint_epoch=None,
        checkpoint_best_epoch=None,
        checkpoint_best_metric=None,
        hyp_dir=str(hyp_dir),
        gt_dir=str(gt_dir),
    )


def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    split: str,
    output_dir: str | Path,
    device_name: str = "auto",
    batch_size: int | None = None,
    num_workers: int | None = None,
    max_samples: int | None = None,
    clean_output: bool = True,
    logger: logging.Logger | None = None,
) -> EvalResult:
    logger = logger or logging.getLogger(__name__)
    import torch
    from torch.utils.data import DataLoader

    from models.score_unfolding import build_model
    from utils.vocabulary import load_or_create_vocabulary

    started_at = time.perf_counter()
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    config = load_yaml_config(config_path)
    if max_samples is not None:
        config["data"]["max_samples"] = int(max_samples)

    device = resolve_eval_device(device_name)
    run_name = str(config["project"]["run_name"])
    model_name = str(config["model"]["name"]).upper()
    logger.info("Evaluating %s on %s with checkpoint %s", run_name, device, checkpoint_path)

    test_dataset = build_eval_dataset(config=config, split=split, project_root=PROJECT_ROOT, logger=logger)
    vocab_dir = resolve_path(config["vocab"]["directory"], PROJECT_ROOT)
    w2i, i2w = load_or_create_vocabulary(
        None,
        vocab_dir=vocab_dir,
        name=str(config["vocab"]["name"]),
        sort_tokens=bool(config["vocab"].get("sort_tokens", False)),
        logger=logger,
    )
    test_dataset.set_dictionaries(w2i, i2w)

    max_height, max_width = infer_model_hw(config, model_name, test_dataset)
    blank_idx = len(i2w)
    out_size = len(w2i) + 1
    model = build_model(
        model_name=model_name,
        max_width=max_width,
        max_height=max_height,
        in_channels=int(config["model"]["in_channels"]),
        out_size=out_size,
        dropout=float(config["model"].get("dropout", 0.4)),
        max_len=config["model"].get("max_len"),
        pretrain_path=config["model"].get("pretrain_path"),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(_strip_module_prefix(state_dict), strict=True)
    model.to(device)
    model.eval()

    eval_batch_size = int(batch_size or 1)
    eval_num_workers = int(num_workers if num_workers is not None else config["training"].get("num_workers", 0))
    from utils.data import ctc_collate

    loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        collate_fn=ctc_collate,
        persistent_workers=eval_num_workers > 0,
        pin_memory=device.type == "cuda",
    )

    if clean_output:
        shutil.rmtree(output_dir / "hyp", ignore_errors=True)
        shutil.rmtree(output_dir / "gt", ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cer, ser, ler = evaluate_ctc_model(
        model=model,
        loader=loader,
        i2w=i2w,
        blank_idx=blank_idx,
        device=device,
        output_dir=output_dir,
        write_predictions=True,
    )
    elapsed_seconds = time.perf_counter() - started_at
    result = EvalResult(
        model=model_name,
        run_name=run_name,
        source="checkpoint_inference",
        split=split,
        samples=len(test_dataset),
        cer=cer,
        ser=ser,
        ler=ler,
        checkpoint=str(checkpoint_path),
        checkpoint_epoch=_optional_int(checkpoint.get("epoch")) if isinstance(checkpoint, dict) else None,
        checkpoint_best_epoch=_optional_int(checkpoint.get("best_epoch")) if isinstance(checkpoint, dict) else None,
        checkpoint_best_metric=_optional_float(checkpoint.get("best_metric")) if isinstance(checkpoint, dict) else None,
        hyp_dir=str(output_dir / "hyp"),
        gt_dir=str(output_dir / "gt"),
        elapsed_seconds=elapsed_seconds,
    )
    metrics_json = output_dir / "metrics.json"
    write_results_json([result], metrics_json)
    result.metrics_json = str(metrics_json)
    logger.info(
        "%s %s: CER=%.4f SER=%.4f LER=%.4f samples=%s elapsed=%.1fs",
        model_name,
        split,
        cer,
        ser,
        ler,
        len(test_dataset),
        elapsed_seconds,
    )
    return result


def evaluate_ctc_model(
    model: "torch.nn.Module",
    loader: "DataLoader",
    i2w: dict[int, str],
    blank_idx: int,
    device: "torch.device",
    output_dir: str | Path | None = None,
    write_predictions: bool = False,
) -> tuple[float, float, float]:
    """Paper-style CTC test: greedy decode, write hyp/gt files, compute CER/SER/LER."""
    import torch

    from utils.decoding import greedy_decode_ctc
    from utils.transcription import tokens_to_kern

    model.eval()
    hypotheses: list[str] = []
    references: list[str] = []
    output_dir = Path(output_dir) if output_dir is not None else None
    hyp_dir = output_dir / "hyp" if output_dir else None
    gt_dir = output_dir / "gt" if output_dir else None

    if write_predictions and hyp_dir is not None and gt_dir is not None:
        hyp_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images, targets, input_lengths, target_lengths = batch
            images = images.to(device)
            targets = targets.to(device)
            predictions = model(images)
            decoded, target_tokens = greedy_decode_ctc(
                predictions,
                targets,
                input_lengths.cpu(),
                target_lengths.cpu(),
                i2w,
                blank_idx,
            )

            for sample_idx, (decoded_tokens, gt_tokens) in enumerate(zip(decoded, target_tokens)):
                hypothesis = tokens_to_kern(decoded_tokens)
                reference = tokens_to_kern(gt_tokens)
                hypotheses.append(hypothesis)
                references.append(reference)

                if write_predictions and hyp_dir is not None and gt_dir is not None:
                    filename = f"{batch_idx}.krn" if len(decoded) == 1 else f"{batch_idx}_{sample_idx}.krn"
                    (hyp_dir / filename).write_text(hypothesis, encoding="utf-8")
                    (gt_dir / filename).write_text(reference, encoding="utf-8")

    return get_metrics(hypotheses, references)


def build_eval_dataset(
    config: dict[str, Any],
    split: str,
    project_root: Path,
    logger: logging.Logger,
) -> "GrandStaffDataset":
    from utils.data import GrandStaffDataset

    data_cfg = config["data"]
    split_to_key = {
        "train": "train_partition",
        "val": "val_partition",
        "validation": "val_partition",
        "test": "test_partition",
    }
    if split not in split_to_key:
        raise ValueError(f"Unsupported split '{split}'. Choose train, val, or test.")
    return GrandStaffDataset(
        partition_file=resolve_path(data_cfg[split_to_key[split]], project_root),
        data_root=resolve_path(data_cfg["data_root"], project_root),
        resize_ratio=float(data_cfg["resize_ratio"]),
        load_distorted=bool(data_cfg["load_distorted"]),
        extension=data_cfg["extension"],
        max_samples=data_cfg.get("max_samples"),
        preload_images=bool(data_cfg.get("preload_images", False)),
        image_cache_dir=resolve_path(data_cfg.get("image_cache_dir"), project_root),
        metadata_cache_dir=resolve_path(data_cfg.get("metadata_cache_dir"), project_root),
        logger=logger,
    )


def infer_model_hw(config: dict[str, Any], model_name: str, dataset: GrandStaffDataset) -> tuple[int, int]:
    needs_max_hw = model_name.upper() in {"CNNT", "STAVE_CNNT"}
    configured_max_height = config["data"].get("max_height")
    configured_max_width = config["data"].get("max_width")
    if needs_max_hw and configured_max_height is not None and configured_max_width is not None:
        return int(configured_max_height), int(configured_max_width)
    if needs_max_hw:
        return dataset.get_max_hw()
    return dataset.get_sample_hw(0)


def resolve_eval_device(device_name: str) -> "torch.device":
    import torch

    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is not available.")
    return device


def write_results_json(results: list[EvalResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_results_markdown(results: list[EvalResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "| Model | Source | Split | Samples | CER | SER | LER | Checkpoint epoch | Hyp dir | GT dir |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for result in results:
        checkpoint_epoch = "" if result.checkpoint_epoch is None else str(result.checkpoint_epoch)
        rows.append(
            "| {model} | {source} | {split} | {samples} | {cer:.4f} | {ser:.4f} | {ler:.4f} | "
            "{checkpoint_epoch} | `{hyp_dir}` | `{gt_dir}` |".format(
                model=result.model,
                source=result.source,
                split=result.split,
                samples=result.samples,
                cer=result.cer,
                ser=result.ser,
                ler=result.ler,
                checkpoint_epoch=checkpoint_epoch,
                hyp_dir=result.hyp_dir,
                gt_dir=result.gt_dir,
            )
        )

    metric_lines = "\n".join(f"- `{name}`: {definition}" for name, definition in METRIC_DEFINITIONS.items())
    content = (
        "# ScoreVision MIDI CTC Evaluation\n\n"
        "Dataset: GrandStaff `test.txt` unless the split column says otherwise. "
        "The metrics are computed with the same CER/SER/LER parser used by the official IJDAR code. "
        "Checkpoint mode uses the paper-style test batch size of 1 by default, which writes "
        "`hyp/0.krn`, `hyp/1.krn`, ... and matching `gt` files.\n\n"
        + "\n".join(rows)
        + "\n\n## Metric Definitions\n\n"
        + metric_lines
        + "\n"
    )
    path.write_text(content, encoding="utf-8")


def default_run_specs(project_root: Path = PROJECT_ROOT) -> list[dict[str, Path | str]]:
    output_root = project_root / "outputs"
    return [
        {
            "model": "CRNN",
            "run_name": "scorevision_grandstaff_bekrn_crnn_ddp_b12_bucketed",
            "config": project_root / "configs" / "score_unfolding.yaml",
            "checkpoint": output_root
            / "scorevision_grandstaff_bekrn_crnn_ddp_b12_bucketed"
            / "weights"
            / "best.pt",
            "hyp_dir": output_root / "scorevision_grandstaff_bekrn_crnn_ddp_b12_bucketed" / "hyp",
            "gt_dir": output_root / "scorevision_grandstaff_bekrn_crnn_ddp_b12_bucketed" / "gt",
        },
        {
            "model": "CNNT",
            "run_name": "scorevision_grandstaff_bekrn_cnnt_ddp_b2",
            "config": project_root / "configs" / "score_unfolding_cnnt.yaml",
            "checkpoint": output_root / "scorevision_grandstaff_bekrn_cnnt_ddp_b2" / "weights" / "best.pt",
            "hyp_dir": output_root / "scorevision_grandstaff_bekrn_cnnt_ddp_b2" / "hyp",
            "gt_dir": output_root / "scorevision_grandstaff_bekrn_cnnt_ddp_b2" / "gt",
        },
    ]


def _files_by_name(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Prediction directory not found: {directory}")
    return {path.name: path for path in directory.iterdir() if path.is_file()}


def _validate_matching_prediction_files(
    hyp_files: dict[str, Path],
    gt_files: dict[str, Path],
    hyp_dir: Path,
    gt_dir: Path,
) -> None:
    missing_gt = sorted(set(hyp_files) - set(gt_files))
    missing_hyp = sorted(set(gt_files) - set(hyp_files))
    if missing_gt or missing_hyp:
        details = []
        if missing_gt:
            details.append(f"{len(missing_gt)} hyp files have no GT match in {gt_dir}")
        if missing_hyp:
            details.append(f"{len(missing_hyp)} GT files have no hyp match in {hyp_dir}")
        raise RuntimeError("; ".join(details))


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _infer_model_name(run_name: str) -> str:
    upper = run_name.upper()
    if "CNNT" in upper:
        return "CNNT"
    if "CRNN" in upper:
        return "CRNN"
    if "FCN" in upper:
        return "FCN"
    return "UNKNOWN"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
