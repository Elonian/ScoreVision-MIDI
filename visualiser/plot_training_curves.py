from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualiser" / "training_curves"


RUNS = {
    "CRNN": {
        "log": PROJECT_ROOT / "logs" / "scorevision_grandstaff_bekrn_crnn_ddp_b12_bucketed_20260530_203444.log",
        "color": "#1f77b4",
        "marker": "o",
    },
    "CNNT": {
        "log": PROJECT_ROOT / "logs" / "scorevision_grandstaff_bekrn_cnnt_ddp_b2_20260530_030814.log",
        "color": "#d62728",
        "marker": "s",
    },
}

EPOCH_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"epoch=(?P<epoch>\d+) train_loss=(?P<train_loss>[0-9.]+) "
    r"val_CER=(?P<val_cer>[0-9.]+) val_SER=(?P<val_ser>[0-9.]+) "
    r"val_LER=(?P<val_ler>[0-9.]+)"
)
EARLY_STOP_RE = re.compile(
    r"Early stopping at epoch=(?P<stop_epoch>\d+); "
    r"best_epoch=(?P<best_epoch>\d+) best_metric=(?P<best_metric>[0-9.]+)"
)


@dataclass(frozen=True)
class EpochPoint:
    model: str
    epoch: int
    train_loss: float
    val_cer: float
    val_ser: float
    val_ler: float
    timestamp: str
    new_best_val_ser: bool
    best_checkpoint_epoch: bool


@dataclass(frozen=True)
class RunSummary:
    model: str
    best_epoch: int
    best_val_ser: float
    stop_epoch: int
    final_epoch: int
    final_train_loss: float
    final_val_cer: float
    final_val_ser: float
    final_val_ler: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CRNN/CNNT training curves from ScoreVision logs.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where plots, CSV, and report are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, list[EpochPoint]] = {}
    summaries: dict[str, RunSummary] = {}
    for model, spec in RUNS.items():
        points, summary = parse_log(model, spec["log"])
        runs[model] = points
        summaries[model] = summary

    fresh_test = load_fresh_test_metrics(PROJECT_ROOT / "evaluation" / "crnn_cnnt_eval.json")
    write_epoch_csv(runs, output_dir / "epoch_progression.csv")
    write_summary_json(summaries, fresh_test, output_dir / "curve_summary.json")

    make_dashboard(runs, summaries, fresh_test, output_dir)
    make_metric_grid(runs, summaries, output_dir)
    make_loss_ser_focus(runs, summaries, fresh_test, output_dir)
    write_report(runs, summaries, fresh_test, output_dir / "training_curves_report.md")

    print(f"Wrote visualisations to {output_dir}")


def parse_log(model: str, log_path: Path) -> tuple[list[EpochPoint], RunSummary]:
    rows = []
    best_seen = float("inf")
    best_epoch_from_stop: int | None = None
    best_metric_from_stop: float | None = None
    stop_epoch: int | None = None

    for line in log_path.read_text(errors="replace").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            data = epoch_match.groupdict()
            epoch = int(data["epoch"])
            val_ser = float(data["val_ser"])
            new_best = val_ser < best_seen
            if new_best:
                best_seen = val_ser
            rows.append(
                EpochPoint(
                    model=model,
                    epoch=epoch,
                    train_loss=float(data["train_loss"]),
                    val_cer=float(data["val_cer"]),
                    val_ser=val_ser,
                    val_ler=float(data["val_ler"]),
                    timestamp=data["timestamp"],
                    new_best_val_ser=new_best,
                    best_checkpoint_epoch=False,
                )
            )
            continue

        stop_match = EARLY_STOP_RE.search(line)
        if stop_match:
            stop_epoch = int(stop_match.group("stop_epoch"))
            best_epoch_from_stop = int(stop_match.group("best_epoch"))
            best_metric_from_stop = float(stop_match.group("best_metric"))

    if not rows:
        raise RuntimeError(f"No epoch rows found in {log_path}")

    best_epoch = best_epoch_from_stop or min(rows, key=lambda row: row.val_ser).epoch
    best_metric = best_metric_from_stop or min(row.val_ser for row in rows)
    rows = [
        EpochPoint(
            model=row.model,
            epoch=row.epoch,
            train_loss=row.train_loss,
            val_cer=row.val_cer,
            val_ser=row.val_ser,
            val_ler=row.val_ler,
            timestamp=row.timestamp,
            new_best_val_ser=row.new_best_val_ser,
            best_checkpoint_epoch=row.epoch == best_epoch,
        )
        for row in rows
    ]
    final = rows[-1]
    return rows, RunSummary(
        model=model,
        best_epoch=best_epoch,
        best_val_ser=best_metric,
        stop_epoch=stop_epoch or final.epoch,
        final_epoch=final.epoch,
        final_train_loss=final.train_loss,
        final_val_cer=final.val_cer,
        final_val_ser=final.val_ser,
        final_val_ler=final.val_ler,
    )


def load_fresh_test_metrics(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    result = {}
    for model, metrics in data.get("fresh_checkpoint_test", {}).items():
        result[model] = {
            "cer": float(metrics["cer"]),
            "ser": float(metrics["ser"]),
            "ler": float(metrics["ler"]),
        }
    return result


def write_epoch_csv(runs: dict[str, list[EpochPoint]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model",
                "epoch",
                "train_loss",
                "val_CER",
                "val_SER",
                "val_LER",
                "new_best_val_SER",
                "best_checkpoint_epoch",
                "timestamp",
            ],
        )
        writer.writeheader()
        for model in sorted(runs):
            for row in runs[model]:
                writer.writerow(
                    {
                        "model": row.model,
                        "epoch": row.epoch,
                        "train_loss": f"{row.train_loss:.6f}",
                        "val_CER": f"{row.val_cer:.4f}",
                        "val_SER": f"{row.val_ser:.4f}",
                        "val_LER": f"{row.val_ler:.4f}",
                        "new_best_val_SER": int(row.new_best_val_ser),
                        "best_checkpoint_epoch": int(row.best_checkpoint_epoch),
                        "timestamp": row.timestamp,
                    }
                )


def write_summary_json(
    summaries: dict[str, RunSummary],
    fresh_test: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    payload = {
        model: {
            "best_epoch": summary.best_epoch,
            "best_val_SER": summary.best_val_ser,
            "stop_epoch": summary.stop_epoch,
            "final_epoch": summary.final_epoch,
            "final_val_CER": summary.final_val_cer,
            "final_val_SER": summary.final_val_ser,
            "final_val_LER": summary.final_val_ler,
            "fresh_test": fresh_test.get(model),
        }
        for model, summary in summaries.items()
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_dashboard(
    runs: dict[str, list[EpochPoint]],
    summaries: dict[str, RunSummary],
    fresh_test: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=160)
    fig.suptitle("ScoreVision MIDI: CRNN vs CNNT Training Curves", fontsize=18, fontweight="bold")

    ax_loss, ax_ser, ax_metrics, ax_notes = axes.ravel()
    plot_train_loss(ax_loss, runs, summaries)
    plot_val_ser(ax_ser, runs, summaries)
    plot_all_val_metrics(ax_metrics, runs, summaries)
    plot_notes(ax_notes, summaries, fresh_test)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_dir / "training_dashboard")


def make_metric_grid(runs: dict[str, list[EpochPoint]], summaries: dict[str, RunSummary], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=160, sharex=False)
    metrics = [
        ("val_cer", "Validation CER", "CER (%)"),
        ("val_ser", "Validation SER", "SER (%)"),
        ("val_ler", "Validation LER", "LER (%)"),
    ]
    for ax, (attr, title, ylabel) in zip(axes, metrics):
        for model, points in runs.items():
            epochs = np.array([row.epoch for row in points])
            values = np.array([getattr(row, attr) for row in points])
            color = RUNS[model]["color"]
            ax.plot(epochs, values, label=model, color=color, linewidth=2.0)
            if attr == "val_ser":
                best = summaries[model]
                best_row = points[best.best_epoch - 1]
                ax.scatter([best.best_epoch], [best.best_val_ser], color=color, edgecolor="black", zorder=4)
                ax.annotate(
                    f"{model} best\nE{best.best_epoch}",
                    xy=(best.best_epoch, best.best_val_ser),
                    xytext=(8, -18 if model == "CRNN" else 12),
                    textcoords="offset points",
                    fontsize=8,
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8},
                )
        style_axis(ax, title, "Epoch", ylabel)
    axes[0].legend(frameon=False)
    fig.suptitle("Validation Metrics By Epoch", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, output_dir / "validation_metrics_comparison")


def make_loss_ser_focus(
    runs: dict[str, list[EpochPoint]],
    summaries: dict[str, RunSummary],
    fresh_test: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(13, 7), dpi=160)
    ax2 = ax1.twinx()

    for model, points in runs.items():
        epochs = np.array([row.epoch for row in points])
        loss = np.array([row.train_loss for row in points])
        ser = np.array([row.val_ser for row in points])
        color = RUNS[model]["color"]
        ax1.plot(epochs, loss, color=color, linestyle="-", linewidth=2.0, alpha=0.45)
        ax2.plot(epochs, ser, color=color, linestyle="--", linewidth=2.3, label=f"{model} Val SER")
        best = summaries[model]
        ax2.axvline(best.best_epoch, color=color, linestyle=":", linewidth=1.4, alpha=0.8)
        ax2.scatter([best.best_epoch], [best.best_val_ser], color=color, edgecolor="black", zorder=5)

    ax1.set_yscale("log")
    ax1.set_ylabel("Training loss, log scale", fontsize=11)
    ax2.set_ylabel("Validation SER (%)", fontsize=11)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.grid(True, color="#d8d8d8", linewidth=0.8, alpha=0.7)
    ax2.legend(frameon=False, loc="upper right")
    ax1.set_title("Loss Falls Quickly; Validation SER Selects The Checkpoint", fontsize=15, fontweight="bold")

    cnnt = summaries["CNNT"]
    crnn = summaries["CRNN"]
    note = (
        "How to read this plot:\n"
        "- Solid faint lines: training loss.\n"
        "- Dashed lines: validation SER, the early-stopping metric.\n"
        f"- CNNT best Val SER {cnnt.best_val_ser:.2f}% at epoch {cnnt.best_epoch}.\n"
        f"- CRNN best Val SER {crnn.best_val_ser:.2f}% at epoch {crnn.best_epoch}.\n"
        "- Later loss improvements are small; checkpoint choice is based on validation SER."
    )
    ax1.text(
        0.03,
        0.28,
        note,
        transform=ax1.transAxes,
        fontsize=9.5,
        va="top",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f6f6f6", "edgecolor": "#bfbfbf"},
    )
    if fresh_test:
        test_note = (
            "Fresh full-test SER:\n"
            f"CNNT {fresh_test['CNNT']['ser']:.2f}%\n"
            f"CRNN {fresh_test['CRNN']['ser']:.2f}%"
        )
        ax1.text(
            0.78,
            0.48,
            test_note,
            transform=ax1.transAxes,
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#ffffff", "edgecolor": "#aaaaaa"},
        )

    fig.tight_layout()
    save_figure(fig, output_dir / "loss_vs_validation_ser")


def plot_train_loss(ax: plt.Axes, runs: dict[str, list[EpochPoint]], summaries: dict[str, RunSummary]) -> None:
    for model, points in runs.items():
        epochs = np.array([row.epoch for row in points])
        values = np.array([row.train_loss for row in points])
        color = RUNS[model]["color"]
        ax.plot(epochs, values, label=model, color=color, linewidth=2.1)
        ax.plot(epochs, moving_average(values, 5), color=color, linewidth=1.4, alpha=0.45)
    ax.set_yscale("log")
    style_axis(ax, "Training Loss", "Epoch", "Loss, log scale")
    ax.legend(frameon=False)
    ax.text(
        0.04,
        0.08,
        "Both models learn the easy structure early; later epochs refine notation details.",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ffffff", "edgecolor": "#bbbbbb"},
    )


def plot_val_ser(ax: plt.Axes, runs: dict[str, list[EpochPoint]], summaries: dict[str, RunSummary]) -> None:
    max_epoch = max(row.epoch for points in runs.values() for row in points)
    summary_lines = []
    for model, points in runs.items():
        epochs = np.array([row.epoch for row in points])
        values = np.array([row.val_ser for row in points])
        color = RUNS[model]["color"]
        ax.plot(epochs, values, label=model, color=color, linewidth=2.3)
        best = summaries[model]
        ax.scatter([best.best_epoch], [best.best_val_ser], color=color, edgecolor="black", zorder=5)
        ax.axvline(best.stop_epoch, color=color, linestyle=":", linewidth=1.0, alpha=0.5)
        summary_lines.append(f"{model}: best SER {best.best_val_ser:.2f}% at epoch {best.best_epoch}")
    style_axis(ax, "Validation SER And Early Stopping", "Epoch", "SER (%)")
    ax.set_xlim(0, max_epoch + 4)
    ax.legend(frameon=False)
    ax.text(
        0.58,
        0.28,
        "\n".join(summary_lines) + "\nDotted lines mark early stop.",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#ffffff", "edgecolor": "#bbbbbb"},
    )


def plot_all_val_metrics(ax: plt.Axes, runs: dict[str, list[EpochPoint]], summaries: dict[str, RunSummary]) -> None:
    styles = {
        "val_cer": ("CER", "-"),
        "val_ser": ("SER", "--"),
        "val_ler": ("LER", ":"),
    }
    for model, points in runs.items():
        epochs = np.array([row.epoch for row in points])
        color = RUNS[model]["color"]
        for attr, (label, linestyle) in styles.items():
            values = np.array([getattr(row, attr) for row in points])
            ax.plot(epochs, values, color=color, linestyle=linestyle, linewidth=1.7, label=f"{model} {label}")
    style_axis(ax, "All Validation Metrics", "Epoch", "Error rate (%)")
    ax.legend(frameon=False, fontsize=8, ncol=2)


def plot_notes(ax: plt.Axes, summaries: dict[str, RunSummary], fresh_test: dict[str, dict[str, float]]) -> None:
    ax.axis("off")
    cnnt = summaries["CNNT"]
    crnn = summaries["CRNN"]
    ser_gain = crnn.best_val_ser - cnnt.best_val_ser
    test_ser_gain = fresh_test.get("CRNN", {}).get("ser", 0.0) - fresh_test.get("CNNT", {}).get("ser", 0.0)
    notes = [
        ("Main reading", "Validation curves, not training loss alone, choose the checkpoint."),
        ("Best validation SER", f"CNNT {cnnt.best_val_ser:.2f}% at epoch {cnnt.best_epoch}; CRNN {crnn.best_val_ser:.2f}% at epoch {crnn.best_epoch}."),
        ("Validation gap", f"CNNT improves best Val SER by {ser_gain:.2f} percentage points over CRNN."),
        ("Fresh test gap", f"CNNT improves full-test SER by {test_ser_gain:.2f} percentage points over CRNN."),
        ("Curve shape", "Fast early gains, then slower refinement; early stopping catches the plateau."),
    ]
    y = 0.95
    ax.text(0.02, y, "Presentation Notes", fontsize=15, fontweight="bold", transform=ax.transAxes)
    y -= 0.12
    for title, body in notes:
        ax.text(0.03, y, title, fontsize=10.5, fontweight="bold", transform=ax.transAxes)
        ax.text(0.03, y - 0.045, body, fontsize=10, transform=ax.transAxes, wrap=True)
        y -= 0.16


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_report(
    runs: dict[str, list[EpochPoint]],
    summaries: dict[str, RunSummary],
    fresh_test: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    lines = [
        "# Training Curve Summary",
        "",
        "This report explains the CRNN/CNNT training curves generated from the real training logs.",
        "",
        "## Outputs",
        "",
        "| File | Purpose |",
        "| --- | --- |",
        "| `training_dashboard.png` / `.svg` | Four-panel presentation dashboard. |",
        "| `loss_vs_validation_ser.png` / `.svg` | Focused plot linking loss to the early-stopping metric. |",
        "| `validation_metrics_comparison.png` / `.svg` | CER/SER/LER comparison by epoch. |",
        "| `epoch_progression.csv` | Every-epoch numeric values for CRNN and CNNT. |",
        "| `curve_summary.json` | Best epochs, final metrics, and fresh test metrics. |",
        "",
        "## Key Reading",
        "",
        "| Model | Epochs | Best epoch | Best Val SER | Final Val SER | Fresh Test SER |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in ["CRNN", "CNNT"]:
        summary = summaries[model]
        test_ser = fresh_test.get(model, {}).get("ser", float("nan"))
        lines.append(
            f"| {model} | {summary.final_epoch} | {summary.best_epoch} | "
            f"{summary.best_val_ser:.4f} | {summary.final_val_ser:.4f} | {test_ser:.4f} |"
        )
    lines.extend(
        [
            "",
            "CNNT has the better validation and fresh full-test SER. The loss curves keep moving late in training, but the validation curves show the useful plateau and justify early stopping.",
            "",
            "## Theory Notes For Presentation",
            "",
            "- Training loss measures how well the model fits batches seen during optimization.",
            "- Validation CER/SER/LER measure recognition quality on held-out pages.",
            "- SER is the early-stopping metric in these runs, so the selected checkpoint is the epoch with the best validation SER before patience runs out.",
            "- The gap between CRNN and CNNT is clearest on SER/LER: CNNT's transformer unfolding decoder handles long page-level dependencies better than the recurrent decoder in this run.",
            "- The fresh full-test numbers are computed after reloading `best.pt` and evaluating the full test split with `batch_size=1`.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
