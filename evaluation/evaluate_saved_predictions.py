from __future__ import annotations

import argparse
from pathlib import Path

from ctc_eval import (
    PROJECT_ROOT,
    default_run_specs,
    evaluate_saved_predictions,
    write_results_json,
    write_results_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute CER/SER/LER from saved CTC hypothesis and ground-truth files."
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Evaluated split name.")
    parser.add_argument(
        "--markdown-output",
        default=str(PROJECT_ROOT / "evaluation" / "ctc_saved_predictions_eval.md"),
        help="Path for the Markdown results table.",
    )
    parser.add_argument(
        "--json-output",
        default=str(PROJECT_ROOT / "evaluation" / "ctc_saved_predictions_eval.json"),
        help="Path for the JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for spec in default_run_specs():
        print(f"Evaluating saved predictions for {spec['model']} split={args.split}", flush=True)
        results.append(
            evaluate_saved_predictions(
                run_name=str(spec["run_name"]),
                model=str(spec["model"]),
                split=args.split,
                hyp_dir=Path(spec["hyp_dir"]),
                gt_dir=Path(spec["gt_dir"]),
            )
        )
    write_results_json(results, args.json_output)
    write_results_markdown(results, args.markdown_output)
    for result in results:
        print(
            f"{result.model} {result.split}: "
            f"CER={result.cer:.4f} SER={result.ser:.4f} LER={result.ler:.4f} "
            f"samples={result.samples}"
        )
    print(f"Wrote {args.markdown_output}")
    print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
