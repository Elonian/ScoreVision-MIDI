from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ctc_eval import (
    PROJECT_ROOT,
    default_run_specs,
    evaluate_checkpoint,
    evaluate_saved_predictions,
    write_results_json,
    write_results_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the completed CRNN and CNNT runs with the paper-style CER/SER/LER metrics."
    )
    parser.add_argument(
        "--mode",
        choices=["saved", "checkpoint"],
        default="saved",
        help="'saved' recomputes metrics from existing hyp/gt. 'checkpoint' reruns inference from best.pt.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc. Used in checkpoint mode.")
    parser.add_argument("--num-workers", type=int, default=None, help="Eval workers in checkpoint mode.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test sample limit.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "evaluation" / "runs"),
        help="Checkpoint-mode output root for regenerated hyp/gt.",
    )
    parser.add_argument(
        "--markdown-output",
        default=str(PROJECT_ROOT / "evaluation" / "crnn_cnnt_eval.md"),
        help="Markdown results table path.",
    )
    parser.add_argument(
        "--json-output",
        default=str(PROJECT_ROOT / "evaluation" / "crnn_cnnt_eval.json"),
        help="JSON results path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)
    output_root = Path(args.output_root)
    results = []
    for spec in default_run_specs():
        print(f"Evaluating {spec['model']} with mode={args.mode} split={args.split}", flush=True)
        if args.mode == "saved":
            result = evaluate_saved_predictions(
                run_name=str(spec["run_name"]),
                model=str(spec["model"]),
                split=args.split,
                hyp_dir=Path(spec["hyp_dir"]),
                gt_dir=Path(spec["gt_dir"]),
            )
        else:
            result = evaluate_checkpoint(
                config_path=Path(spec["config"]),
                checkpoint_path=Path(spec["checkpoint"]),
                split=args.split,
                output_dir=output_root / f"{spec['run_name']}_{args.split}",
                device_name=args.device,
                num_workers=args.num_workers,
                max_samples=args.max_samples,
                logger=logger,
            )
        results.append(result)
        print(
            f"{result.model} {result.split}: "
            f"CER={result.cer:.4f} SER={result.ser:.4f} LER={result.ler:.4f} "
            f"samples={result.samples}"
        )

    write_results_json(results, args.json_output)
    write_results_markdown(results, args.markdown_output)
    print(f"Wrote {args.markdown_output}")
    print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
