from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ctc_eval import evaluate_checkpoint, write_results_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one ScoreVision CTC checkpoint and write hyp/gt files plus CER/SER/LER."
    )
    parser.add_argument("--config", required=True, help="CRNN/CNNT YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path, usually weights/best.pt.")
    parser.add_argument("--output-dir", required=True, help="Directory where hyp/gt and metrics.json are written.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--batch-size", type=int, default=None, help="Eval batch size. Defaults to paper-style 1.")
    parser.add_argument("--num-workers", type=int, default=None, help="Eval workers. Defaults to config value.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test sample limit.")
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Optional Markdown table path for this single result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    result = evaluate_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        logger=logging.getLogger(__name__),
    )
    if args.markdown_output:
        write_results_markdown([result], Path(args.markdown_output))
    print(
        f"{result.model} {result.split}: "
        f"CER={result.cer:.4f} SER={result.ser:.4f} LER={result.ler:.4f} "
        f"samples={result.samples} hyp={result.hyp_dir} gt={result.gt_dir}"
    )


if __name__ == "__main__":
    main()
