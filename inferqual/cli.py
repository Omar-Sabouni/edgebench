"""Command-line interface for InferQual."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from inferqual.benchmark import BenchmarkError, Settings, run_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inferqual",
        description="Benchmark PyTorch, ONNX Runtime, INT8, and available CUDA backends.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the benchmark and create a report bundle")
    run.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="backend device set (default: auto)",
    )
    run.add_argument(
        "--output",
        type=Path,
        help="new or empty output directory (default: reports/<UTC timestamp>)",
    )
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--iterations", type=int, default=50)
    run.add_argument("--threads", type=int, default=1)

    return parser


def _default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("reports") / stamp


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            output = args.output or _default_output()
            report = run_benchmark(
                output,
                Settings(
                    device=args.device,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    threads=args.threads,
                ),
            )
            winner = next(
                result for result in report["results"] if result["id"] == report["selected"]
            )
            print(f"Selected: {winner['label']} ({winner['latency']['p50_ms']:.3f} ms p50)")
            print(f"JSON:     {output / 'report.json'}")
            print(f"HTML:     {output / 'report.html'}")
            return 0
    except (BenchmarkError, OSError, ValueError) as exc:
        print(f"inferqual: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
