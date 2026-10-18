"""Command-line interface for the strategy-audit project."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .analysis import audit_strategies
from .report import plot_selection_bias, write_audit_report
from .simulation import run_selection_bias_experiment, simulate_null_returns


def _load_returns(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input CSV was not found: {source}")
    frame = pd.read_csv(source)
    # Date and label columns are allowed; only numeric columns are treated as returns.
    numeric = frame.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if numeric.shape[1] < 2:
        raise ValueError(
            "CSV needs at least two numeric columns of strategy returns. "
            "Use decimal returns (0.01 = 1%), not prices or percentages like 1.0 for 1%."
        )
    return numeric


def _run_audit(returns: pd.DataFrame, output_dir: Path, args: argparse.Namespace, synthetic: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results, pbo = audit_strategies(
        returns,
        periods_per_year=args.periods_per_year,
        pbo_splits=args.splits,
        bootstrap_samples=args.bootstrap,
        bootstrap_block_size=args.block_size,
        seed=args.seed,
    )
    results.to_csv(output_dir / "strategy_audit.csv", index=False)
    pbo["partition_details"].to_csv(output_dir / "pbo_partitions.csv", index=False)
    write_audit_report(results, pbo, output_dir / "audit_report.md", synthetic=synthetic)
    print(f"Audit complete. Results saved to: {output_dir.resolve()}")
    print(f"Candidates: {len(results)} | Observations: {len(returns)} | PBO estimate: {pbo['pbo']:.1%}")
    print(f"Read: {output_dir / 'audit_report.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-audit",
        description="Audit backtest results for selection bias and weak statistical evidence.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run a reproducible synthetic zero-alpha demonstration.")
    demo.add_argument("--output-dir", default="outputs/demo", help="Directory for generated CSV, report, and plot files.")
    demo.add_argument("--seed", type=int, default=2026, help="Random seed for reproducibility.")
    demo.add_argument("--repetitions", type=int, default=100, help="Number of synthetic experiments in the selection-bias chart.")
    demo.add_argument("--bootstrap", type=int, default=250, help="Bootstrap samples per candidate for Sharpe intervals.")
    demo.add_argument("--splits", type=int, default=4, help="CSCV uses 2 * this many contiguous blocks.")
    demo.add_argument("--block-size", type=int, default=10, help="Block length for the circular bootstrap.")
    demo.add_argument("--periods-per-year", type=int, default=252)

    audit = subparsers.add_parser("audit", help="Audit strategy return columns from a CSV file.")
    audit.add_argument("--input", required=True, help="CSV path; numeric columns must be strategy simple returns in decimals.")
    audit.add_argument("--output-dir", default="outputs/audit", help="Directory for audit CSV and Markdown report.")
    audit.add_argument("--seed", type=int, default=2026)
    audit.add_argument("--bootstrap", type=int, default=250)
    audit.add_argument("--splits", type=int, default=4)
    audit.add_argument("--block-size", type=int, default=10)
    audit.add_argument("--periods-per-year", type=int, default=252)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_dir = Path(args.output_dir)
        if args.command == "demo":
            output_dir.mkdir(parents=True, exist_ok=True)
            returns = simulate_null_returns(n_periods=756, n_strategies=30, seed=args.seed)
            returns.insert(0, "date", pd.bdate_range("2020-01-02", periods=len(returns)).strftime("%Y-%m-%d"))
            # The CSV's date field is text so it won't enter the numeric return matrix.
            returns.to_csv(output_dir / "synthetic_returns.csv", index=False)
            returns_numeric = returns.drop(columns=["date"])
            _run_audit(returns_numeric, output_dir, args, synthetic=True)

            selection = run_selection_bias_experiment(repetitions=args.repetitions, seed=args.seed)
            selection.to_csv(output_dir / "selection_bias.csv", index=False)
            plot_selection_bias(selection, output_dir / "selection_bias.png")
            print(f"Selection-bias experiment saved to: {(output_dir / 'selection_bias.csv').resolve()}")
            print(f"Plot saved to: {(output_dir / 'selection_bias.png').resolve()}")
            return 0

        returns = _load_returns(args.input)
        _run_audit(returns, output_dir, args, synthetic=False)
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
