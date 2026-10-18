"""Small reporting helpers for command-line runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_selection_bias(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Save a plot comparing selected in-sample and out-of-sample Sharpe."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(summary["candidate_count"], summary["median_in_sample_sharpe"], marker="o", label="Median selected in-sample Sharpe")
    ax.plot(summary["candidate_count"], summary["median_out_of_sample_sharpe"], marker="o", label="Median selected out-of-sample Sharpe")
    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Number of candidate strategies tested (log scale)")
    ax.set_ylabel("Annualized Sharpe ratio")
    ax.set_title("Selection bias in a zero-alpha simulation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_audit_report(
    results: pd.DataFrame,
    pbo: dict[str, object],
    output_path: str | Path,
    *,
    synthetic: bool = False,
) -> None:
    """Write a concise Markdown summary for a run."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    winner = results.iloc[0]
    if synthetic:
        opening = (
            "This report was produced from **synthetic zero-alpha returns**. It demonstrates "
            "the audit workflow; it is not evidence about real markets or a tradable strategy."
        )
    else:
        opening = (
            "This report summarizes the supplied return columns. Confirm that the input data, "
            "timestamps, return definitions, and costs are appropriate before interpreting results."
        )
    report = f"""# Quant Strategy Audit Report

{opening}

## Dataset summary

- Complete observations used: **{int(results['observations'].max())}**
- Candidate return series: **{len(results)}**
- Highest full-sample annualized Sharpe: **{winner['strategy']}** ({winner['annualized_sharpe']:.3f})

The highest full-sample Sharpe is shown only as a descriptive ranking. Selecting a winner on the same sample used to report its performance creates selection bias.

## Multiple-testing diagnostics

The table reports raw mean-return p-values alongside Holm family-wise adjusted p-values and Benjamini–Hochberg q-values. Newey–West HAC standard errors partially account for serial dependence in the return series; the finite-sample p-value is still an approximation.

The Probabilistic Sharpe Ratio (PSR) estimates the probability that a candidate's true Sharpe exceeds a benchmark under its formula assumptions. The reported PSR benchmark is zero. PSR by itself is **not** a correction for strategy selection.

## Probability of Backtest Overfitting (PBO)

- PBO estimate: **{float(pbo['pbo']):.1%}**
- CSCV partitions evaluated: **{int(pbo['n_partitions'])}**
- Contiguous blocks per in-sample/out-of-sample side: **{int(pbo['n_splits'])}**

PBO is the fraction of combinatorial partitions in which the best in-sample candidate ranks at or below the median across candidates out of sample. It is sensitive to the candidate set, sample length, and partition design; do not interpret it as a guaranteed live-failure probability.

## Recommended interpretation

1. Establish a simple benchmark before optimizing parameters.
2. Keep a final chronological test period untouched during model selection.
3. Include commissions, slippage, funding, and realistic execution assumptions where relevant.
4. Report failed and neutral experiments, not only the best candidate.
5. Re-run the audit with multiple-testing and dependence assumptions documented.

## Limitations

- Return series are treated as periodic **simple returns in decimal form** (0.01 means 1%).
- Sharpe ratios use the chosen `periods_per_year` (default: 252) and a zero risk-free rate unless changed in code.
- Bootstrap intervals are marginal intervals for each supplied series and are not selection-adjusted.
- Historical statistics do not guarantee future performance.
"""
    output_path.write_text(report, encoding="utf-8")
