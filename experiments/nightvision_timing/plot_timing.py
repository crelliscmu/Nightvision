#!/usr/bin/env python3
"""Plot the mean latency at each prompt length from one timing results file.

Reads a results/timing_results_prefill/*_timing*.json written by collect_prefill.py (the most recently
modified one by default), drops the top and bottom 10% of latencies within
each measurement, averages the rest, merging repeated measurements of the same
prompt length, and plots the means against prompt length with standard-error
bars. Writes <results file stem>.png next to
the results file unless --out is given.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "timing_results_prefill"
TRIM_FRACTION = 0.1  # fraction of latencies dropped from each end of a measurement


def latest_results() -> Path:
    """Most recently modified timing results file in RESULTS_DIR."""
    files = sorted(RESULTS_DIR.glob("*_timing*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no *_timing*.json files in {RESULTS_DIR}")
    return files[-1]


def trimmed(latencies: list[float]) -> np.ndarray:
    """Latencies with the lowest and highest TRIM_FRACTION of them removed."""
    x = np.sort(np.asarray(latencies, dtype=float))
    k = int(len(x) * TRIM_FRACTION)
    return x[k:len(x) - k]


def mean_by_length(measurements: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prompt lengths with the trimmed mean latency and its standard error at each."""
    by_len = defaultdict(list)
    for m in measurements:
        by_len[m["prompt_tokens"]].extend(trimmed(m["latencies"]))
    lengths = np.array(sorted(by_len))
    samples = [np.asarray(by_len[n], dtype=float) for n in lengths]
    means = np.array([x.mean() for x in samples])
    sems = np.array([x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0 for x in samples])
    return lengths, means, sems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="?", type=Path,
                        help="timing results file (default: newest in results/timing_results_prefill/)")
    parser.add_argument("--out", type=Path, help="output image path")
    args = parser.parse_args()

    path = args.results or latest_results()
    if not path.exists() and (RESULTS_DIR / path).exists():
        path = RESULTS_DIR / path
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)

    lengths, means, sems = mean_by_length(rec["measurements"])
    means_ms, sems_ms = means * 1e3, sems * 1e3

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(lengths, means_ms, yerr=sems_ms, fmt="o-", ms=4, capsize=3)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("mean latency (ms)")
    ax.set_title(f"{rec['model']} ({rec.get('num_trials', '?')} trials per length)")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = args.out or path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"{path.name}: {len(lengths)} lengths -> {out}")
    for n, m, s in zip(lengths, means_ms, sems_ms):
        print(f"  {n:>7d} tokens  {m:9.3f} ± {s:.3f} ms")


if __name__ == "__main__":
    main()
