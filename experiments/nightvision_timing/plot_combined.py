#!/usr/bin/env python3
"""Plot the mean latency against prompt length for several timing results files on one axes.

Each file is reduced exactly as in plot_timing.py (trimmed mean per prompt
length with standard-error bars). Models are colored by family and told apart
within a family by marker, so the legend stays readable with ~20 models.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_timing import mean_by_length  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent

# Family -> color, in fixed categorical order; a model's family is the first
# prefix its name (lowercased, without org) starts with.
FAMILIES = [
    ("qwen2.5", "#2a78d6"),
    ("qwen3", "#eb6834"),
    ("llama", "#1baf7a"),
    ("olmo", "#eda100"),
    ("granite", "#e87ba4"),
]
OTHER_COLOR = "#6250d6"
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def family_color(model: str) -> str:
    name = model.split("/")[-1].lower()
    for prefix, color in FAMILIES:
        if name.startswith(prefix):
            return color
    return OTHER_COLOR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", type=Path, help="timing results files")
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "plots" / "combined_timing.png",
                        help="output image path (default: plots/combined_timing.png)")
    parser.add_argument("--logx", action="store_true", help="log-scale prompt length axis")
    parser.add_argument("--logy", action="store_true", help="log-scale latency axis")
    parser.add_argument("--title", default="Prefill latency vs prompt length")
    args = parser.parse_args()

    recs = []
    for path in args.results:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        if "measurements" not in rec:
            print(f"skipping {path.name}: no measurements")
            continue
        recs.append(rec)
    recs.sort(key=lambda r: r["model"].split("/")[-1].lower())
    if not recs:
        raise SystemExit("no usable results files")

    fig, ax = plt.subplots(figsize=(11, 6))
    markers_used: dict[str, int] = {}
    for rec in recs:
        color = family_color(rec["model"])
        marker = MARKERS[markers_used.get(color, 0) % len(MARKERS)]
        markers_used[color] = markers_used.get(color, 0) + 1
        lengths, means, sems = mean_by_length(rec["measurements"])
        ax.errorbar(lengths, means * 1e3, yerr=sems * 1e3, fmt="-", marker=marker, ms=5,
                    lw=1.5, capsize=2, color=color, label=rec["model"].split("/")[-1])
        print(f"{rec['model']}: {len(lengths)} lengths")

    if args.logx:
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("mean latency (ms)")
    ax.set_title(f"{args.title} ({len(recs)} models)")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
