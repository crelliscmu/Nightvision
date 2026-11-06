#!/usr/bin/env python3
"""Fit the prefill timing relations with the spectral d_hat at each sampling budget.

fit_timing.py uses one d_hat per model, from the end of sampling. This script
reads the d_hat of every recorded budget from
results/commonset_results/<stem>_sampling.json and runs the same leave-one-out
fits, --fit layers and --fit params, once per budget. Every model takes its d_hat
from the same budget. A budget where any of the timing models has no d_hat yet
(d_hat = -1) is skipped entirely, so every fit written covers all of them.

The true-d fits of fit_timing.py do not depend on the budget, so they are not
repeated here. The largest budget reproduces
results/timing_fits/timing_fit_{layers,params}_dhat.json.

Writes, under <out-dir> (default results/timing_fits_each_budget/):

    budget_<calls>/timing_fit_{layers,params}_dhat.json
                  the same format as fit_timing.py, e.g. budget_8B
    budgets.json  the budgets fitted, with each model's d_hat, and the budgets
                  skipped, with the models that had no d_hat
"""

import argparse
import json
from pathlib import Path

from fit_timing import (
    COMMONSET_DIR, EXCLUDED, MODEL_CONFIGS, embedding_params, load_models, load_sampling_run, run_fit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PREFILL_DIR = REPO_ROOT / "results" / "timing_results_prefill"
OUT_DIR = REPO_ROOT / "results" / "timing_fits_each_budget"
FITS = ("layers", "params")

def load_timing_models(prefill_dir: Path) -> list[dict]:
    """The timing models with their prefill points and the architecture fields the fits need."""
    with open(MODEL_CONFIGS, encoding="utf-8") as f:
        configs = json.load(f)
    models = [m for m in load_models(prefill_dir, 0) if not any(s in m["model"] for s in EXCLUDED)]
    if not models:
        raise SystemExit(f"no timing models in {prefill_dir}")
    for m in models:
        if m["model"] not in configs:
            raise SystemExit(f"{m['model']} is not in {MODEL_CONFIGS}")
        cfg = configs[m["model"]]
        m.update(P=cfg["active_parameters"], P_total=cfg["total_parameters"], V=cfg["vocab_size"],
                 tied=cfg.get("tie_word_embeddings", False))
        m["P_ne"] = m["P"] - embedding_params(m, m["d"])
    return models


def load_d_hats(commonset_dir: Path, models: list[dict]) -> dict[int, dict[str, int]]:
    """{API calls: {model: d_hat}}, with -1 where a model has no d_hat at that budget."""
    budgets: dict[int, dict[str, int]] = {}
    for m in models:
        run = load_sampling_run(commonset_dir, m["model"], m["d"])
        for step in run["steps"]:
            calls = step["draws_per_prompt_total"] * run["config"]["n_p"]
            budgets.setdefault(calls, {})[m["model"]] = step["d_hat"]
    for calls, d_hats in budgets.items():
        # A budget only some runs recorded cannot be fitted across all the models.
        for m in models:
            d_hats.setdefault(m["model"], -1)
    return dict(sorted(budgets.items()))


def budget_name(calls: int) -> str:
    """Directory name of a budget: budget_8B."""
    return f"budget_{calls / 1e9:g}B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--commonset-dir", type=Path, default=COMMONSET_DIR,
                        help="Sampling runs with d_hat per budget (default: %(default)s)")
    parser.add_argument("--prefill-dir", type=Path, default=PREFILL_DIR,
                        help="collect_prefill.py output (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Where the fits go (default: %(default)s)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = load_timing_models(args.prefill_dir)
    budgets = load_d_hats(args.commonset_dir, models)

    fitted, skipped = [], []
    for calls, d_hats in budgets.items():
        missing = sorted(model for model, d_hat in d_hats.items() if d_hat <= 0)
        if missing:
            print(f"Skipping {calls:,} calls: no d_hat for {len(missing)} of {len(models)} models")
            skipped.append({"calls": calls, "missing": missing})
            continue
        print(f"\n{'#' * 100}\n{calls:,} calls")
        out_dir = args.out_dir / budget_name(calls)
        run_args = argparse.Namespace(out_dir=out_dir, keep_longest=0, exclude_ref=EXCLUDED, exclude_pred=EXCLUDED)
        for fit in FITS:
            run_fit(models, run_args, fit, d_hats, plots=False)
        fitted.append({"calls": calls, "dir": out_dir.name, "d_hat": d_hats})

    if not fitted:
        raise SystemExit("no budget has a d_hat for every model")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "budgets.json", "w", encoding="utf-8") as f:
        json.dump({"models": [m["model"] for m in models], "fitted": fitted, "skipped": skipped}, f, indent=1)
    print(f"\nFitted {len(fitted)} budgets and skipped {len(skipped)}; saved to {args.out_dir}")


if __name__ == "__main__":
    main()
