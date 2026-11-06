#!/usr/bin/env python3
"""Fit the timing attack across two hardware platforms (timing attack, cross-platform calibration).

The same procedure as fit_timing.py (paper, "Inferring Depth under Hardware
Mismatch"), except that the reference relation and the target measurements come
from different machines. Writing Q for the unknown (L with --fit layers, the
non-embedding parameter count P_ne with --fit params) and q(d) for the feature
with Q removed (l^2 d or l^2 / d), for a held-out target model g:

    T_ref ~ beta Q q(d) + alpha          (Q-known fit, reference platform, true d)
    T_ref ~ gamma_ref q(d) + zeta        (Q-unknown fit, reference platform, true d)
    T_tgt ~ gamma_tgt q(d_hat) + zeta'   (Q-unknown fit, target platform, estimated d_hat)
    k = gamma_tgt / gamma_ref            (platform transfer factor)
    T_g ~ gamma_g q(d_hat_g) + zeta_g    (per-model fit of g, target platform)
    Q_hat_g = gamma_g / (k beta)

The target model is left out of the reference and the target group fits, and
only models measured on both platforms are used. With the two platforms set to
the same directory this reduces exactly to the prefill estimate of fit_timing.py.

Each pair of platforms is run in both directions, for both fits, with the true
hidden dimension (d_star) and the spectral estimate (d_hat), the final sampling
step's estimate in <commonset-dir>/<stem>_sampling.json (default
results/commonset_results/). Writes
<out-dir>/timing_cross_<ref>_to_<tgt>_<fit>[_dhat].json and prints a summary table.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fit_timing import (
    COMMONSET_DIR, EXCLUDED, MODEL_CONFIGS, embedding_params, errors, feature, known_feature,
    load_final_d_hats, load_models, stack, wfit,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parents[1] / "results" / "timing_fits_cross"
RESULTS_DIR = SCRIPT_DIR.parents[1] / "results"
PLATFORMS = {"h100": RESULTS_DIR / "timing_results_prefill", "h200": RESULTS_DIR / "timing_results_prefill_h200"}
FITS = ("layers", "params")
ERROR_KEY = {"layers": "L", "params": "P"}


def parse_platform(text: str) -> tuple[str, Path]:
    """NAME=DIR on the command line."""
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=DIR, got {text!r}")
    return name, Path(path)


def load_platform(path: Path, configs: dict) -> dict[str, dict]:
    """{model name: flat record} for one platform's prefill measurements."""
    records = {}
    for rec in load_models(path, 0):
        if rec["model"] not in configs:
            print(f"skipping {rec['model']}: not in {MODEL_CONFIGS.name}")
            continue
        cfg = configs[rec["model"]]
        rec.update(P=cfg["active_parameters"], V=cfg["vocab_size"],
                   tied=cfg.get("tie_word_embeddings", False))
        rec["P_ne"] = rec["P"] - embedding_params(rec, rec["d"])
        records[rec["model"]] = rec
    return records


def estimate(target: dict, refs: list[dict], tgts: list[dict], fit: str) -> dict:
    """Q_hat_g = gamma_g / (k beta), with beta and gamma_ref from refs and gamma_tgt from tgts."""
    beta, alpha = wfit(*stack(refs, [lambda m: known_feature(m, fit)]))
    gamma_ref = wfit(*stack(refs, [lambda m: feature(m, fit, m["d"])]))[0]
    gamma_tgt = wfit(*stack(tgts, [lambda m: feature(m, fit, m["d_hat"])]))[0]
    k = gamma_tgt / gamma_ref
    gamma_g, zeta_g = wfit([feature(target, fit, target["d_hat"])], target["T"], target["sigma"])
    q_hat = gamma_g / (k * beta)
    hat = {"L": q_hat} if fit == "layers" else {"P": q_hat + embedding_params(target, target["d_hat"])}
    return {**errors(target, **hat), "beta": beta, "alpha": alpha, "gamma_ref": gamma_ref,
            "gamma_tgt": gamma_tgt, "k": k, "gamma_g": gamma_g, "zeta_g": zeta_g}


def run_direction(ref: dict[str, dict], tgt: dict[str, dict], names: list[str], fit: str) -> list[dict]:
    """Leave-one-out estimates of every model on the target platform."""
    results = []
    for name in names:
        refs = [ref[n] for n in names if n != name]
        tgts = [tgt[n] for n in names if n != name]
        t = tgt[name]
        results.append({"model": name, "d": t["d"], "d_hat": t["d_hat"], "L": t["L"], "P": t["P"],
                        **estimate(t, refs, tgts, fit)})
    return results


def format_hat(result: dict, key: str) -> str:
    """One estimate with its signed relative error."""
    value = result[f"{key}_hat"]
    shown = f"{value / 1e9:6.2f}B" if key == "P" else f"{value:7.1f}"
    return f"{shown} ({result[f'{key}_rel_err']:+7.1%})"


def print_tables(runs: dict, directions: list[tuple[str, str]]) -> None:
    """Per-model estimates and the mean absolute relative error of every run."""
    for fit in FITS:
        key = ERROR_KEY[fit]
        for d_source in ("d_star", "d_hat"):
            labels = [f"{a}->{b}" for a, b in directions]
            print(f"\n{'=' * 90}\nfit: {fit}, d: {d_source}")
            print(f"{'model':42s} {'true':>7s} " + " ".join(f"{label:^18s}" for label in labels))
            columns = [runs[fit, d_source, a, b] for a, b in directions]
            for rows in zip(*columns):
                true = f"{rows[0]['P'] / 1e9:.2f}B" if key == "P" else str(rows[0]["L"])
                print(f"{rows[0]['model']:42s} {true:>7s} "
                      + " ".join(f"{format_hat(r, key):^18s}" for r in rows))
            means = [np.mean([abs(r[f"{key}_rel_err"]) for r in c]) for c in columns]
            print(f"{'mean |rel err|':42s} {'':7s} " + " ".join(f"{m:^18.1%}" for m in means))


def repo_relative(path: Path) -> str:
    """path relative to the repository root when it lies inside it, else as given."""
    path = Path(path).resolve()
    return str(path.relative_to(RESULTS_DIR.parent)) if path.is_relative_to(RESULTS_DIR.parent) else str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--platforms", nargs=2, type=parse_platform, metavar="NAME=DIR",
                        default=list(PLATFORMS.items()),
                        help="The two platforms and their prefill results directories "
                             "(default: h100=results/timing_results_prefill h200=results/timing_results_prefill_h200)")
    parser.add_argument("--commonset-dir", type=Path, default=COMMONSET_DIR,
                        help="Sampling runs <stem>_sampling.json; d_hat is the final step's estimate "
                             "(default: results/commonset_results/)")
    parser.add_argument("--exclude", nargs="*", default=EXCLUDED,
                        help="Models left out (default: the exclusions of fit_timing.py)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help=f"Where the fit JSONs go (default: results/timing_fits_cross/)")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(MODEL_CONFIGS, encoding="utf-8") as f:
        configs = json.load(f)
    (name_a, dir_a), (name_b, dir_b) = args.platforms
    platforms = {name_a: load_platform(dir_a, configs), name_b: load_platform(dir_b, configs)}
    shared = set(platforms[name_a]) & set(platforms[name_b])
    names = sorted(n for n in shared if not any(s in n for s in args.exclude))
    if len(names) < 3:
        raise SystemExit(f"need at least 3 models measured on both platforms, found {len(names)}")
    print(f"{len(names)} models measured on both {name_a} and {name_b}")
    d_hats = load_final_d_hats(args.commonset_dir, [platforms[name_a][n] for n in names])
    bad = {n: v for n, v in d_hats.items() if v <= 0}
    if bad:
        raise SystemExit(f"{args.commonset_dir}: no final d_hat for {', '.join(bad)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    directions = [(name_a, name_b), (name_b, name_a)]
    runs = {}
    for fit in FITS:
        for d_source in ("d_star", "d_hat"):
            for rec in (platform[n] for platform in platforms.values() for n in names):
                rec["d_hat"] = float(d_hats[rec["model"]]) if d_source == "d_hat" else float(rec["d"])
            for ref, tgt in directions:
                results = run_direction(platforms[ref], platforms[tgt], names, fit)
                runs[fit, d_source, ref, tgt] = results
                suffix = "_dhat" if d_source == "d_hat" else ""
                out = args.out_dir / f"timing_cross_{ref}_to_{tgt}_{fit}{suffix}.json"
                with open(out, "w", encoding="utf-8") as f:
                    json.dump({"fit": fit, "d": d_source, "reference": ref, "target": tgt,
                               "reference_dir": repo_relative(dict(args.platforms)[ref]),
                               "target_dir": repo_relative(dict(args.platforms)[tgt]),
                               "results": results}, f, indent=1)
    print_tables(runs, directions)
    print(f"\nSaved the fits to {args.out_dir}")


if __name__ == "__main__":
    main()
