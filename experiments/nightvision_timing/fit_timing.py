#!/usr/bin/env python3
"""Fit the prefill timing scaling relations and estimate depth and parameter count (timing attack, stages 2-3).

The prefill signal (results/timing_results_prefill/*_timing.json, written by
collect_prefill.py) is fit leave-one-out: every model is the target once with
all other models as the reference set. Every length is reduced to a 10%-trimmed mean with its
standard error std / sqrt(N_kept), and all fits are weighted least squares
with weights 1 / sigma.

The signal is the latency of a max_tokens=1 request less the l=1 baseline.
With --fit layers (the paper's relation) the unknown is L and the feature is
x = L l^2 d; with --fit params the unknown is the non-embedding active
parameter count P_ne = P - E(V, d) and the feature is x = P_ne l^2 / d,
since the block parameters scale as L d^2 so P_ne / d^2 stands in for L.
P is the active parameter count, the weights one token passes through
(active_parameters in model_configs.json); E(V, d) = (1 or 2) V d is the
embedding plus unembedding, excluded because with max_tokens=1 the
unembedding runs on the last position only. Writing Q for the unknown
and q(d) for the feature with Q removed:

    T ~ beta Q q(d) + alpha             (Q-known fit over the reference set)
    T ~ gamma_ref q(d) + zeta           (Q-unknown fit over the reference set, true d)
    T ~ gamma_tgt q(d_hat) + zeta       (same fit with the estimated d_hat)
    k = gamma_tgt / gamma_ref           (transfer factor; 1 when d_hat == d)
    T_g ~ gamma_g q(d_hat_g) + zeta_g   (per-model fit of the target)
    Q_hat_g = gamma_g / (k beta)

--fit layers estimates the depth L and --fit params the active parameter
count P = P_ne_hat + E(V, d_hat).

--fit all runs every fit in turn: layers and params each with the true d
(d_star) and again with the spectral d_hat. d_hat is the estimate of the final
sampling step in <commonset-dir>/<stem>_sampling.json (default
results/commonset_results/). Every model that is estimated or used as a
reference must have a sampling file with a positive final d_hat.

Writes <out-dir>/timing_fit_<fit>.json (timing_fit_<fit>_dhat.json when the
estimate used d_hat) and one scaling plot, which shows the reference fits
with the true d.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_CONFIGS = SCRIPT_DIR / "model_configs.json"  # true active_parameters per model (the 14 timing models)
OUT_DIR = SCRIPT_DIR.parents[1] / "results" / "timing_fits"
PREFILL_DIR = SCRIPT_DIR.parents[1] / "results" / "timing_results_prefill"
COMMONSET_DIR = SCRIPT_DIR.parents[1] / "results" / "commonset_results"  # <stem>_sampling.json per model
PREFILL_SUFFIX = "_timing.json"
TRIM_FRAC = 0.10  # 10%-trimmed mean
KEEP_LONGEST = 0  # points kept per model, longest first; 0 keeps all


def trimmed_stats(samples) -> tuple[float, float]:
    """Trimmed mean and its standard error std/sqrt(N_kept)."""
    x = np.sort(np.asarray(samples, dtype=float))
    cut = int(len(x) * TRIM_FRAC)
    x = x[cut:len(x) - cut]
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def prefill_points(rec: dict, keep_longest: int) -> dict:
    """Baseline-subtracted prefill latency per length, longest `keep_longest` lengths."""
    by_len = sorted(rec["measurements"], key=lambda m: m["prompt_tokens"])
    baseline, _ = trimmed_stats(by_len[0]["latencies"])  # l=1 entry (shortest prompt)
    points = by_len[1:][-keep_longest:]
    stats = [trimmed_stats(m["latencies"]) for m in points]
    return {
        "ell": np.array([m["prompt_tokens"] for m in points], dtype=float),
        "T": np.array([s[0] - baseline for s in stats]),
        "sigma": np.array([max(s[1], 1e-9) for s in stats]),
    }


def load_models(directory: Path, keep_longest: int) -> list[dict]:
    """One record per model with its architecture and its prefill points (ell, T, sigma)."""
    models: dict[str, dict] = {}
    for path in sorted(directory.glob(f"*{PREFILL_SUFFIX}")):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        arch = rec["arch"]
        models[rec["model"]] = {
            "model": rec["model"],
            "d": arch["hidden_size"],
            "L": arch["num_hidden_layers"],
            **prefill_points(rec, keep_longest),
        }
    return [models[k] for k in sorted(models)]


def wfit(cols: list[np.ndarray], y: np.ndarray, sigma: np.ndarray) -> list[float]:
    """Weighted least squares y = sum_i c_i cols_i + intercept, weights 1/sigma.

    Returns [c_1, ..., c_n, intercept]. Columns are rescaled to unit maximum
    before solving, since l and l^2 features differ by orders of magnitude.
    """
    X = np.column_stack(list(cols) + [np.ones_like(y)])
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1.0
    coef, *_ = np.linalg.lstsq(X / scale / sigma[:, None], y / sigma, rcond=None)
    return [float(c) for c in coef / scale]


def stack(models: list[dict], xfuns: list) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Concatenate ([x_1, ..., x_n], T, sigma) over models, x_i = xfuns[i](model) per point."""
    cols = [np.concatenate([xf(m) for m in models]) for xf in xfuns]
    y = np.concatenate([m["T"] for m in models])
    s = np.concatenate([m["sigma"] for m in models])
    return cols, y, s


# --- spectral d_hat ----------------------------------------------------------

# Sampling-file stem of each of the 14 timing models.
SAMPLING_STEMS = {
    "ibm-granite/granite-3.3-2b-instruct": "granite_3.3_2b",
    "ibm-granite/granite-3.3-8b-instruct": "granite_3.3_8b",
    "ibm-granite/granite-4.0-micro": "granite_4.0_micro",
    "meta-llama/Llama-3.1-8B-Instruct": "llama_3.1_8b",
    "meta-llama/Llama-3.2-1B-Instruct": "llama_3.2_1b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama_3.2_3b",
    "allenai/OLMo-2-0425-1B-Instruct": "olmo_2_1b",
    "allenai/OLMo-2-1124-7B-Instruct": "olmo_2_7b",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5_1.5b",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5_3b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5_7b",
    "Qwen/Qwen3-1.7B": "qwen3_1.7b",
    "Qwen/Qwen3-4B": "qwen3_4b",
    "Qwen/Qwen3-8B": "qwen3_8b",
}


def load_sampling_run(commonset_dir: Path, model: str, d: int) -> dict:
    """The sampling run of one model, checked against its true hidden size d."""
    if model not in SAMPLING_STEMS:
        raise SystemExit(f"no sampling file known for {model}; add it to SAMPLING_STEMS")
    path = commonset_dir / f"{SAMPLING_STEMS[model]}_sampling.json"
    if not path.is_file():
        raise SystemExit(f"missing sampling file: {path}")
    with open(path, encoding="utf-8") as f:
        run = json.load(f)
    if run["config"]["true_dim"] != d:
        raise SystemExit(f"{path.name}: true_dim {run['config']['true_dim']} but {model} has d={d}")
    return run


def load_final_d_hats(commonset_dir: Path, models: list[dict]) -> dict[str, int]:
    """{model: d_hat of the final sampling step} for the given models."""
    return {m["model"]: load_sampling_run(commonset_dir, m["model"], m["d"])["steps"][-1]["d_hat"]
            for m in models}


# --- prefill relation -------------------------------------------------------

def feature(m: dict, fit: str, d: float) -> np.ndarray:
    """q(d): the scaling feature with the unknown quantity removed."""
    return m["ell"] ** 2 * d if fit == "layers" else m["ell"] ** 2 / d


def known_feature(m: dict, fit: str) -> np.ndarray:
    """Q q(d) with the true Q and d: L l^2 d or P_ne l^2 / d."""
    return (m["L"] if fit == "layers" else m["P_ne"]) * feature(m, fit, m["d"])


def embedding_params(m: dict, d: float) -> float:
    """Embedding plus unembedding parameters, V d if tied else 2 V d."""
    return (1 if m["tied"] else 2) * m["V"] * d


QUANTITY = {"layers": "L", "params": "P"}  # --fit value -> the one quantity its estimates report


def errors(target: dict, **hats: float | None) -> dict:
    """{Q_hat, Q_rel_err} for every estimated quantity Q in (L, P) that is not None."""
    true = {"L": target["L"], "P": target["P"]}
    out = {}
    for q, hat in hats.items():
        if hat is not None:
            out[f"{q}_hat"] = hat
            out[f"{q}_rel_err"] = (hat - true[q]) / true[q]
    return out


def estimate_prefill(target: dict, refs: list[dict], fit: str) -> dict:
    """Stage 2 and 3 of the timing attack on the prefill signal."""
    beta, alpha = wfit(*stack(refs, [lambda m: known_feature(m, fit)]))
    gamma_ref = wfit(*stack(refs, [lambda m: feature(m, fit, m["d"])]))[0]
    gamma_tgt = wfit(*stack(refs, [lambda m: feature(m, fit, m["d_hat"])]))[0]
    k = gamma_tgt / gamma_ref
    gamma_g, _ = wfit([feature(target, fit, target["d_hat"])], target["T"], target["sigma"])
    q_hat = gamma_g / (k * beta)
    emb_hat = embedding_params(target, target["d_hat"])
    hat = {"L": q_hat} if fit == "layers" else {"P": emb_hat + q_hat}
    return {**errors(target, **hat), "beta": beta, "alpha": alpha, "k": k, "gamma_g": gamma_g}


# --- plots ------------------------------------------------------------------

def plot_prefill_scaling(models: list[dict], fit: str, out: Path) -> None:
    """Measured prefill time against the attention feature."""
    beta, alpha = wfit(*stack(models, [lambda m: known_feature(m, fit)]))
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        ax.errorbar(known_feature(m, fit), m["T"], yerr=m["sigma"],
                    fmt="o", ms=3, label=m["model"].split("/")[-1])
    xs = np.array(ax.get_xlim())
    ax.plot(xs, beta * xs + alpha, "k--", label=f"fit: beta={beta:.3g}, alpha={alpha:.3g}")
    ax.set_xlabel(r"$L\,\ell^2\,d$" if fit == "layers" else r"$P_{ne}\,\ell^2 / d$")
    ax.set_ylabel("prefill time (s), baseline-subtracted")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --- driver -----------------------------------------------------------------

# The 14 timing models are the dense validation models above 1B parameters whose
# every layer uses global multi-head attention (MHA or GQA), with one Granite 3.x
# release kept per architecture. Everything else is neither predicted nor used as
# a reference.
EXCLUDED = [
    # Below 1B parameters.
    "ibm-granite/granite-4.0-350m", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B",
    "HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct",
    # Mixture-of-experts.
    "ibm-granite/granite-3.0-1b-a400m-instruct", "ibm-granite/granite-3.0-3b-a800m-instruct",
    "ibm-granite/granite-3.1-1b-a400m-instruct", "ibm-granite/granite-3.1-3b-a800m-instruct",
    "allenai/OLMoE-1B-7B-0924-Instruct", "Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-30B-A3B",
    # Not global attention in every layer: Gated DeltaNet (Qwen3.5), sliding window (OLMo-3).
    "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B",
    "allenai/Olmo-3-7B-Instruct",
    # Development models, held out from every reported result.
    "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM3-3B",
    # Duplicate architectures, which would skew the fits: Granite 3.3 stands in for 3.0-3.2.
    "ibm-granite/granite-3.0-2b-instruct", "ibm-granite/granite-3.1-2b-instruct",
    "ibm-granite/granite-3.2-2b-instruct", "ibm-granite/granite-3.0-8b-instruct",
    "ibm-granite/granite-3.1-8b-instruct", "ibm-granite/granite-3.2-8b-instruct",
]
EXCLUDED_REF = EXCLUDED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fit", choices=FITS + ("all",), default="all",
                        help="Quantity the prefill attention term is attributed to (default: "
                             "all); layers reports the depth L only and params the "
                             "active parameter count P only. "
                             "all runs every fit, layers and params both with the true d and "
                             "with d_hat")
    parser.add_argument("--prefill-dir", type=Path, default=PREFILL_DIR,
                        help=f"Directory of *{PREFILL_SUFFIX} files (default: results/{PREFILL_DIR.name}/)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help=f"Where the fit JSON and scaling plots go (default: results/timing_fits/)")
    parser.add_argument("--keep-longest", type=int, default=KEEP_LONGEST,
                        help=f"Longest lengths kept per model, 0 for all (default: {KEEP_LONGEST})")
    parser.add_argument("--commonset-dir", type=Path, default=COMMONSET_DIR,
                        help="Sampling runs <stem>_sampling.json; d_hat is the final step's estimate. Every "
                             "estimated or reference model must have one (default: results/commonset_results/)")
    parser.add_argument("--exclude-ref", nargs="*", default=EXCLUDED_REF,
                        help="Models whose name contains any of these substrings are still "
                             "estimated but never used as references (default: every model outside "
                             "the 14 timing models)")
    parser.add_argument("--exclude-pred", nargs="*", default=EXCLUDED,
                        help="Models whose name contains any of these substrings are not "
                             "estimated (no prediction row) but may still serve as references "
                             "(default: every model outside the 14 timing models)")
    return parser.parse_args()


FORMATS = {  # quantity -> (true value of a result row, value formatter)
    "L": (lambda r: f"{r['L']:6d}", lambda v: f"{v:7.1f}"),
    "P": (lambda r: f"{r['P'] / 1e9:5.2f}B", lambda v: f"{v / 1e9:6.2f}B"),
}


def cell(r: dict | None, key: str) -> str:
    """One table cell: estimate with its relative error, or a dash."""
    if r is None or f"{key}_hat" not in r:
        return f"{'-':^16s}"
    return f"{FORMATS[key][1](r[f'{key}_hat'])} ({r[f'{key}_rel_err']:+6.1%})"


def print_table(results: list[dict], key: str, title: str, d_label: str = "d_hat", extra=None) -> None:
    """Prefill estimates of one quantity with relative errors, then the mean |rel err|.

    d_label names the width column: d when the estimates used the true d, d_hat otherwise.
    """
    true_of, _ = FORMATS[key]
    print(f"\n{title}")
    print(f"{'model':45s} {d_label:>6s} {'true':>6s} {'prefill':^16s}"
          + (f"  {extra[0]}" if extra else ""))
    for r in results:
        print(f"{r['model']:45s} {r['d_hat']:6.0f} {true_of(r):>6s} {cell(r['estimates'].get('prefill'), key)}"
              + (f"  {extra[1](r)}" if extra else "") + ("  (MoE)" if r["moe"] else ""))
    errs = [abs(r["estimates"]["prefill"][f"{key}_rel_err"]) for r in results if "prefill" in r["estimates"]]
    print(f"mean |{key} rel err|: prefill {np.mean(errs):.1%} (n={len(errs)})")


FITS = ("layers", "params")


def is_ref(m: dict, args: argparse.Namespace) -> bool:
    return not any(s in m["model"] for s in args.exclude_ref)


def is_pred(m: dict, args: argparse.Namespace) -> bool:
    return not any(s in m["model"] for s in args.exclude_pred)


def check_d_hats(models: list[dict], d_hats: dict, args: argparse.Namespace) -> None:
    """Exit unless every estimated or reference model has a positive d_hat in d_hats."""
    used = [m["model"] for m in models if is_ref(m, args) or is_pred(m, args)]
    missing = [name for name in used if name not in d_hats]
    bad = [name for name in used if name in d_hats
           and (isinstance(d_hats[name], bool) or not isinstance(d_hats[name], (int, float))
                or not np.isfinite(d_hats[name]) or d_hats[name] <= 0)]
    if missing or bad:
        raise SystemExit(
            f"{args.commonset_dir}: every estimated or reference model needs a positive numeric d_hat"
            + (f"\n  missing: {', '.join(missing)}" if missing else "")
            + (f"\n  invalid: {', '.join(f'{n}={d_hats[n]!r}' for n in bad)}" if bad else "")
            + "\nAdd the estimates or exclude the models with --exclude-ref and --exclude-pred.")


def run_fit(models: list[dict], args: argparse.Namespace, fit: str, d_hats: dict, plots: bool) -> None:
    """One complete fit: leave-one-out estimates, tables, JSON and (optionally) the scaling plots.

    d_hats is {model: d_hat} for an estimate with the spectral d_hat, or {} to use
    the true d (d_star) everywhere.
    """
    for m in models:
        # Models outside both sets are not fit.
        m["d_hat"] = float(d_hats.get(m["model"], np.nan)) if d_hats else float(m["d"])
    d_source = "d_hat" if d_hats else "d_star"
    d_label = "d_hat" if d_hats else "d"  # width column of the tables

    if d_hats:
        check_d_hats(models, d_hats, args)

    results = []
    for target in (m for m in models if is_pred(m, args)):
        estimates = {}
        refs = [m for m in models if m is not target and is_ref(m, args)]
        if len(refs) >= 3:
            estimates["prefill"] = estimate_prefill(target, refs, fit)
        results.append({
            "model": target["model"], "d": target["d"], "d_hat": target["d_hat"],
            "L": target["L"],
            "moe": target["P_total"] != target["P"], "P_total": target["P_total"], "P": target["P"],
            "estimates": estimates,
        })

    print(f"\n{'=' * 100}\nfit: {fit}, d: {d_source}")
    key = QUANTITY[fit]
    print_table(results, key, {"L": "Depth L", "P": "Active parameters P"}[key],
                d_label)
    print(("P is the active parameter count (= total for dense models); " if key == "P" else "")
          + ("d_hat is the final spectral estimate from --commonset-dir." if d_hats else "d is the true hidden size (d_star)."))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"timing_fit_{fit}{'_dhat' if d_hats else ''}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"fit": fit, "d": d_source, "keep_longest": args.keep_longest,
                   "exclude_ref": args.exclude_ref, "exclude_pred": args.exclude_pred,
                   "results": results}, f, indent=1)
    saved = []
    refs = [m for m in models if is_ref(m, args)]
    if plots and refs:
        png = args.out_dir / f"scaling_law_{fit}_prefill.png"
        plot_prefill_scaling(refs, fit, png)
        saved.append(png.name)
    print(f"Saved {out}" + (f" and {', '.join(saved)}" if saved else ""))


def main():
    args = parse_args()
    models = load_models(args.prefill_dir, args.keep_longest)
    if not models:
        raise SystemExit(f"no results files found in {args.prefill_dir}")
    with open(MODEL_CONFIGS, encoding="utf-8") as f:
        configs = json.load(f)
    for m in list(models):
        if m["model"] not in configs:
            print(f"skipping {m['model']}: not in {MODEL_CONFIGS.name}")
            models.remove(m)
            continue
        cfg = configs[m["model"]]
        m["P"] = cfg["active_parameters"]
        m["P_total"] = cfg["total_parameters"]
        m["V"] = cfg["vocab_size"]
        m["tied"] = cfg.get("tie_word_embeddings", False)
        m["P_ne"] = m["P"] - embedding_params(m, m["d"])

    used = [m for m in models if is_ref(m, args) or is_pred(m, args)]
    d_hats = load_final_d_hats(args.commonset_dir, used)
    check_d_hats(models, d_hats, args)  # before any fit, so --fit all writes nothing on failure
    if args.fit != "all":
        run_fit(models, args, args.fit, d_hats, plots=True)
        return
    for fit in FITS:
        run_fit(models, args, fit, {}, plots=True)  # true d (d_star)
        run_fit(models, args, fit, d_hats, plots=False)


if __name__ == "__main__":
    main()
