#!/usr/bin/env python3
"""Generate every NightVision figure: hidden-dimension recovery and the timing attack.

    python3 experiments/plotting/generate_plots.py
    python3 experiments/plotting/generate_plots.py --figure timing

Inputs, resolved from the repository root:

    model_configs.json               architecture, total_parameters and vocab_size
    results/commonset_results/       <stem>_sampling.json, one trajectory per model
                                     (nightvision_common_set/sampling/run_sampling.py)
    results/timing_fits/             timing_fit_<fit>[_dhat].json (fit_timing.py)
    results/timing_fits_cross/       timing_cross_<ref>_to_<tgt>_<fit>[_dhat].json
                                     (fit_timing_cross.py)
    results/timing_results_prefill/  <model>_timing.json (collect_prefill.py)
    results/timing_fits_each_budget/ budget_<calls>/timing_fit_{layers,params}_dhat.json
                                     and budgets.json (fit_timing_budgets.py)
    results/qwen_3.5_timing/         <model>_timing.json (collect_prefill.py), the
                                     Qwen3.5 models, in the global fit figure only

Figures, written to results/figs/ unless --output-dir or a --save-* path says
otherwise. --figure picks a group: error, ratio, timing or all (the default).

    error:
    avg_error_dense_vs_moe.pdf       mean relative d error of each size tier against
                                     API calls per prompt (Dense solid, MoE dashed)

    ratio:
    call_ratio_by_target.pdf         cost overhead relative to a Carlini-style logit
                                     extraction, by accuracy target

    timing:
    per_model_error_bars.png         L and P error with the true d and the spectral
                                     d_hat, above the d_hat error
    cross_gpu_comparison.png         L and P error in both reference -> target
                                     directions, using d_hat
    cross_gpu_comparison_true_d.png  the same with the true d
    global_timing_fit.png            measured time against L l^2 d, 18 models
    figure_2.png                     d, L and P error against API calls per prompt,
                                     with d at every sampling budget
    err_vs_api_calls_cost_tokens.png the same against calls per prompt, total USD
                                     and total tokens
    L_N_error_vs_query_budget_by_group.png
                                     P and L error against calls per prompt, per size group,
                                     with the d_hat of each budget

Budgets are in API calls (one sampled token per call). The timing budget figures
share the style of plot_average_relative_error: the mean of each series at every
budget, on linear axes.

The bar and query-budget figures only read saved fits, with one point per budget
that fit_timing_budgets.py fitted, so every query-budget point covers all the
models. Their largest budget must reproduce results/timing_fits/. The exception
is the d line of figure_2.png, which needs no fit: it also covers the budgets
fit_timing_budgets.py skipped, averaging the models that have a d_hat there.

Averages over timing models are taken per budget step; see step_average.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # figures are only ever saved, never shown
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Repository root: the directory holding model_configs.json and results/.
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

RESULTS_ROOT = REPO_ROOT / "results"
# Sampling trajectories, <stem>_sampling.json (sampling/run_sampling.py).
SAMPLING_RESULTS_DIR = RESULTS_ROOT / "commonset_results"
SAMPLING_GLOB = "*_sampling.json"
# avg_prompt_tokens per HF model id (prompt_generation/postprocess.py).
# Used for token-unit budgets and by the timing figures' cost and token axes.
TOKEN_BUDGET_FILE = RESULTS_ROOT / "avg_prompt_tokens.json"
# architecture / total_parameters / vocab_size per HF model id, at the repository root.
MODEL_CONFIGS_FILE = REPO_ROOT / "model_configs.json"
# Where the figures go unless --output-dir or a --save-* option says otherwise.
FIGS_DIR = RESULTS_ROOT / "figs"
# The error figure is drawn against API calls only.
DEFAULT_ERROR_FIGURE = "avg_error_dense_vs_moe.pdf"
DEFAULT_RATIO_FIGURE = "call_ratio_by_target.pdf"
# Inputs of the timing figures.
FITS_DIR = RESULTS_ROOT / "timing_fits"
CROSS_DIR = RESULTS_ROOT / "timing_fits_cross"
PREFILL_DIR = RESULTS_ROOT / "timing_results_prefill"
QWEN35_DIR = RESULTS_ROOT / "qwen_3.5_timing"
BUDGET_FITS_DIR = RESULTS_ROOT / "timing_fits_each_budget"
PREFILL_SUFFIX = "_timing.json"
TRIM_FRAC = 0.10  # 10%-trimmed mean of the prefill latencies, as in fit_timing.py

# `relative_error` is the error of the grid-rounded estimate `d_hat`.
# Set this to `relative_error_raw` to plot the error of the unrounded peak
# index (`aoi_peak`) instead.
PLOT_ERROR = "relative_error"

# Budget unit used when a helper is called with budget=None. `main` draws
# the error and ratio figures against API calls.
BUDGET_UNIT = "calls"

# API calls charged to the run: `total_samples` is the cumulative number of
# sampled tokens over all prompts (draws_per_prompt_total x n_p) and is the
# cost axis used by every figure here.
CALLS_COL = "total_samples"

# Prompt pool of every run (config["n_p"]). avg_error_dense_vs_moe.pdf,
# figure_2.png, L_N_error_vs_query_budget_by_group.png and the calls panel
# of err_vs_api_calls_cost_tokens.png label
# their budget axis in API calls per prompt (total calls / N_PROMPTS); the
# calls to different prompts are independent and can be batched.
N_PROMPTS = 20_000
PER_PROMPT_LABEL = f"API Calls per Prompt ({N_PROMPTS:,} prompts)"

# Prompt tokens per Carlini logit-bias query in token-budget mode. None uses
# each model's own `avg_prompt_tokens` (the same prompts we send), which
# makes the token ratio equal to the call ratio; set a number to price
# Carlini queries at a fixed prompt length instead.
CARLINI_TOKENS_PER_CALL = None

# Accuracy targets swept by the call-ratio figure, tightest last.
ACCURACY_TARGETS = (0.7, 0.5, 0.3, 0.125, 0.0)

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

# Run names -> Hugging Face ids, the keys used by TOKEN_BUDGET_FILE and
# MODEL_CONFIGS_FILE.
MODEL_HF_IDS = {
    "qwen2.5_0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5_1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5_3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5_7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3_0.6b": "Qwen/Qwen3-0.6B",
    "qwen3_1.7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_30b": "Qwen/Qwen3-30B-A3B",
    "qwen3_30b_instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3.5_0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5_2b": "Qwen/Qwen3.5-2B",
    "qwen3.5_4b": "Qwen/Qwen3.5-4B",
    "qwen3.5_9b": "Qwen/Qwen3.5-9B",
    "granite_3.0_1b_a400m": "ibm-granite/granite-3.0-1b-a400m-instruct",
    "granite_3.0_2b": "ibm-granite/granite-3.0-2b-instruct",
    "granite_3.0_3b_a800m": "ibm-granite/granite-3.0-3b-a800m-instruct",
    "granite_3.0_8b": "ibm-granite/granite-3.0-8b-instruct",
    "granite_3.1_1b_a400m": "ibm-granite/granite-3.1-1b-a400m-instruct",
    "granite_3.1_2b": "ibm-granite/granite-3.1-2b-instruct",
    "granite_3.1_3b_a800m": "ibm-granite/granite-3.1-3b-a800m-instruct",
    "granite_3.1_8b": "ibm-granite/granite-3.1-8b-instruct",
    "granite_3.2_2b": "ibm-granite/granite-3.2-2b-instruct",
    "granite_3.2_8b": "ibm-granite/granite-3.2-8b-instruct",
    "granite_3.3_2b": "ibm-granite/granite-3.3-2b-instruct",
    "granite_3.3_8b": "ibm-granite/granite-3.3-8b-instruct",
    "granite_4.0_350m": "ibm-granite/granite-4.0-350m",
    "granite_4.0_micro": "ibm-granite/granite-4.0-micro",
    "llama_3.1_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama_3.2_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama_3.2_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "olmo_2_1b": "allenai/OLMo-2-0425-1B-Instruct",
    "olmo_2_7b": "allenai/OLMo-2-1124-7B-Instruct",
    "olmo_3_7b": "allenai/Olmo-3-7B-Instruct",
    "olmoe_7b": "allenai/OLMoE-1B-7B-0924-Instruct",
    "smol2_135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smol2_360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smol2_1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smol3_3b": "HuggingFaceTB/SmolLM3-3B",
}

# Tiers use total parameter count: small < 3B, mid 3B to 6B, and
# large >= 6B. MoE models are classified by their full architecture,
# not by active parameter count.
TIER_BOUNDS = {"small": (0, 3e9), "mid": (3e9, 6e9), "large": (6e9, float("inf"))}
TIER_ORDER = list(TIER_BOUNDS)


def load_model_configs(path=MODEL_CONFIGS_FILE):
    """Run name -> {"architecture", "total_parameters", ...} from model_configs.json."""
    with Path(path).open() as f:
        by_hf_id = json.load(f)
    return {model: by_hf_id[hf_id] for model, hf_id in MODEL_HF_IDS.items()
            if hf_id in by_hf_id}


def _size_tier(total_parameters):
    for tier, (low, high) in TIER_BOUNDS.items():
        if low <= total_parameters < high:
            return tier
    raise ValueError(f"No size tier for {total_parameters} parameters")


MODEL_CONFIGS = load_model_configs()
MODEL_TO_ARCHITECTURE = {
    model: config["architecture"] for model, config in MODEL_CONFIGS.items()
}
MODEL_TO_TIER = {
    model: _size_tier(config["total_parameters"]) for model, config in MODEL_CONFIGS.items()
}
# Models per tier, smallest first.
TIERS = {
    tier: sorted((m for m, t in MODEL_TO_TIER.items() if t == tier),
                 key=lambda m: MODEL_CONFIGS[m]["total_parameters"])
    for tier in TIER_ORDER
}
MOE_MODELS = {model for model, arch in MODEL_TO_ARCHITECTURE.items() if arch == "MoE"}
ALL_MODELS = [model for tier_models in TIERS.values() for model in tier_models]

# Sample-count columns and their token counterparts (calls x avg prompt tokens).
TOKEN_COLUMNS = {
    "total_samples": "total_tokens",
}

# Budget units selectable with `budget=`.
BUDGETS = {
    "calls": {
        "label": "Budget (API Calls)",
        "unit": "API Calls",
        "short": "calls",
        "singular": "call",
        "columns": {col: col for col in TOKEN_COLUMNS},
    },
    "tokens": {
        "label": "Budget (Tokens)",
        "unit": "Tokens",
        "short": "tokens",
        "singular": "token",
        "columns": dict(TOKEN_COLUMNS),
    },
}


def _budget(budget=None):
    """Resolve a budget unit name (default `BUDGET_UNIT`) to its spec."""
    budget = BUDGET_UNIT if budget is None else budget
    if budget not in BUDGETS:
        raise ValueError(f"budget must be one of {sorted(BUDGETS)}, got {budget!r}")
    return BUDGETS[budget]


def _budget_col(budget=None, samples_col="total_samples"):
    """Column holding `samples_col` expressed in the requested budget unit."""
    return _budget(budget)["columns"][samples_col]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _model_from_path(path):
    """Model stem of a <stem>_sampling.json path."""
    return Path(path).stem.removesuffix("_sampling")


def _load_runs(paths, what):
    """Parse run JSONs, reporting every unreadable file (truncated copy?) at once."""
    payloads, unreadable = {}, []
    for path in paths:
        try:
            payloads[path] = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            unreadable.append(f"{path.name} ({path.stat().st_size} bytes: {exc.msg} at char {exc.pos})")
    if unreadable:
        raise ValueError(f"{what} JSON files that cannot be parsed (truncated copy?):\n  "
                         + "\n  ".join(unreadable))
    return payloads


def load_prompt_token_averages(path=TOKEN_BUDGET_FILE):
    """Map run name -> average prompt tokens per API call.

    Returns an empty mapping (with a warning) when `path` does not exist,
    in which case every token-unit column is NaN and token figures are
    skipped.
    """
    if path is None or not Path(path).exists():
        print(f"[warn] no prompt token averages at {path}; token budgets unavailable")
        return {}
    with Path(path).open() as f:
        entries = json.load(f)["models"]
    by_hf_id = {entry["model"]: float(entry["avg_prompt_tokens"]) for entry in entries}
    return {model: by_hf_id[hf_id] for model, hf_id in MODEL_HF_IDS.items()
            if hf_id in by_hf_id}


def load_sampling_metrics(metrics_dir=SAMPLING_RESULTS_DIR, pattern=SAMPLING_GLOB,
                          token_file=TOKEN_BUDGET_FILE):
    """Flatten the sampling trajectories into one row per model and step.

    Each `<stem>_sampling.json` records `draws_per_prompt_total` per step,
    the cumulative draws of every prompt; `total_samples` is that times the
    prompt count `config["n_p"]`, i.e. the cumulative sampled tokens (API
    calls) over all prompts. `relative_error` is the file's `rel_err` of the
    grid-rounded `d_hat`, and `relative_error_raw` the error of the raw
    peak index `aoi_peak`. Steps without an estimate (`d_hat == -1`) carry
    NaN. `vocab_size` comes from model_configs.json. The `*_tokens` columns
    convert calls to prompt tokens with the model's `avg_prompt_tokens`
    from `token_file`, when available.
    """
    rows = []
    paths = sorted(Path(metrics_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {Path(metrics_dir) / pattern}")

    for path, payload in _load_runs(paths, "Sampling").items():
        model = _model_from_path(path)
        config = payload.get("config", {})
        model_config = MODEL_CONFIGS.get(model, {})
        true_dim = config.get("true_dim")
        if true_dim is None or true_dim <= 0:
            true_dim = model_config.get("hidden_size")
        vocab_size = model_config.get("vocab_size")
        n_prompts = config.get("n_p")
        if not n_prompts:
            raise ValueError(f"Missing config.n_p in {path.name}")
        steps = payload.get("steps", [])
        if not steps:
            print(f"[skip] no steps in {path.name}")
            continue

        draws = [step.get("draws_per_prompt_total") for step in steps]
        if any(value is None for value in draws):
            raise ValueError(f"Missing draws_per_prompt_total in {path.name}")
        if any(next_value < value for value, next_value in zip(draws, draws[1:])):
            raise ValueError(f"Non-monotonic draws_per_prompt_total in {path.name}")

        for step in steps:
            d_hat = step.get("d_hat")
            if d_hat is None or d_hat < 0:
                d_hat = np.nan
            peak = step.get("aoi_peak")
            if peak is None or peak < 0:
                peak = np.nan
            rel_err = step.get("rel_err")
            if (rel_err is None or not np.isfinite(rel_err)) and np.isfinite(d_hat) and true_dim:
                rel_err = abs(d_hat - true_dim) / true_dim
            rel_err_raw = abs(peak - true_dim) / true_dim if true_dim and np.isfinite(peak) else np.nan
            rows.append({
                "model": model,
                "architecture": MODEL_TO_ARCHITECTURE.get(model),
                "size_tier": MODEL_TO_TIER.get(model),
                "step": step.get("step"),
                "draws_per_prompt_total": step.get("draws_per_prompt_total"),
                "n_prompts": n_prompts,
                "total_samples": step.get("draws_per_prompt_total") * n_prompts,
                "d_hat": d_hat,
                "aoi_saddle": step.get("aoi_saddle"),
                "aoi_peak": peak,
                "sc_snapback": step.get("sc_snapback"),
                "true_dim": true_dim,
                "vocab_size": vocab_size,
                "relative_error": rel_err,
                "relative_error_raw": rel_err_raw,
                "spectrum_class": step.get("spectrum_class"),
            })

    frame = pd.DataFrame(rows).sort_values(["model", "step"]).reset_index(drop=True)
    numeric = ["step", "draws_per_prompt_total", "n_prompts", "total_samples", "d_hat",
               "aoi_saddle", "aoi_peak", "sc_snapback", "true_dim", "vocab_size",
               "relative_error", "relative_error_raw"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")

    frame["avg_prompt_tokens"] = frame["model"].map(load_prompt_token_averages(token_file))
    for samples_col, tokens_col in TOKEN_COLUMNS.items():
        frame[tokens_col] = frame[samples_col] * frame["avg_prompt_tokens"]
    return frame


# ---------------------------------------------------------------------------
# Figure and axis helpers
# ---------------------------------------------------------------------------

def _save_figure(fig, savepath):
    """Write `fig` to `savepath` (creating the directory) and close it.

    Figures are never shown interactively; with savepath=None the figure is
    only closed, which lets the helpers run for their printed tables alone.
    """
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")
        print(f"Saved figure to {savepath}")
    plt.close(fig)


def _format_per_prompt(value, _position=None):
    """Tick formatter for a total-calls axis shown per prompt: 2e10 -> '1M'."""
    return _format_samples(value / N_PROMPTS if value is not None else None)


def _format_samples(value, _position=None):
    """Tick formatter: 1.2e10 -> '12B', 3.4e8 -> '340M'."""
    if value is None or not np.isfinite(value):
        return ""
    magnitude = abs(value)
    if magnitude >= 1e12:
        return f"{value / 1e12:.1f}T"
    if magnitude >= 1e9:
        return f"{value / 1e9:.0f}B"
    if magnitude >= 1e6:
        return f"{value / 1e6:.0f}M"
    if magnitude >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{value:.0f}"


# Horizontal grid lines.
Y_TICKS = list(range(10, 101, 10))


# ---------------------------------------------------------------------------
# Averaging
# ---------------------------------------------------------------------------

def final_group_relative_errors(data, groups, *, error_col=PLOT_ERROR, budget=None):
    """Summarize final relative error for each plotted average group."""
    x_col = _budget_col(budget)
    final = (data.sort_values(["model", "step"])
                 .groupby("model", as_index=False, sort=False)
                 .tail(1)
                 .set_index("model"))
    rows = []
    for label, models in groups.items():
        values = final.reindex(models)[error_col].dropna() * 100
        if values.empty:
            continue
        end_totals = final.reindex(values.index)[x_col].dropna()
        rows.append({
            "group": label,
            "n_models": len(values),
            "end_budget": end_totals.min(),
            "mean_relative_error_pct": values.mean(),
            "median_relative_error_pct": values.median(),
            "min_relative_error_pct": values.min(),
            "max_relative_error_pct": values.max(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mean relative error figure
# ---------------------------------------------------------------------------

ARCH_LINESTYLE = {"Dense": "-", "MoE": "--"}


def plot_average_relative_error(data, groups, *, error_col=PLOT_ERROR, budget=None,
                                architecture=None, log_x=False, per_prompt=False, figsize=(6, 4),
                                ax=None, title=None, savepath=None):
    """Mean relative error of each group at every recorded budget, one axes.

    `groups` maps curve label -> [model, ...] (the size tiers by default).
    Each marker is the mean over the group's models that have an estimate at
    that budget:

    - `architecture=None` splits every group into a Dense (solid) and a MoE
      (dashed) line sharing the group's color.
    - `"both"` averages Dense and MoE together into one line per group.
    - `"Dense"` / `"MoE"` keeps only that subset.

    `budget` is "calls" or "tokens"; `log_x` puts the budget axis on a log
    scale. `per_prompt` labels a calls axis in API calls per prompt
    (total / N_PROMPTS). The legend sits inside the axes.
    """
    if error_col not in {"relative_error", "relative_error_raw"}:
        raise ValueError("error_col must be 'relative_error' or 'relative_error_raw'")
    if architecture not in (None, "both", "Dense", "MoE"):
        raise ValueError("architecture must be None, 'both', 'Dense' or 'MoE'")
    spec = _budget(budget)
    x_col = _budget_col(budget)
    data = data.dropna(subset=[x_col])
    if architecture not in (None, "both"):
        data = data[data["architecture"] == architecture]
    split_archs = architecture is None
    if per_prompt:
        if spec["short"] != "calls":
            raise ValueError("per_prompt needs budget='calls'")
        if set(data["n_prompts"].unique()) != {N_PROMPTS}:
            raise ValueError(f"per_prompt assumes n_p={N_PROMPTS} for every run, "
                             f"got {sorted(data['n_prompts'].unique())}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    plotted = []        # one mean series per drawn line
    table_groups = {}   # legend label -> models, for the printed summary

    def _plot_series(sub, *, color, linestyle, label):
        if sub.empty:
            return False
        series = sub.groupby(x_col)[error_col].mean().dropna().sort_index()
        if series.empty:
            return False
        ax.plot(series.index.to_numpy(dtype=float), 100 * series.to_numpy(dtype=float),
                marker="o", linestyle=linestyle, color=color, label=label)
        plotted.append(series)
        table_groups[label] = sorted(sub["model"].unique())
        return True

    cmap = plt.get_cmap("tab10" if len(groups) <= 10 else "tab20")
    for i, (label, models) in enumerate(groups.items()):
        sub = data[data["model"].isin(list(models))]
        if sub.empty:
            print(f"No rows for group {label!r}; skipping.")
            continue
        color = cmap(i % cmap.N)
        if split_archs:
            for arch in ("Dense", "MoE"):
                arch_sub = sub[sub["architecture"] == arch]
                _plot_series(arch_sub, color=color, linestyle=ARCH_LINESTYLE[arch],
                             label=f"{label} [{arch}] (n={arch_sub['model'].nunique()})")
        else:
            _plot_series(sub, color=color, linestyle="-",
                         label=f"{label} (n={sub['model'].nunique()})")
    if not plotted:
        raise ValueError("No selected models are present in the data.")

    ax.set_xlabel(PER_PROMPT_LABEL if per_prompt else spec["label"], fontsize=14)
    ax.set_ylabel("Relative Error (%)", fontsize=14)
    if log_x:
        ax.set_xscale("log")
    else:
        # One tick per recorded budget.
        ax.set_xticks(sorted(set().union(*(series.index for series in plotted))))
    ax.set_yticks(Y_TICKS)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_per_prompt if per_prompt else _format_samples))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(True, which="both", alpha=0.3)
    ax.grid(True, axis="y", which="major", alpha=0.3)
    if title:
        ax.set_title(title, fontsize=13)
    # Legend inside the axes, in the empty top-right corner.
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)
    fig.tight_layout()
    _save_figure(fig, savepath)

    final = final_group_relative_errors(data, table_groups, error_col=error_col, budget=budget)
    print(f"\n{title or 'Mean relative error by size tier'} — end-of-sampling relative "
          f"error (end_budget in {spec['unit']})")
    printable = final.copy()
    printable["end_budget"] = printable["end_budget"].map(lambda x: f"{x:,.0f}")
    for col in ["mean_relative_error_pct", "median_relative_error_pct",
                "min_relative_error_pct", "max_relative_error_pct"]:
        printable[col] = printable[col].map(lambda x: f"{x:.4f}%")
    print(printable.to_string(index=False))
    return ax, final


# ---------------------------------------------------------------------------
# Sampling vs. Carlini cost-overhead ratio
# ---------------------------------------------------------------------------

def carlini_api_calls(true_dim, vocab_size, k=1):
    """Number of API calls a Carlini-style top-K logit extraction needs.

    ``calls = true_dim * vocab_size``            (K == 1)
    ``calls = true_dim * vocab_size / (K - 1)``  (K  > 1)
    """
    if true_dim is None or vocab_size is None:
        return np.nan
    if not (np.isfinite(true_dim) and np.isfinite(vocab_size)):
        return np.nan
    if k <= 1:
        return float(true_dim) * float(vocab_size)
    return float(true_dim) * float(vocab_size) / (k - 1)


def carlini_call_ratio_frame(data, *, k=1, models=None, architecture=None,
                             error_col=PLOT_ERROR, err_threshold=0.3,
                             reduce="min_n", calls_col=CALLS_COL, budget=None,
                             carlini_tokens_per_call=CARLINI_TOKENS_PER_CALL):
    """Sampling / Carlini cost ratio at a fixed accuracy target.

    `budget` picks the cost unit ("calls" or "tokens"). `sampling_cost`,
    `carlini_cost`, and `cost_ratio` are in that unit; `api_calls`,
    `carlini_calls`, and `call_ratio` are always kept as well.

    Keeps only the sampling steps that reach relative error
    ``<= err_threshold`` so every row is iso-accuracy, then (with
    ``reduce="min_n"``) collapses each model to the cheapest such step.
    ``reduce="all"`` keeps every qualifying step instead.
    """
    if error_col not in {"relative_error", "relative_error_raw"}:
        raise ValueError("error_col must be 'relative_error' or 'relative_error_raw'")
    cost_col = _budget_col(budget, calls_col)
    needed = list(dict.fromkeys(["true_dim", "vocab_size", calls_col, cost_col, error_col]))
    missing = [column for column in needed if column not in data.columns]
    if missing:
        print(f"Missing required columns: {missing}.")
        return pd.DataFrame()

    frame = data.copy()
    if models:
        frame = frame[frame["model"].isin(list(models))]
    if architecture not in (None, "both"):
        frame = frame[frame["architecture"] == architecture]
    frame = frame.dropna(subset=needed)
    if frame.empty:
        print("No rows after filtering.")
        return frame

    frame = frame[frame[error_col].astype(float) <= err_threshold]
    if frame.empty:
        print(f"No sampling steps with relative error <= {err_threshold}.")
        return frame

    if reduce in ("min_n", "min_calls"):
        frame = frame.loc[frame.groupby("model")[calls_col].idxmin()]
    elif reduce not in ("all", None):
        raise ValueError(f"reduce must be 'min_n' or 'all', got {reduce!r}")

    frame = frame.copy()
    frame["carlini_calls"] = [
        carlini_api_calls(true_dim, vocab_size, k)
        for true_dim, vocab_size in zip(frame["true_dim"], frame["vocab_size"])
    ]
    frame["api_calls"] = frame[calls_col].astype(float)
    frame["call_ratio"] = frame["api_calls"] / frame["carlini_calls"]
    frame["sampling_cost"] = frame[cost_col].astype(float)
    if cost_col == calls_col:
        frame["carlini_cost"] = frame["carlini_calls"]
    else:
        tokens_per_call = (frame["avg_prompt_tokens"] if carlini_tokens_per_call is None
                           else float(carlini_tokens_per_call))
        frame["carlini_cost"] = frame["carlini_calls"] * tokens_per_call
    frame["cost_ratio"] = frame["sampling_cost"] / frame["carlini_cost"]
    return frame.sort_values(["true_dim", "model"]).reset_index(drop=True)


def _target_label(target, n_models=None):
    """Legend/tick label for one accuracy target."""
    base = "exact (0%)" if target <= 0 else f"≤{100 * target:g}% err"
    return base if n_models is None else f"{base} (n={n_models})"


# Labels for the aggregations `_aggregate_ratios` supports.
AGG_LABELS = {"mean": "mean", "geomean": "geomean", "median": "median"}


def _aggregate_ratios(values, agg):
    """Central value for a set of ratios. Arithmetic mean by default."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array) & (array > 0)]
    if not array.size:
        return np.nan
    if agg == "mean":
        return float(np.mean(array))
    if agg == "geomean":
        return float(np.exp(np.mean(np.log(array))))
    if agg == "median":
        return float(np.median(array))
    raise ValueError(f"agg must be 'mean', 'geomean', or 'median', got {agg!r}")


def call_ratio_by_target(data, targets=ACCURACY_TARGETS, *, k=1, models=None,
                         architecture=None, error_col=PLOT_ERROR,
                         reduce="min_n", calls_col=CALLS_COL, budget=None,
                         carlini_tokens_per_call=CARLINI_TOKENS_PER_CALL):
    """One row per (accuracy target, model) with its cost-overhead ratio.

    Runs `carlini_call_ratio_frame` once per target and stacks the
    results, so each target keeps only the models that actually reach it
    and the cheapest step at which they do. Model coverage therefore
    shrinks as the target tightens.
    """
    frames = []
    for target in targets:
        frame = carlini_call_ratio_frame(
            data, k=k, models=models, architecture=architecture,
            error_col=error_col, err_threshold=target,
            reduce=reduce, calls_col=calls_col, budget=budget,
            carlini_tokens_per_call=carlini_tokens_per_call)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["target"] = target
        frame["target_label"] = _target_label(target)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _group_values(frame, group_by):
    """Ordered group keys for the requested x-axis."""
    if group_by == "true_dim":
        return sorted(frame["true_dim"].dropna().unique())
    if group_by == "size_tier":
        present = set(frame["size_tier"].dropna().unique())
        return [tier for tier in TIER_ORDER if tier in present]
    if group_by == "model":
        return sorted(frame["model"].unique())
    raise ValueError("group_by must be 'true_dim', 'size_tier', or 'model'")


def _print_target_table(frame, targets, group_by, agg, budget=None):
    """Print the aggregated ratios and model counts behind a figure."""
    if group_by is None:
        rows = []
        for target in targets:
            sub = frame[frame["target"] == target]
            if sub.empty:
                continue
            ratios = sub["cost_ratio"].to_numpy(dtype=float)
            rows.append({
                "target": _target_label(target),
                "n_models": len(ratios),
                f"{agg}_ratio": f"{_aggregate_ratios(ratios, agg):.4g}x",
                "p25": f"{np.percentile(ratios, 25):.4g}x",
                "p75": f"{np.percentile(ratios, 75):.4g}x",
                "min": f"{ratios.min():.4g}x",
                "max": f"{ratios.max():.4g}x",
            })
        print(pd.DataFrame(rows).to_string(index=False))
        return

    groups = _group_values(frame, group_by)
    ratio_table, count_table = {}, {}
    for target in targets:
        sub = frame[frame["target"] == target]
        if sub.empty:
            continue
        label = _target_label(target)
        ratio_table[label] = {
            group: _aggregate_ratios(
                sub.loc[sub[group_by] == group, "cost_ratio"], agg)
            for group in groups
        }
        count_table[label] = {
            group: int((sub[group_by] == group).sum()) for group in groups
        }
    singular = _budget(budget)["singular"]
    print(f"{AGG_LABELS.get(agg, agg)} {singular} ratio by {group_by}:")
    print(pd.DataFrame(ratio_table).reindex(groups).round(2).to_string())
    print(f"\nmodels contributing by {group_by}:")
    print(pd.DataFrame(count_table).reindex(groups).to_string())


def plot_call_ratio_by_target(data, targets=ACCURACY_TARGETS, *, kind="line",
                              agg="mean", k=1, models=None, architecture=None,
                              error_col=PLOT_ERROR, reduce="min_n",
                              calls_col=CALLS_COL, budget=None,
                              carlini_tokens_per_call=CARLINI_TOKENS_PER_CALL,
                              group_by="true_dim",
                              spread=True, log_y=False, cmap="viridis",
                              figsize=(6, 4), ax=None, title=None,
                              savepath=None):
    """Cost-overhead ratio with one line or bar per accuracy target.

    `budget` is "calls" or "tokens". Three variations, selected by `kind`:

    - ``"line"``    — x is `group_by` (hidden dimension by default), one
      line per accuracy target.
    - ``"bar"``     — the same numbers as grouped bars, one bar per
      accuracy target within each group.
    - ``"summary"`` — one bar per accuracy target over all models, with
      whiskers at the inter-quartile range. `group_by` is ignored.

    `agg` is the arithmetic mean (default), the geometric mean, or the
    median. The mean is sensitive to the heavy right tail of these
    ratios, so the printed table carries the quartiles and the max
    alongside it. Coverage shrinks as the target tightens, so every
    label carries the number of models behind it.
    """
    spec = _budget(budget)
    frame = call_ratio_by_target(
        data, targets, k=k, models=models, architecture=architecture,
        error_col=error_col, reduce=reduce, calls_col=calls_col, budget=budget,
        carlini_tokens_per_call=carlini_tokens_per_call)
    if frame.empty:
        print("No models reach any of the requested accuracy targets.")
        return None, frame

    targets = [target for target in targets if (frame["target"] == target).any()]
    colors = plt.get_cmap(cmap)(np.linspace(0.12, 0.88, len(targets)))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if kind == "summary":
        positions = np.arange(len(targets))
        heights, lows, highs, counts = [], [], [], []
        for target in targets:
            ratios = frame.loc[frame["target"] == target, "cost_ratio"].to_numpy(float)
            center = _aggregate_ratios(ratios, agg)
            heights.append(center)
            lows.append(max(center - np.percentile(ratios, 25), 0))
            highs.append(max(np.percentile(ratios, 75) - center, 0))
            counts.append(len(ratios))
        ax.bar(positions, heights, width=0.62, color=colors,
               edgecolor="black", linewidth=0.4,
               yerr=[lows, highs], capsize=4,
               error_kw={"elinewidth": 1.0, "ecolor": "0.3"})
        for position, height, count in zip(positions, heights, counts):
            ax.annotate(f"{height:.4g}x\nn={count}", (position, height + highs[position]),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=13)
        ax.set_xticks(positions)
        ax.set_xticklabels([_target_label(target).replace(" ", "\n", 1)
                            for target in targets])
        ax.set_xlabel("Accuracy Target", fontsize=17)
        ax.margins(y=0.18)
    else:
        groups = _group_values(frame, group_by)
        positions = np.arange(len(groups))
        width = 0.8 / max(len(targets), 1)
        for index, (target, color) in enumerate(zip(targets, colors)):
            sub = frame[frame["target"] == target]
            values = np.array([
                _aggregate_ratios(sub.loc[sub[group_by] == group, "cost_ratio"], agg)
                for group in groups
            ], dtype=float)
            label = _target_label(target, int((frame["target"] == target).sum()))
            if kind == "line":
                ax.plot(positions, values, marker="o", linewidth=1.8,
                        color=color, label=label)
                if spread and group_by != "model":
                    counts = np.array([
                        int(((sub[group_by] == group)).sum()) for group in groups])
                    ax.scatter(positions[counts == 1], values[counts == 1],
                               s=18, color=color, edgecolors="white",
                               linewidths=0.6, zorder=3)
            elif kind == "bar":
                offset = (index - (len(targets) - 1) / 2) * width
                ax.bar(positions + offset, np.nan_to_num(values), width=width,
                       color=color, edgecolor="black", linewidth=0.4, label=label)
            else:
                raise ValueError("kind must be 'line', 'bar', or 'summary'")
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{group:,.0f}" if group_by == "true_dim" else str(group)
                            for group in groups])
        ax.set_xlabel({"true_dim": "Hidden Dimension d",
                       "size_tier": "Size Tier",
                       "model": "Model"}[group_by], fontsize=17)
        if group_by == "model":
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend(fontsize=9.5, frameon=True, title="Accuracy target",
                  title_fontsize=9.5)

    if log_y:
        ax.set_yscale("log")
    else:
        # Ticks at multiples of 10 up to the tallest bar.
        tallest = max((bar.get_height() for bar in ax.patches), default=0.0)
        if np.isfinite(tallest) and tallest > 0:
            top = int(np.ceil(tallest / 10.0)) * 10
            ax.set_yticks(np.arange(0, top + 1, 10))
    ax.axhline(1.0, color="0.4", linewidth=1.0, linestyle=":", zorder=0)
    ax.set_ylabel(f"{spec['singular'].title()} Overhead Ratio "
                  f"({AGG_LABELS.get(agg, agg)})", fontsize=15.5)
    if title:
        ax.set_title(title, fontsize=13)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    ax.set_axisbelow(True)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.figure.tight_layout()
    _save_figure(ax.figure, savepath)

    print(f"\nSampling vs. Carlini top-{k} {spec['unit']} — {AGG_LABELS.get(agg, agg)} {spec['singular']} ratio "
          "by accuracy target")
    _print_target_table(frame, targets, None if kind == "summary" else group_by, agg,
                        budget=budget)
    return ax, frame


# ---------------------------------------------------------------------------
# Timing-attack figures
# ---------------------------------------------------------------------------

# The 14 timing models: price tier and size group. The average prompt tokens
# per API call are filled in from TOKEN_BUDGET_FILE below.
_TIMING_MODEL_TIERS = {
    "ibm-granite/granite-3.3-2b-instruct": "small",
    "ibm-granite/granite-3.3-8b-instruct": "large",
    "ibm-granite/granite-4.0-micro": "mid",
    "meta-llama/Llama-3.1-8B-Instruct": "large",
    "meta-llama/Llama-3.2-1B-Instruct": "small",
    "meta-llama/Llama-3.2-3B-Instruct": "mid",
    "allenai/OLMo-2-0425-1B-Instruct": "small",
    "allenai/OLMo-2-1124-7B-Instruct": "large",
    "Qwen/Qwen2.5-1.5B-Instruct": "small",
    "Qwen/Qwen2.5-3B-Instruct": "mid",
    "Qwen/Qwen2.5-7B-Instruct": "large",
    "Qwen/Qwen3-1.7B": "small",
    "Qwen/Qwen3-4B": "mid",
    "Qwen/Qwen3-8B": "large",
}
with TOKEN_BUDGET_FILE.open() as _f:
    _AVG_PROMPT_TOKENS = {e["model"]: float(e["avg_prompt_tokens"]) for e in json.load(_f)["models"]}
MODELS = {model: (tier, _AVG_PROMPT_TOKENS[model]) for model, tier in _TIMING_MODEL_TIERS.items()}
# USD per million tokens, from the paper's cost estimate.
TIER_PRICING = {
    "small": {"input": 0.15, "output": 0.60},  # GPT-4o mini
    "mid": {"input": 0.75, "output": 4.50},  # GPT-4.1 mini
    "large": {"input": 2.50, "output": 15.00},  # GPT-4.1
}

# Same-machine fits behind each error panel: (legend label, JSON stem).
L_FITS = (("True $d$", "layers"), (r"Spectral $\hat{d}$", "layers_dhat"))
P_FITS = (("True $d$", "params"), (r"Spectral $\hat{d}$", "params_dhat"))
FAMILY_ORDER = ("ibm-granite/", "meta-llama/", "allenai/", "Qwen/")
ERROR_CEILING = 120  # % on the budget figures; the lowest-budget averages of L can run past it
MAX_BUDGET_TICKS = 12  # one x tick per budget up to this many, else evenly spaced ticks

# Series colors, assigned in this order.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
# Cycled over the models of the global fit figure.
GLOBAL_FIT_MARKERS = ("o", "s", "^", "D", "v", "<", ">", "P", "*", "h", "X", "8")


# --- loading ----------------------------------------------------------------

def trimmed_stats(samples) -> tuple[float, float]:
    """Trimmed mean and its standard error std/sqrt(N_kept)."""
    x = np.sort(np.asarray(samples, dtype=float))
    cut = int(len(x) * TRIM_FRAC)
    x = x[cut:len(x) - cut]
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def prefill_points(rec: dict) -> dict:
    """Baseline-subtracted prefill latency per length."""
    by_len = sorted(rec["measurements"], key=lambda m: m["prompt_tokens"])
    baseline, _ = trimmed_stats(by_len[0]["latencies"])  # l=1 entry (shortest prompt)
    points = by_len[1:]
    stats = [trimmed_stats(m["latencies"]) for m in points]
    return {
        "ell": np.array([m["prompt_tokens"] for m in points], dtype=float),
        "T": np.array([s[0] - baseline for s in stats]),
        "sigma": np.array([max(s[1], 1e-9) for s in stats]),
    }


def load_models(directory: Path) -> list[dict]:
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
            **prefill_points(rec),
        }
    return [models[k] for k in sorted(models)]


def read_results(path: Path) -> list[dict]:
    """The per-model records of one fit JSON."""
    if not path.is_file():
        raise SystemExit(f"missing fit file: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["results"]


def load_fit(fits_dir: Path, stem: str) -> dict[str, dict]:
    """{model: record} from timing_fit_<stem>.json, with the estimate fields made top-level."""
    path = fits_dir / f"timing_fit_{stem}.json"
    fit = {}
    for rec in read_results(path):
        if "prefill" not in rec["estimates"]:
            raise SystemExit(f"{path.name}: no prefill estimate for {rec['model']}")
        fit[rec["model"]] = {**rec, **rec["estimates"]["prefill"]}
    return fit


def load_cross(cross_dir: Path) -> dict[tuple[str, str, str, str], dict[str, dict]]:
    """{(fit, d source, reference, target): {model: record}} for every cross-platform JSON."""
    pattern = re.compile(r"timing_cross_(.+)_to_(.+)_(layers|params)(_dhat)?\.json")
    runs = {}
    for path in sorted(cross_dir.glob("timing_cross_*.json")):
        match = pattern.fullmatch(path.name)
        if match:
            ref, tgt, fit, dhat = match.groups()
            key = (fit, "d_hat" if dhat else "d_star", ref, tgt)
            runs[key] = {r["model"]: r for r in read_results(path)}
    return runs


def load_timing_models(prefill_dir: Path) -> list[dict]:
    """The prefill points of the 14 timing models."""
    models = [m for m in load_models(prefill_dir) if m["model"] in MODELS]
    if {m["model"] for m in models} != set(MODELS):
        raise SystemExit(f"{prefill_dir}: missing {sorted(set(MODELS) - {m['model'] for m in models})}")
    return models


def load_budget_fits(budget_dir: Path, models: list[dict], fits: dict) -> dict[str, dict[str, np.ndarray]]:
    """{model: {calls, tokens, cost, d, L, P}} with errors in %, one entry per fitted budget."""
    index = budget_dir / "budgets.json"
    if not index.is_file():
        raise SystemExit(f"missing {index}; run experiments/nightvision_timing/fit_timing_budgets.py")
    with open(index, encoding="utf-8") as f:
        budgets = sorted(json.load(f)["fitted"], key=lambda b: b["calls"])
    if not budgets:
        raise SystemExit(f"{index} lists no fitted budgets")
    rows = {m["model"]: [] for m in models}
    for budget in budgets:
        layers = load_fit(budget_dir / budget["dir"], "layers_dhat")
        params = load_fit(budget_dir / budget["dir"], "params_dhat")
        if set(layers) != set(MODELS) or set(params) != set(MODELS):
            raise SystemExit(f"{budget['dir']}: the fits do not cover the {len(MODELS)} timing models")
        for model, model_rows in rows.items():
            model_rows.append((budget["calls"], rel_err(layers[model]["d_hat"], layers[model]["d"]),
                               abs(layers[model]["L_rel_err"]) * 100, abs(params[model]["P_rel_err"]) * 100))
    # The largest budget is the end of sampling, which results/timing_fits/ also holds.
    for model in rows:
        check_full_budget(model, {"L": layers[model]["L_hat"], "P": params[model]["P_hat"]},
                          {"L": fits["layers_dhat"], "P": fits["params_dhat"]}, "query-budget")
    curves = {}
    for model, model_rows in rows.items():
        tier, avg_tokens = MODELS[model]
        rate = TIER_PRICING[tier]
        calls, d_err, l_err, p_err = map(np.array, zip(*model_rows))
        tokens = calls * avg_tokens
        curves[model] = {"calls": calls.astype(float), "tokens": tokens,
                         "cost": tokens / 1e6 * rate["input"] + calls / 1e6 * rate["output"],
                         "d": d_err, "L": l_err, "P": p_err}
    return curves


def load_d_curve(sampling_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per recorded budget: API calls, mean d error in % and the models behind that mean.

    A model with no d_hat yet (d_hat = -1) is left out of the mean at that budget.
    """
    stems = {hf_id: stem for stem, hf_id in MODEL_HF_IDS.items()}
    errors: dict[int, list[float]] = {}
    for model in MODELS:
        path = sampling_dir / f"{stems[model]}_sampling.json"
        if not path.is_file():
            raise SystemExit(f"missing sampling file: {path}")
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        for step in run["steps"]:
            if step["d_hat"] > 0:
                calls = step["draws_per_prompt_total"] * run["config"]["n_p"]
                errors.setdefault(calls, []).append(rel_err(step["d_hat"], run["config"]["true_dim"]))
    budgets = sorted(errors)
    return (np.array(budgets, dtype=float), np.array([np.mean(errors[b]) for b in budgets]),
            np.array([len(errors[b]) for b in budgets]))


# --- sweeps -----------------------------------------------------------------

def rel_err(hat: float, true: float) -> float:
    """Absolute relative error in %."""
    return abs(hat / true - 1) * 100


def check_full_budget(model: str, estimates: dict, saved: dict[str, dict[str, dict]], sweep: str) -> None:
    """The full-budget sweep point must be the estimate already saved in the fit JSONs."""
    for quantity, stem in saved.items():
        hat, kept = estimates[quantity], stem[model][f"{quantity}_hat"]
        if not np.isclose(hat, kept, rtol=1e-9):
            raise SystemExit(f"{sweep} sweep: {quantity}_hat of {model} is {hat}, the saved fit has {kept}. "
                             "Are the fits, prefill and sampling results from the same runs?")


def step_average(curves: list[dict], x_key: str, quantity: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean budget and mean error per budget step over the given models.

    Every model has the same steps, so a step is the same effort for all of them.
    Interpolating onto a shared token or USD axis instead would change which
    models are averaged along the curve, since prompt length and price differ.
    """
    return (np.mean([c[x_key] for c in curves], axis=0), np.mean([c[quantity] for c in curves], axis=0))


# --- drawing ----------------------------------------------------------------

def model_order(records: dict[str, dict]) -> list[str]:
    """Models by family (same order as the paper's bar figures), then size."""
    def key(model: str) -> tuple:
        family = next((i for i, f in enumerate(FAMILY_ORDER) if model.startswith(f)), len(FAMILY_ORDER))
        return family, records[model]["P"], model
    return sorted(records, key=key)


def short_name(model: str) -> str:
    """Axis label: the repository name without its organization or instruct suffix."""
    name = model.split("/")[-1]
    for suffix in ("-Instruct", "-instruct"):
        name = name.removesuffix(suffix)
    return name


def draw_error_bars(ax, labels: list[str], series: list[tuple[str, np.ndarray, str]], ylabel: str) -> None:
    """Grouped per-model error bars in %, on a linear axis to 100%, or to the next 10 above a larger bar."""
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 2)
    for i, (label, values, color) in enumerate(series):
        ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width, label=label, color=color)
    top = max(100, int(np.ceil(max(float(np.max(values)) for _, values, _ in series) / 10)) * 10)
    ax.set_yticks(range(10, top + 1, 10))
    ax.set_ylim(0, top)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.legend(fontsize=11, loc="upper left", bbox_to_anchor=(0.015, 0.99), framealpha=0.92, ncol=len(series))
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-0.5, len(labels) - 0.5)


def save_error_bars(fig, axes, title: str, out: Path) -> None:
    """Bold title, panels stacked with almost no gap."""
    axes[0].set_title(title, fontsize=18, fontweight="bold")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def bar_labels(fits: dict, models: list[str]) -> list[str]:
    return [f"{short_name(m)}\n(d={fits['layers'][m]['d']})" for m in models]


def rel_err_pct(fit: dict[str, dict], models: list[str], quantity: str) -> np.ndarray:
    """Absolute relative error (%) of L or P for each model, in order."""
    return np.array([abs(fit[m][f"{quantity}_rel_err"]) * 100 for m in models])


# --- figures ----------------------------------------------------------------

def plot_per_model(fits: dict[str, dict], models: list[str], out: Path) -> None:
    """L, P and d error per model for the same-machine fits."""
    labels = bar_labels(fits, models)
    d_err = np.array([abs(fits["layers_dhat"][m]["d_hat"] / fits["layers_dhat"][m]["d"] - 1) * 100 for m in models])
    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(models) * 0.9), 13), sharex=True)
    for ax, group, quantity in ((axes[0], L_FITS, "L"), (axes[1], P_FITS, "P")):
        series = [(label, rel_err_pct(fits[stem], models, quantity), color)
                  for (label, stem), color in zip(group, SERIES_COLORS)]
        draw_error_bars(ax, labels, series, f"{quantity} relative error (%)")
    # d_hat keeps the color it has in the panels above.
    draw_error_bars(axes[2], labels, [(r"Spectral $\hat{d}$", d_err, SERIES_COLORS[1])], "d relative error (%)")
    save_error_bars(fig, axes, "Per-model error", out)


def plot_cross(runs: dict, directions: list[tuple[str, str]], d_source: str,
               models: list[str], labels: list[str], title: str, out: Path) -> None:
    """L and P error per model, one bar per reference -> target direction."""
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(models) * 0.9), 10), sharex=True)
    for ax, fit, quantity, colors in ((axes[0], "layers", "L", ("#1a4a9e", "#7bafd4")),
                                      (axes[1], "params", "P", ("#8b4513", "#d2956a"))):
        series = [(f"{ref.upper()} → {tgt.upper()}", rel_err_pct(runs[fit, d_source, ref, tgt], models, quantity),
                   color) for (ref, tgt), color in zip(directions, colors)]
        draw_error_bars(ax, labels, series, f"{quantity} relative error (%)")
    save_error_bars(fig, axes, title, out)


def attention_feature(m: dict) -> np.ndarray:
    """L l^2 d, the x axis of the global fit figure."""
    return m["L"] * m["ell"] ** 2 * m["d"]


def weighted_line_fit(x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of y on x by least squares with weights 1/sigma."""
    X = np.column_stack([x, np.ones_like(y)])
    scale = np.abs(X).max(axis=0)  # rescaled to unit maximum before solving
    coef, *_ = np.linalg.lstsq(X / scale / sigma[:, None], y / sigma, rcond=None)
    slope, intercept = coef / scale
    return float(slope), float(intercept)


def plot_global_fit(models: list[dict], out: Path) -> None:
    """Measured time against L l^2 d, with each model's own least-squares fit dashed.

    The lines are unweighted: 1/sigma weights favor the short prompts, whose small
    spread pulls the line above the points at long ones. Models take consecutive tab20 colors in name order, which keeps Qwen3.5 in the
    red, pink and purples that the paper's caption names.
    """
    plt.rcParams.update({"mathtext.fontset": "cm"})
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, m in enumerate(sorted(models, key=lambda m: m["model"])):
        color = cmap(i)
        x = attention_feature(m)
        t_us = m["T"] * 1e6
        ax.scatter(x, t_us, s=34, color=color, marker=GLOBAL_FIT_MARKERS[i % len(GLOBAL_FIT_MARKERS)],
                   label=m["model"].split("/")[-1], zorder=3)
        slope, intercept = np.polyfit(x, t_us, 1)
        ax.plot(x, slope * x + intercept, "--", color=color, linewidth=1.1, alpha=0.75, zorder=2)
    ax.set_ylabel(r"T ($\mu$s)", fontsize=14)
    
    ax.set_xlabel(r"$L \ell^{2} d$", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(14)
    ax.yaxis.get_offset_text().set_fontsize(14)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9, labelspacing=0.3, borderpad=0.4, handletextpad=0.4)
    save_budget_figure(fig, out)
    beta, alpha = weighted_line_fit(*(np.concatenate([f(m) for m in models])
                                      for f in (attention_feature, lambda m: m["T"], lambda m: m["sigma"])))
    print(f"  global fit: beta={beta:.4g}, alpha={alpha:.4g}")


def draw_averages(ax, lines: list[tuple[str, np.ndarray, np.ndarray]], xlabel: str, ylabel: bool = True,
                  x_format=_format_samples, square: bool = False) -> None:
    """Mean-error lines in the style of plot_average_relative_error."""
    cmap = plt.get_cmap("tab10")
    for i, (label, x, y) in enumerate(lines):
        ax.plot(x, y, marker="o", linestyle="-", color=cmap(i), label=label)
    ax.set_xlabel(xlabel, fontsize=14)
    if ylabel:
        ax.set_ylabel("Relative Error (%)", fontsize=14)
    budgets = sorted({float(v) for _, x, _ in lines for v in x})
    if len(budgets) <= MAX_BUDGET_TICKS:
        ax.set_xticks(budgets)  # one tick per recorded budget
    else:  # the series sit at different budgets; no 2.5 step, which the formatter would round
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
    ax.set_yticks(Y_TICKS)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(x_format))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)
    if square:
        ax.set_box_aspect(1)


def d_l_p_lines(curves: dict, x_key: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """One averaged line per estimated quantity."""
    return [(f"{label} (n={len(curves)})", *step_average(list(curves.values()), x_key, quantity))
            for quantity, label in (("d", "Hidden size $d$"), ("L", "Depth $L$"), ("P", "Parameter count $P$"))]


def save_budget_figure(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure_2(curves: dict, d_curve: tuple[np.ndarray, np.ndarray, np.ndarray], out: Path) -> None:
    """d error at every sampling budget, L and P error at the budgets with a fit, against API calls per prompt."""
    calls, d_err, counts = d_curve
    n = f"{counts.min()}–{counts.max()}" if counts.min() != counts.max() else f"{counts.max()}"
    lines = [(f"Hidden size $d$ (n={n})", calls, d_err), *d_l_p_lines(curves, "calls")[1:]]
    fig, ax = plt.subplots(figsize=(6, 5))
    draw_averages(ax, lines, PER_PROMPT_LABEL, x_format=_format_per_prompt, square=False)
    save_budget_figure(fig, out)


def plot_cost_tokens(curves: dict, out: Path) -> None:
    """The same averages against API calls per prompt, total estimated USD and total prompt tokens."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    draw_averages(axes[0], d_l_p_lines(curves, "calls"), PER_PROMPT_LABEL, x_format=_format_per_prompt)
    draw_averages(axes[1], d_l_p_lines(curves, "cost"), "Budget (Estimated USD)", ylabel=False,
                  x_format=lambda v, _=None: f"${v / 1e3:.0f}K")
    draw_averages(axes[2], d_l_p_lines(curves, "tokens"), BUDGETS["tokens"]["label"], ylabel=False)
    save_budget_figure(fig, out)


def plot_by_group(curves: dict, out: Path) -> None:
    """P and L error against API calls per prompt per size group, with the d_hat of each budget."""
    groups = {tier: [m for m in curves if MODELS[m][0] == tier] for tier in ("small", "mid", "large")}
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.6), sharex=True)
    for ax, quantity, panel in zip(axes, ("P", "L"), ("Parameter count $P$", "Depth $L$")):
        lines = [(f"{tier} (n={len(members)})", *step_average([curves[m] for m in members], "calls", quantity))
                 for tier, members in groups.items()]
        draw_averages(ax, lines, PER_PROMPT_LABEL, x_format=_format_per_prompt)
        ax.set_title(rf"{panel}, spectral $\hat{{d}}$", fontsize=13)
    axes[0].set_xlabel("")
    save_budget_figure(fig, out)


# --- driver -----------------------------------------------------------------

def print_table(title: str, models: list[str], columns: dict[str, np.ndarray]) -> None:
    """The values behind one set of bars, then their means over all the models given."""
    print(f"\n{title}\n{'model':34s} " + " ".join(f"{c:>12s}" for c in columns))
    for i, model in enumerate(models):
        print(f"{short_name(model):34s} " + " ".join(f"{v[i]:11.1f}%" for v in columns.values()))
    print(f"{'mean':34s} " + " ".join(f"{v.mean():11.1f}%" for v in columns.values()))


def print_sweep(curves: dict) -> None:
    """Mean error per fitted budget."""
    print(f"\nQuery budget: mean error over the {len(curves)} models\n"
          f"{'calls (B)':>10s} {'d':>8s} {'L':>8s} {'P':>8s}")
    means = {q: step_average(list(curves.values()), "calls", q) for q in ("d", "L", "P")}
    for i, calls in enumerate(means["d"][0]):
        print(f"{calls / 1e9:10g} " + " ".join(f"{means[q][1][i]:7.1f}%" for q in ("d", "L", "P")))


def print_d_curve(d_curve: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """The d line of figure_2.png, with the models behind each mean."""
    print(f"\nd error at every sampling budget\n{'calls (B)':>10s} {'d':>8s} {'n':>4s}")
    for calls, err, n in zip(*d_curve):
        print(f"{calls / 1e9:10g} {err:7.1f}% {n:4d}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--figure", choices=("error", "ratio", "timing", "all"), default="all",
                        help="which figures to draw (default: all)")
    parser.add_argument("--sampling-dir", type=Path, default=SAMPLING_RESULTS_DIR, metavar="DIR",
                        help=f"run_sampling.py output, {SAMPLING_GLOB} (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=FIGS_DIR, help="Figure directory (default: %(default)s)")

    sampling = parser.add_argument_group("error and ratio figures")
    sampling.add_argument("--budget", choices=sorted(BUDGETS), default=None,
                          help="cost unit for the ratio figure (default: calls); "
                               "the error figure always uses calls")
    sampling.add_argument("--token-file", default=TOKEN_BUDGET_FILE, metavar="JSON",
                          help="avg_prompt_tokens.json for token budgets (default: "
                               f"{TOKEN_BUDGET_FILE}; skipped when missing)")

    error = parser.add_argument_group("error figure")
    error.add_argument("--architecture", choices=("split", "both", "Dense", "MoE"),
                       default="split",
                       help="split: Dense solid and MoE dashed per tier (default); "
                            "both: one line per tier; Dense/MoE: that subset only")
    error.add_argument("--xscale", choices=("linear", "log"), default="linear")
    error.add_argument("--save-error", default=None, metavar="PATH",
                       help=f"output file (default: {DEFAULT_ERROR_FIGURE} in --output-dir)")

    ratio = parser.add_argument_group("ratio figure")
    ratio.add_argument("--kind", choices=("summary", "line", "bar"), default="summary",
                       help="summary: one bar per accuracy target (default)")
    ratio.add_argument("--agg", choices=sorted(AGG_LABELS), default="mean",
                       help="how ratios are aggregated (default: mean)")
    ratio.add_argument("--group-by", choices=("true_dim", "size_tier", "model"),
                       default="true_dim", help="x-axis for kind=line/bar")
    ratio.add_argument("--k", type=int, default=1, help="Carlini top-K (default: 1)")
    ratio.add_argument("--save-ratio", default=None, metavar="PATH",
                       help=f"output file (default: {DEFAULT_RATIO_FIGURE} in --output-dir)")

    timing = parser.add_argument_group("timing figures")
    timing.add_argument("--fits-dir", type=Path, default=FITS_DIR, help="fit_timing.py output (default: %(default)s)")
    timing.add_argument("--cross-dir", type=Path, default=CROSS_DIR,
                        help="fit_timing_cross.py output (default: %(default)s)")
    timing.add_argument("--prefill-dir", type=Path, default=PREFILL_DIR,
                        help="collect_prefill.py output (default: %(default)s)")
    timing.add_argument("--qwen35-dir", type=Path, default=QWEN35_DIR,
                        help="collect_prefill.py output for Qwen3.5 (default: %(default)s)")
    timing.add_argument("--budget-fits-dir", type=Path, default=BUDGET_FITS_DIR,
                        help="fit_timing_budgets.py output (default: %(default)s)")
    return parser


def generate_sampling_figures(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """The error and ratio figures, as selected by --figure."""
    error_figure = args.save_error or args.output_dir / DEFAULT_ERROR_FIGURE
    ratio_figure = args.save_ratio or args.output_dir / DEFAULT_RATIO_FIGURE

    df = load_sampling_metrics(args.sampling_dir, token_file=args.token_file)
    print(f"Loaded {df['model'].nunique()} models and {len(df):,} sampling steps "
          f"from {args.sampling_dir}/")
    have_tokens = df["avg_prompt_tokens"].notna()
    tokens_available = bool(have_tokens.any())
    if tokens_available:
        missing_tokens = sorted(df.loc[~have_tokens, "model"].unique())
        if missing_tokens:
            print(f"[warn] no avg_prompt_tokens in {args.token_file} for: {missing_tokens}; "
                  "add them to MODEL_HF_IDS to include these models in token-budget plots")
    elif args.budget == "tokens":
        parser.error(f"--budget tokens needs prompt token averages; none at {args.token_file}")

    if args.figure in ("error", "all"):
        plot_average_relative_error(
            df,
            TIERS,
            budget="calls",
            architecture=None if args.architecture == "split" else args.architecture,
            log_x=args.xscale == "log",
            per_prompt=True,
            savepath=error_figure,
        )

    if args.figure in ("ratio", "all"):
        plot_call_ratio_by_target(
            df,
            budget=args.budget or "calls",
            kind=args.kind,
            agg=args.agg,
            k=args.k,
            group_by=args.group_by,
            models=ALL_MODELS,
            savepath=ratio_figure,
        )



def generate_timing_figures(args: argparse.Namespace) -> None:
    """Every timing-attack figure, then the values behind them."""
    fits = {stem: load_fit(args.fits_dir, stem) for _, stem in L_FITS + P_FITS}
    runs = load_cross(args.cross_dir)
    directions = sorted({(ref, tgt) for _, _, ref, tgt in runs})
    wanted = {(fit, d, ref, tgt) for fit in ("layers", "params") for d in ("d_star", "d_hat") for ref, tgt in directions}
    if len(directions) != 2 or wanted != set(runs):
        raise SystemExit(f"expected every fit and d source in two directions in {args.cross_dir}, "
                         f"found {sorted(runs)}")

    # Every bar and curve in a figure must describe the same models.
    model_sets = {stem: set(fit) for stem, fit in fits.items()} | {key: set(run) for key, run in runs.items()}
    if any(s != set(MODELS) for s in model_sets.values()):
        raise SystemExit(f"the fit files do not all cover the {len(MODELS)} timing models: "
                         f"{ {k: len(s) for k, s in model_sets.items()} }")

    order = model_order(fits["params"])
    labels = bar_labels(fits, order)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_per_model(fits, order, out_dir / "per_model_error_bars.png")
    for d_source, d_text, suffix in (("d_hat", r"spectral $\hat{d}$", ""), ("d_star", "true $d$", "_true_d")):
        plot_cross(runs, directions, d_source, order, labels,
                   f"Reference → target, {d_text}",
                   out_dir / f"cross_gpu_comparison{suffix}.png")

    models = load_timing_models(args.prefill_dir)
    qwen35 = load_models(args.qwen35_dir)
    if not qwen35:
        raise SystemExit(f"{args.qwen35_dir}: no timing files")
    plot_global_fit(models + qwen35, out_dir / "global_timing_fit.png")

    query = load_budget_fits(args.budget_fits_dir, models, fits)
    d_curve = load_d_curve(args.sampling_dir)
    plot_figure_2(query, d_curve, out_dir / "figure_2.png")
    plot_cost_tokens(query, out_dir / "err_vs_api_calls_cost_tokens.png")
    plot_by_group(query, out_dir / "L_N_error_vs_query_budget_by_group.png")

    for quantity, group in (("L", L_FITS), ("P", P_FITS)):
        print_table(f"Same-machine |{quantity} rel err|", order,
                    {stem: rel_err_pct(fits[stem], order, quantity) for _, stem in group})
    for d_source in ("d_star", "d_hat"):
        for fit, quantity in (("layers", "L"), ("params", "P")):
            print_table(f"Cross-platform |{quantity} rel err|, {d_source}", order,
                        {f"{ref}->{tgt}": rel_err_pct(runs[fit, d_source, ref, tgt], order, quantity)
                         for ref, tgt in directions})
    print_sweep(query)
    print_d_curve(d_curve)



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # In this order: plot_global_fit changes the math font for every figure drawn after it.
    if args.figure in ("error", "ratio", "all"):
        generate_sampling_figures(args, parser)
    if args.figure in ("timing", "all"):
        generate_timing_figures(args)


if __name__ == "__main__":
    main()
