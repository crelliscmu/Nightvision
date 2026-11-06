#!/usr/bin/env python3
"""Estimate hidden dimension from cached logits without loading a model.

Each input is a torch .pt file containing a 2-D (prompts, vocabulary) tensor.
The first --max-prompts rows form the prompt set. The --budget is split into
--n-steps equal step quotas, and each step draws quota // n_p samples from
every prompt.

After each step (or every --eval-every steps) the presence matrix is pruned to
its largest all-observed biclique, that block's cached log-probabilities form
the common-token matrix (CTM), and its Gram eigenvalues supply the AOI
landmarks, spectrum class and peak estimate. The estimate is the raw AOI peak
floored onto ROUND_GRID. Log probabilities are computed once in GPU chunks and
cached on CPU so the original logits can be released.

logits_path accepts a single file or a directory matched by --pattern.
Each run writes <output_dir>/<run_tag>/<run_tag>_sampling.json and per-step
PNGs under plots/. A summary is rewritten after each file.

CLI defaults, paths, chunk sizes, and analysis constants are defined below.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
# Growable CUDA segments: the sampling loop allocates many ~3 GiB chunk
# temporaries of varying size, which fragments the caching allocator
# (6 GiB "reserved but unallocated" at 20k prompts on a 32 GiB V100).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent


def _find_support_dir() -> Path:
    """Locate the dir holding model_configs.json (the NightVision repository root).

    Walk up from the script dir until it is found so the script also works when
    copied into a separate workspace.
    """
    for d in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        if (d / "model_configs.json").exists():
            return d
    return SCRIPT_DIR


SUPPORT_DIR = _find_support_dir()
if str(SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPORT_DIR))


# CONFIGURATION
# DEFAULT_* values are CLI defaults; other constants configure runtime behavior.

# Paths and devices
# Logits come from collect_logits/collect.py (one <stem>_logits.pt per prompt
# set in ../prompts); trajectories, plots and the summary go to results/.
DEFAULT_LOGITS_PATH = SCRIPT_DIR.parent / "collect_logits" / "results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_PATTERN = "*.pt"
LOGITS_SUFFIX = "_logits"                 # stripped from input filenames
PLOTS_SUBDIR = "plots"
TRAJECTORY_JSON_SUFFIX = "_sampling.json"
SUMMARY_JSON_NAME = "sampling_summary.json"
# What each evaluated step records. Everything else is derivable from these:
# the raw peak is aoi_peak, tightness is aoi_saddle / aoi_peak, and samples drawn
# are draws_per_prompt_total * n_p.
STEP_RESULT_FIELDS = ("d_hat", "aoi_saddle", "aoi_peak", "spectrum_class", "sc_snapback", "rel_err")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLING_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Sampling
DEFAULT_TEMPERATURE = 2.0
DEFAULT_SAMPLES_PER_ITER = 10_000         # largest draws per prompt per multinomial call
DEFAULT_SEED = 42
GRAM_DTYPE_CHOICES = ("float32", "float64")
DEFAULT_GRAM_DTYPE = "float64"
# Storage precision of the CPU log-prob cache the CTM is gathered from.
LOGPROB_DTYPES = {"float16": torch.float16, "float32": torch.float32,
                  "bfloat16": torch.bfloat16}
DEFAULT_LOGPROB_DTYPE = "bfloat16"

# Budget allocation
DEFAULT_MAX_PROMPTS = 20_000              # prompts in the set; also the input row limit
DEFAULT_BUDGET = 20_000_000_000           # total samples across prompts
DEFAULT_N_STEPS = 100                     # equal budget slices
DEFAULT_EVAL_EVERY = 1                    # evaluate after every k-th step (last step always)

# Pruning and memory limits
MIN_NON_NAN_PER_COL = 2                # minimum column support before search
PRUNE_ROW_CHUNK = 5000                   # rows per support reduction
SAMPLE_PROMPT_CHUNK = 5000               # rows per sampling or CTM chunk
PREPARE_PROMPT_CHUNK = 1024              # rows per initial softmax/log-softmax chunk
SAT_VOCAB_CHUNK = 16384                  # tokens per saturation reduction

# Common-token block search
SWEEP_PASSES = 2                         # later passes reorder using the winning rows
SWEEP_COL_CHUNK = 4096                   # columns per prefix scan chunk
SHRINK_GUARD_MIN_STEP = 10               # searches at steps <= this run without the CTM
                                         # no-shrink guard; after that the guard follows the
                                         # latest spectrum class (on while C2, off otherwise)

# Spectrum analysis
SLOPE_WINDOW = 100                       # local regression window
SMOOTH_WINDOW = 100                      # smoothing window for detection and plotting
PRED_SMOOTH = True                       # _pred smooths the slopes before its saddle/peak search
ROUND_GRID = 128                         # prediction flooring interval
PEAK_PLATEAU_FRAC = 0.90                 # recovery fraction marking the peak
SNAP_TRUST = 0.60                        # snapback threshold for C3_good
SC_R_TRUST = 0.70                        # saddle/peak tightness threshold for C3_good

# Plot styling
D_HAT_STYLE = ("tab:brown", "--")        # vline style for the rounded estimate
SEG_FIT_STYLE = ("tab:red", "-")         # line style for the two-segment fit
SEG_FIT_MIN_PTS = 3                      # minimum samples per fitted segment
GAP_RANK_STYLE = ("tab:green", "-.")     # vline style for the largest successive log-drop
GAP_RANK_MIN_DROP = math.log(100.0)      # a drop below ~ln(100) means there is no cliff
INPUT_PANEL_TITLES = {
    "raw": "raw E (log y)",
    "arctan": "arctan(E)",
    "slopes": "windowed slope of arctan(E)",
}
SPECTRUM_CLASS_COLORS = {
    "C3_good": "tab:green",
    "C2_marginal": "tab:olive",
    "C1_underestimate": "tab:orange",
    "C0_not_predictive": "tab:red",
}

# Filename prefixes mapped to model_configs.json entries for hidden_size lookup.
_STEM_TO_HF = {
    # train
    "granite_3.1_1b_a400m": "ibm-granite/granite-3.1-1b-a400m-instruct",
    "granite_3.1_2b":       "ibm-granite/granite-3.1-2b-instruct",
    "qwen3.5_9b":           "Qwen/Qwen3.5-9B",
    "qwen3_0.6b":           "Qwen/Qwen3-0.6B",
    "smol2_360m":           "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smol2_1.7b":           "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smol3_3b":             "HuggingFaceTB/SmolLM3-3B",

    # validation
    "qwen2.5_0.5b":         "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5_1.5b":         "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5_3b":           "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5_7b":           "Qwen/Qwen2.5-7B-Instruct",
    "qwen3_1.7b":           "Qwen/Qwen3-1.7B",
    "qwen3_4b":             "Qwen/Qwen3-4B",
    "qwen3_8b":             "Qwen/Qwen3-8B",
    "qwen3_30b":            "Qwen/Qwen3-30B-A3B",
    "qwen3_30b_instruct":   "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3.5_0.8b":         "Qwen/Qwen3.5-0.8B",
    "qwen3.5_2b":           "Qwen/Qwen3.5-2B",
    "qwen3.5_4b":           "Qwen/Qwen3.5-4B",
    "granite_3.0_3b_a800m": "ibm-granite/granite-3.0-3b-a800m-instruct",
    "granite_3.0_2b":       "ibm-granite/granite-3.0-2b-instruct",
    "granite_3.0_8b":       "ibm-granite/granite-3.0-8b-instruct",
    "granite_3.1_3b_a800m": "ibm-granite/granite-3.1-3b-a800m-instruct",
    "granite_3.1_8b":       "ibm-granite/granite-3.1-8b-instruct",
    "granite_3.2_2b":       "ibm-granite/granite-3.2-2b-instruct",
    "granite_3.2_8b":       "ibm-granite/granite-3.2-8b-instruct",
    "granite_3.3_2b":       "ibm-granite/granite-3.3-2b-instruct",
    "granite_3.3_8b":       "ibm-granite/granite-3.3-8b-instruct",
    "granite_4.0_350m":     "ibm-granite/granite-4.0-350m",
    "granite_4.0_micro":    "ibm-granite/granite-4.0-micro",
    "llama_3.2_1b":         "meta-llama/Llama-3.2-1B-Instruct",
    "llama_3.2_3b":         "meta-llama/Llama-3.2-3B-Instruct",
    "llama_3.1_8b":         "meta-llama/Llama-3.1-8B-Instruct",
    "olmo_2_1b":            "allenai/OLMo-2-0425-1B-Instruct",
    "olmo_2_7b":            "allenai/OLMo-2-1124-7B-Instruct",
    "olmoe_7b":             "allenai/OLMoE-1B-7B-0924-Instruct",
    "olmo_3_7b":            "allenai/Olmo-3-7B-Instruct",
    "smol2_135m":           "HuggingFaceTB/SmolLM2-135M-Instruct",
    "granite_3.0_1b_a400m": "ibm-granite/granite-3.0-1b-a400m-instruct",
}


def _load_true_dims() -> dict:
    """stem -> hidden_size for every stem whose model is in model_configs.json.

    Stems whose model is absent from the config are left out (true_dim_for
    then reports -1 for them, i.e. unknown) rather than aborting the run.
    """
    cfg_path = SUPPORT_DIR / "model_configs.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    missing = [stem for stem, hf in _STEM_TO_HF.items() if hf not in cfg]
    if missing:
        print(f"[config] no hidden_size in {cfg_path.name} for: {', '.join(missing)} "
              "(true dim will be reported as unknown)", flush=True)
    return {stem: cfg[hf]["hidden_size"] for stem, hf in _STEM_TO_HF.items() if hf in cfg}


# True hidden sizes, for the reported relative error.
TRUE_DIMS = _load_true_dims()

# END CONFIGURATION


# ----------------------------------------------------------------------------
# common-token block: the maximum edge biclique of the presence matrix
# ----------------------------------------------------------------------------

@dataclass
class SortResult:
    """One completed prune: the shape of the retained block and how it was
    found.  The block itself comes back from `prune_common_rows_cols` as the
    `(rows, cols)` index tensors alongside this record; it is the best maximal
    biclique the search found, so it is all-observed by construction."""
    n_rows: int
    n_cols: int
    n_bulk: int                      # columns removed by the support floor
    support_floor: int
    seconds: float
    info: dict = field(repr=False, default_factory=dict)   # method-specific extras

    @property
    def n_ev(self) -> int:
        """Spectrum length: the Gram matrix has min(D', |C|) eigenvalues."""
        return min(self.n_rows, self.n_cols)

    @property
    def cells(self) -> int:
        return self.n_rows * self.n_cols

    def __str__(self):
        return (f"D'={self.n_rows}  |C|={self.n_cols}  "
                f"n_ev={self.n_ev}  cells={self.cells/1e6:.2f}M  "
                f"bulk={self.n_bulk}  {self.seconds:.1f}s")


def column_support(presence: torch.Tensor,
                   row_chunk: int = PRUNE_ROW_CHUNK) -> torch.Tensor:
    """Observations per column, reduced in row slices.

    A whole-matrix `sum(0)` promotes every cell to a 4-8 byte temporary
    (38 GiB for a 40k x 128k bool matrix), hence the chunking.
    """
    out = torch.zeros(presence.shape[1], dtype=torch.int32, device=presence.device)
    for s in range(0, presence.shape[0], row_chunk):
        out += presence[s:s + row_chunk].sum(0, dtype=torch.int32)
    return out


def reclose_block(sub: torch.Tensor, rows: torch.Tensor,
                  row_chunk: int = PRUNE_ROW_CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """Closure of a row set: ``J = cols observed on every row``, then
    ``I = rows observed on every column of J``.

    Both sides can only grow (``I`` contains the input rows, ``J`` contains
    every column the input block had), so re-closing last step's block against
    this step's presence is a free, strictly monotone improvement: presence is
    write-only, so the extra samples since then may have completed columns the
    block was one observation short of.

    Counts zeros rather than ANDing so the reduction stays in int32 slices.
    """
    dev = sub.device
    miss_c = torch.zeros(sub.shape[1], dtype=torch.int32, device=dev)
    for s in range(0, rows.numel(), row_chunk):
        blk = sub.index_select(0, rows[s:s + row_chunk])
        miss_c += (~blk).sum(0, dtype=torch.int32)
        del blk
    J = torch.nonzero(miss_c == 0, as_tuple=False).flatten()
    if J.numel() == 0:
        return rows, J
    keep_r = torch.empty(sub.shape[0], dtype=torch.bool, device=dev)
    for s in range(0, sub.shape[0], row_chunk):
        keep_r[s:s + row_chunk] = sub[s:s + row_chunk].index_select(1, J).all(1)
    return torch.nonzero(keep_r, as_tuple=False).flatten(), J


def block_score(n_rows: int, n_cols: int) -> int:
    """What the search maximises: the block's edge count ``D' * |C|``."""
    return n_rows * n_cols


def _degree_cap(deg: torch.Tensor, cells_floor: int) -> int:
    """Largest ``k`` with ``k * (k-th largest degree) > cells_floor``, else 0.

    A block with ``|J| = k`` has every column observed by all of ``I``, so
    ``|I| <= min support over J <= s_(k)``, the k-th largest support.  Hence
    ``cells <= k * s_(k)``: any ``k`` failing that test is unreachable, and the
    largest surviving ``k`` caps the block's width.
    """
    s = torch.sort(deg.to(torch.int64), descending=True).values
    k = torch.arange(1, s.numel() + 1, device=s.device, dtype=torch.int64)
    ok = (k * s) > int(cells_floor)
    return int(torch.nonzero(ok, as_tuple=False).max()) + 1 if bool(ok.any()) else 0

def _pred(slopes: torch.Tensor, slopes_off: int = 0) -> tuple[int, int]:

    if isinstance(slopes, np.ndarray):
        slopes = torch.from_numpy(slopes)
    slopes = slopes.to(torch.float64)
    n = int(slopes.shape[0])

    if PRED_SMOOTH:
        slopes = _smooth(slopes, min(SMOOTH_WINDOW, n))
    valid = torch.nonzero(~torch.isnan(slopes), as_tuple=True)[0]
    if valid.numel() == 0:
        return -1, -1
    first = int(valid[0].item())
    last = int(valid[-1].item()) + 1
    slopes = slopes[first:last]
    slopes_off += first
    n = int(slopes.shape[0])
    W = int(math.sqrt(n))

    def _g(i: int) -> int:                   # local 0-based -> global 1-based
        return i + slopes_off + 1 if i >= 0 else -1

    local_minima = torch.nonzero(
      (slopes[1:-1] < slopes[:-2])
      & (slopes[1:-1] <= slopes[2:]),
      as_tuple=True,
      )[0] + 1

    saddle, hi = -1, -1
    for m in local_minima.tolist():                     # ascending index order
        # crest of m's hump: highest point before the curve drops below m's level
        below = torch.nonzero(slopes[m + 1:] < slopes[m], as_tuple=True)[0]
        stop = m + 1 + int(below[0].item()) if below.numel() > 0 else n
        c = m + int(torch.argmax(slopes[m:stop]).item())
        if c - m < W:
            continue
        if stop < c + (c - m) and bool((slopes[stop:] > slopes[m]).any()):
            continue
        saddle, hi = m, c
        break

    if saddle < 0:
        print("SADDLE ERROR OR PEAK ERROR: saddle_g: -1, peak: -1")
        return -1, -1

    segment = slopes[saddle:hi + 1]
    saddle_val = segment[0]
    plateau_val = segment.max()
    amp = float((plateau_val - saddle_val).item())

    if amp <= 0.0:                           # flat: no recovery to mark a peak on
        print(f"ERROR: Negative amplitude: saddle_g: {_g(saddle)}, peak: -1")
        return _g(saddle), -1

    target = saddle_val + PEAK_PLATEAU_FRAC * (plateau_val - saddle_val)
    crossed = torch.nonzero(segment >= target, as_tuple=True)[0]

    # amp > 0 guarantees segment.max() >= target, so crossed is non-empty and
    # its first entry is > 0 (segment[0] is the saddle depth itself)
    peak = saddle + (int(crossed[0].item()) if crossed.numel() > 0 else 0)

    if peak <= saddle:
        print(f"SADDLE ERROR OR PEAK ERROR: saddle_g: {_g(saddle)}, peak: {peak}")
        return _g(saddle), -1

    return _g(saddle), _g(peak)



def certificate_peel(sub: torch.Tensor, score_floor: int,
                     row_chunk: int = PRUNE_ROW_CHUNK, max_rounds: int = 6):
    """Rows and columns that can still appear in a block scoring above
    ``score_floor``, given the incumbent block that certifies that floor.

    The certificate is *sound* -- a dropped row or column provably cannot
    improve on the incumbent -- and is counted *within* the survivors, so each
    drop tightens the opposite axis's bound and the loop runs to a fixpoint.

    Edges (``|I|*|J|``) can be bought with aspect ratio, so the bound comes
    from degree ordering: a block with ``|J| = j`` has ``|I| <= s_(j)``, the
    j-th largest column support, so ``cells <= j * s_(j)`` and
    ``j <= j_max = _degree_cap(...)``.  Every column of a better block is then
    seen by ``|I| > score_floor / j_max`` rows.  Symmetric on rows.

    Returns ``(row_idx, col_idx, rounds)``; an empty column index means the
    incumbent cannot be beaten at all and the search can be skipped.
    """
    dev = sub.device
    m, n = sub.shape
    rmask = torch.ones(m, dtype=torch.bool, device=dev)
    cmask = torch.ones(n, dtype=torch.bool, device=dev)
    if score_floor <= 0:
        return (torch.nonzero(rmask, as_tuple=False).flatten(),
                torch.nonzero(cmask, as_tuple=False).flatten(), 0)

    for rnd in range(1, max_rounds + 1):
        cs = torch.zeros(n, dtype=torch.int32, device=dev)
        rs = torch.zeros(m, dtype=torch.int32, device=dev)
        for s in range(0, m, row_chunk):
            e = min(s + row_chunk, m)
            blk = sub[s:e] & cmask                      # only surviving columns count
            rs[s:e] = blk.sum(1, dtype=torch.int32)
            blk &= rmask[s:e].unsqueeze(1)              # only surviving rows count
            cs += blk.sum(0, dtype=torch.int32)
            del blk
        cs = torch.where(cmask, cs, torch.zeros_like(cs))
        rs = torch.where(rmask, rs, torch.zeros_like(rs))

        # degree-ordering bound
        j_max = _degree_cap(cs, score_floor)
        i_max = _degree_cap(rs, score_floor)
        if j_max == 0 or i_max == 0:                    # incumbent is unbeatable
            return (torch.empty(0, dtype=torch.long, device=dev),
                    torch.empty(0, dtype=torch.long, device=dev), rnd)
        new_c = cmask & (cs.to(torch.int64) * j_max > score_floor)
        new_r = rmask & (rs.to(torch.int64) * i_max > score_floor)
        dropped = int((cmask & ~new_c).sum()) + int((rmask & ~new_r).sum())
        cmask, rmask = new_c, new_r
        if not dropped:
            break
    return (torch.nonzero(rmask, as_tuple=False).flatten(),
            torch.nonzero(cmask, as_tuple=False).flatten(), rnd)


def restricted_column_support(sub: torch.Tensor, rows: torch.Tensor,
                              row_chunk: int = PRUNE_ROW_CHUNK) -> torch.Tensor:
    """Observations per column counted over ``rows`` only."""
    out = torch.zeros(sub.shape[1], dtype=torch.int32, device=sub.device)
    for s in range(0, rows.numel(), row_chunk):
        out += sub.index_select(0, rows[s:s + row_chunk]).sum(0, dtype=torch.int32)
    return out


def prefix_alive(sub: torch.Tensor, order: torch.Tensor,
                 col_chunk: int = SWEEP_COL_CHUNK) -> torch.Tensor:
    """``|I_k|`` for every prefix of ``order``: how many rows are observed on
    *all* of the first k columns, for every k at once.

    One O(rows x cols) pass.  A row survives the first k columns iff it has
    zero zeros among them, so a running cumulative sum of the zero indicator
    down the ordering gives every prefix's row count in one sweep -- no
    per-candidate GEMM, and the whole ``|I|`` vs ``|J|`` trade-off curve comes
    out instead of a single block.
    """
    m = sub.shape[0]
    n_alive = torch.zeros(order.numel(), dtype=torch.int32, device=sub.device)
    acc = torch.zeros(m, dtype=torch.int32, device=sub.device)   # zeros seen so far
    for s in range(0, order.numel(), col_chunk):
        e = min(s + col_chunk, order.numel())
        c = (~sub.index_select(1, order[s:e])).to(torch.int32).cumsum(1) + acc.unsqueeze(1)
        n_alive[s:e] = (c == 0).sum(0, dtype=torch.int32)
        acc = c[:, -1].clone()
        del c
        if s and int(n_alive[s:e].max()) == 0:
            break                       # every row is dead; the tail is all zeros
    return n_alive


def sweep_biclique(sub: torch.Tensor, passes: int = None,
                   row_chunk: int = PRUNE_ROW_CHUNK):
    """Best all-ones block over the prefixes of a support ordering.

    Pass 1 orders columns by global support; each later pass re-orders by
    support restricted to the winning row set, which is where the ordering is
    actually wrong (a column the block's rows all saw can sit low in the global
    order).  The prefix's row set is closed on the column side afterwards, so
    the returned block is maximal.

    Returns ``(rows, cols)``.
    """
    passes = SWEEP_PASSES if passes is None else int(passes)
    n = sub.shape[1]
    order = torch.argsort(column_support(sub, row_chunk), descending=True)
    best_rows = best_cols = None
    best = -1
    for p in range(max(1, passes)):
        na = prefix_alive(sub, order).to(torch.int64)
        k = torch.arange(1, n + 1, device=sub.device, dtype=torch.int64)
        score = na * k
        kk = int(torch.argmax(score)) + 1
        # rows observed on the whole prefix, then closed on the column side
        keep = torch.empty(sub.shape[0], dtype=torch.bool, device=sub.device)
        J = order[:kk]
        for s in range(0, sub.shape[0], row_chunk):
            keep[s:s + row_chunk] = sub[s:s + row_chunk].index_select(1, J).all(1)
        I0 = torch.nonzero(keep, as_tuple=False).flatten()
        if I0.numel() == 0:
            break
        rows, cols = reclose_block(sub, I0, row_chunk)
        sc = block_score(int(rows.numel()), int(cols.numel()))
        if sc > best:
            best, best_rows, best_cols = sc, rows, cols
        if p + 1 < passes:
            order = torch.argsort(restricted_column_support(sub, rows, row_chunk),
                                  descending=True)
    if best_rows is None:               # nothing survived any prefix
        empty = torch.empty(0, dtype=torch.long, device=sub.device)
        return empty, empty
    return best_rows, best_cols


class EmptyCTM(RuntimeError):
    """No all-observed block exists at this step: the support floor dropped
    every column, or nothing survived closure. A legitimate empty spectrum,
    unlike any other RuntimeError (CUDA OOM is one), which must propagate."""


def prune_common_rows_cols(good: torch.Tensor,
                           min_non_nan_per_col: int = MIN_NON_NAN_PER_COL,
                           incumbent: tuple[torch.Tensor, torch.Tensor] | None = None,
                           shrink_guard_active: bool = True):
    """Retain the largest all-observed block of `good` (True where sampled).

    Four stages:

    1. **Support floor.**  Columns seen by fewer than `min_non_nan_per_col`
       prompts are dropped outright.  They can only shrink a biclique that
       contains them, and dropping them first is what keeps the search
       tractable in both memory and time.
    2. **Incumbent.**  `incumbent` is the previous step's block in original
       (prompt, token) indices.  `presence` is write-only and the floor never
       rises, so that block is still all-ones here: it is a valid lower bound
       the search is not allowed to lose.  It is re-closed first, which can
       only grow it (`reclose_block`).
    3. **Certificate peel.**  Rows and columns that provably cannot beat the
       incumbent's score are dropped before the search runs
       (`certificate_peel`).  A peel that empties the matrix is a proof the
       incumbent is optimal, and the search is skipped entirely.
    4. **Search.**  The prefix sweep (`sweep_biclique`, one O(rows x cols)
       pass over the support ordering).  The winner is the better of that
       result and the incumbent by edge count (`block_score`).
       When `shrink_guard_active` is true, a search result is eligible only
       when it has at least as many rows and columns as the re-closed
       incumbent. The sampling loop enables that guard while the latest
       spectrum classification is C2 (a revert to C1/C0 clears it) and only
       after the first ``SHRINK_GUARD_MIN_STEP`` steps.

    Returns the kept `(rows, cols)` index tensors on `good`'s device, plus a
    `SortResult`.  Raises `EmptyCTM` if the support floor empties the
    matrix -- the caller reads that as an empty CTM, not as a failure.
    """
    t0 = time.time()
    if good.dtype != torch.bool:
        good = good.bool()
    D, V = good.shape
    dev = good.device
    floor = int(min_non_nan_per_col or 0)
    shrink_guard_active = bool(shrink_guard_active)

    # -- stage 1: bulk drop the columns nobody much sampled --------------------
    support = column_support(good)
    if floor > 0:
        keep_c = torch.nonzero(support >= floor, as_tuple=False).flatten()
    else:
        keep_c = torch.arange(V, device=dev)
    n_bulk = V - int(keep_c.numel())
    if keep_c.numel() == 0:
        raise EmptyCTM(f"support floor {floor} removed every column")
    if n_bulk:
        print(f"    bulk-dropped {n_bulk} columns with support < {floor}"
              f"  ({keep_c.numel()} columns left)", flush=True)

    sub = good.index_select(1, keep_c) if n_bulk else good

    # -- stage 2: carry the incumbent forward ----------------------------------
    # keep_c maps local column ids -> token ids; invert it to bring the
    # incumbent's token ids into this step's local numbering.
    inc_rows = inc_cols = None
    inc_cells = inc_score = 0
    t_inc = time.time()
    if incumbent is not None and incumbent[0].numel() and incumbent[1].numel():
        local = torch.full((V,), -1, dtype=torch.long, device=dev)
        local[keep_c] = torch.arange(keep_c.numel(), device=dev)
        p_rows = incumbent[0].to(dev)
        p_cols = local[incumbent[1].to(dev)]
        p_rows = p_rows[p_rows < D]                    # rows only ever get added
        p_cols = p_cols[p_cols >= 0]                   # survived the floor
        if p_rows.numel() and p_cols.numel():
            inc_rows, inc_cols = reclose_block(sub, p_rows)
            inc_cells = int(inc_rows.numel()) * int(inc_cols.numel())
            inc_score = block_score(int(inc_rows.numel()), int(inc_cols.numel()))
            if not bool(sub.index_select(0, inc_rows).index_select(1, inc_cols).all()):
                raise AssertionError("carried incumbent is not all-observed")
            print(f"    incumbent {int(p_rows.numel())}x{int(p_cols.numel())} re-closed to "
                  f"{inc_rows.numel()}x{inc_cols.numel()} = {inc_cells / 1e6:.2f}M cells, "
                  f"edges={inc_score} ({time.time() - t_inc:.1f}s)", flush=True)
        del local
    t_inc = time.time() - t_inc

    # -- stage 3: drop what provably cannot beat the incumbent -----------------
    t_peel = time.time()
    pr, pc, peel_rounds = certificate_peel(sub, inc_score)
    peel_rows, peel_cols = int(pr.numel()), int(pc.numel())
    t_peel = time.time() - t_peel
    if inc_score:
        print(f"    certificate peel ({peel_rounds} rounds, {t_peel:.1f}s): "
              f"{sub.shape[0]}x{sub.shape[1]} -> {peel_rows}x{peel_cols}"
              f"  ({100 * (1 - peel_rows * peel_cols / (sub.shape[0] * sub.shape[1])):.1f}%"
              f" of cells eliminated)", flush=True)

    # -- stage 4: search what is left -----------------------------------------
    rows, cols, won_by = inc_rows, inc_cols, "incumbent"
    best = inc_score

    def can_replace_incumbent(candidate_rows: int, candidate_cols: int) -> bool:
        """When enabled, a replacement must not trade either CTM dimension."""
        return (not shrink_guard_active or inc_rows is None or
                (candidate_rows >= int(inc_rows.numel()) and
                 candidate_cols >= int(inc_cols.numel())))

    t_sweep = 0.0
    if peel_rows and peel_cols:
        peeled = sub if (peel_rows, peel_cols) == tuple(sub.shape) \
            else sub.index_select(0, pr).index_select(1, pc)

        t = time.time()
        s_rows, s_cols = sweep_biclique(peeled)
        t_sweep = time.time() - t
        if s_rows.numel() and s_cols.numel():
            sc = block_score(int(s_rows.numel()), int(s_cols.numel()))
            print(f"    sweep: {s_rows.numel()}x{s_cols.numel()} edges={sc} "
                  f"({t_sweep:.1f}s)", flush=True)
            if (can_replace_incumbent(int(s_rows.numel()), int(s_cols.numel()))
                    and sc > best):
                best, won_by = sc, "sweep"
                rows, cols = pr.index_select(0, s_rows), pc.index_select(0, s_cols)

        if peeled is not sub:
            del peeled
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        if won_by == "incumbent" and inc_score:
            print(f"    sweep did not beat the incumbent (edges={inc_score}); "
                  f"keeping it", flush=True)
    elif inc_score:
        print("    peel emptied the matrix: the incumbent is optimal, search skipped",
              flush=True)
    if rows is None or not rows.numel() or not cols.numel():
        raise EmptyCTM("no non-empty all-observed block found")
    cols = keep_c.index_select(0, cols)               # back to original token ids
    del sub, support

    res = SortResult(
        n_rows=int(rows.numel()), n_cols=int(cols.numel()),
        n_bulk=n_bulk, support_floor=floor,
        seconds=time.time() - t0,
        info={"won_by": won_by},
    )
    got = block_score(res.n_rows, res.n_cols)
    if inc_score and got < inc_score:
        raise AssertionError(f"block regressed below the incumbent: "
                             f"edges {got} < {inc_score}")
    if shrink_guard_active and inc_rows is not None and (
            res.n_rows < int(inc_rows.numel()) or res.n_cols < int(inc_cols.numel())):
        raise AssertionError("block shrank below the re-closed incumbent: "
                             f"{res.n_rows}x{res.n_cols} < "
                             f"{inc_rows.numel()}x{inc_cols.numel()}")
    return rows, cols, res


def smaller_side_gram(M: torch.Tensor) -> torch.Tensor:
    m, n = M.shape
    return M @ M.T if m <= n else M.T @ M


def stem_from_logits_path(path: Path) -> str:
    """Run stem for a cached-logits file: the filename stem minus a trailing
    ``_logits`` (``granite_3.1_2b_logits.pt`` -> ``granite_3.1_2b``)."""
    stem = path.stem
    if stem.endswith(LOGITS_SUFFIX) and len(stem) > len(LOGITS_SUFFIX):
        stem = stem[: -len(LOGITS_SUFFIX)]
    return stem


def resolve_logits_files(logits_path: Path, pattern: str) -> list[Path]:
    """The .pt files to process: the path itself, or every ``pattern`` match
    directly inside it when it is a directory."""
    if logits_path.is_dir():
        files = sorted(p for p in logits_path.glob(pattern) if p.is_file())
        if not files:
            raise SystemExit(f"No files matching {pattern!r} in {logits_path}")
        return files
    if not logits_path.exists():
        raise SystemExit(f"Logits path does not exist: {logits_path}")
    return [logits_path]


def load_logits(path: Path, max_prompts: int | None) -> torch.Tensor:
    """mmap the cached logits and return their first ``max_prompts`` rows.

    n_max doubles as the read cap: the prompt set is the first n_max rows, so
    there is no reason to read any row past them.
    mmap keeps the untouched tail off the heap, so the cap applied to a 100 GiB
    cache only ever faults in the rows it slices.
    """
    logits = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if logits.ndim != 2:
        raise ValueError(
            f"{path}: expected a 2-D (prompts, vocab) logits tensor, "
            f"got shape={tuple(logits.shape)}"
        )
    cached = int(logits.shape[0])
    if max_prompts is not None and max_prompts > 0:
        if max_prompts > cached:
            print(f"  n_max {max_prompts} exceeds the {cached} cached prompts; using all of them")
        else:
            logits = logits[:max_prompts]
    print(f"  logits {tuple(logits.shape)} {logits.dtype} "
          f"(of {cached} cached prompts) from {path}")
    return logits


def true_dim_for(stem: str) -> int:
    for k in sorted(TRUE_DIMS, key=len, reverse=True):
        if stem.startswith(k):
            return TRUE_DIMS[k]
    return -1


# --------------------------- predictors ------------------------------------
def _floor_grid(x: int, grid: int = ROUND_GRID) -> int:
    """Snap ``x`` DOWN to the previous multiple of ``grid``.

    Returns -1 for no prediction, which includes anything that floors below a
    full grid step -- a prediction under ``grid`` carries no multiple to land
    on, so it is reported as absent rather than as 0.
    """
    if x <= 0:
        return -1
    floored = int((x // grid) * grid)
    return floored if floored > 0 else -1


def get_slopes(y: torch.Tensor, window: int | None = None) -> torch.Tensor:
    """Vectorised per-index windowed local slope of ``y`` on its source device.

    Closed-form linear-regression slope on a centered odd window via prefix
    sums — O(n). Entries within ``half`` of either edge are NaN. Internal
    accumulations use float64 to avoid precision loss in the cumulative sums.
    """
    n = y.shape[0]
    if window is None:
        window = min(SLOPE_WINDOW, n)
    if window % 2 == 0:
        window += 1
    half = window // 2
    slopes = torch.full((n,), float("nan"), dtype=torch.float64, device=y.device)
    if n < window or half < 1:
        return slopes

    y64 = y.to(torch.float64)
    j = torch.arange(n, dtype=torch.float64, device=y.device)
    zero = y64.new_zeros(1)
    cy  = torch.cat([zero, torch.cumsum(y64, dim=0)])
    cjy = torch.cat([zero, torch.cumsum(j * y64, dim=0)])

    sum_x2 = half * (half + 1) * (2 * half + 1) / 3.0

    idx = torch.arange(half, n - half, device=y.device)
    lo = idx - half
    hi = idx + half + 1
    sum_y  = cy[hi]  - cy[lo]
    sum_jy = cjy[hi] - cjy[lo]
    sum_xy = sum_jy - idx.to(torch.float64) * sum_y
    slopes[idx] = sum_xy / sum_x2
    return slopes



def _smooth(y: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1 or y.shape[0] < window:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    y_p = y.to(torch.float64).view(1, 1, -1)
    y_p = F.pad(y_p, (pad, pad), mode="replicate")
    kernel = torch.full((1, 1, window), 1.0 / window,
                        dtype=torch.float64, device=y.device)
    return F.conv1d(y_p, kernel).view(-1)


def _smooth_np(y: np.ndarray, window: int) -> np.ndarray:
    """Numpy variant of ``_smooth`` for the plotting boundary."""
    if window <= 1 or len(y) < window:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    yp = np.pad(y, pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(yp, kernel, mode="valid")



# --------------------------- predictor dispatch ----------------------------
def derive_inputs(E) -> tuple[dict, tuple]:
    """Derive signals and the AOI tuple from ``E``.

    Accepts ``E`` as either a 1-D tensor (any device) or a numpy array; numpy
    inputs are converted to a tensor and the returned ``inputs`` values back
    to numpy.

    Returns ``(inputs, aoi)`` where ``aoi`` is the
    ``(saddle_g, peak_g)`` tuple from :func:`_pred`.
    Returning it here lets callers avoid a second pass.
    """
    is_numpy_input = isinstance(E, np.ndarray)
    if is_numpy_input:
        E = torch.from_numpy(E).to(DEVICE)

    a_ev = torch.atan(E)

    slopes_full = get_slopes(a_ev)
    mask = ~torch.isnan(slopes_full)
    if mask.any():
        valid_idx = torch.nonzero(mask, as_tuple=True)[0]
        first = int(valid_idx[0].item())
        last  = int(valid_idx[-1].item()) + 1
        slopes_trim = slopes_full[first:last]
        slopes_offset = first
    else:
        slopes_trim = torch.zeros(0, dtype=torch.float64, device=E.device)
        slopes_offset = 0

    aoi = _pred(slopes_trim, slopes_offset)


    inputs = {
        "raw":    (E,           0),
        "arctan": (a_ev,        0),
        "slopes": (slopes_trim, slopes_offset),
    }

    if is_numpy_input:
        inputs = {
            k: (sig.detach().cpu().numpy(), off)
            for k, (sig, off) in inputs.items()
        }
    return inputs, aoi


def run_predictors(aoi: tuple) -> dict:
    """``{"peak_floor": (index, value)}`` read straight off the AOI landmarks.

    ``index`` is the 1-based AOI peak; ``-1`` is the no-prediction sentinel the
    reporting path understands, and the paired ``value`` is 0.0 for a landmark
    read and NaN when there is no prediction. The ROUND_GRID-floored variant is
    reported alongside it as d_hat, not as a separate predictor.
    """
    _, peak_g = aoi
    return {"peak_floor": ((int(peak_g), 0.0) if peak_g > 0
                           else (-1, float("nan")))}


def classify_spectrum(inputs: dict, aoi: tuple) -> dict:
    """Return ``{spectrum_class, sc_snapback, sc_r, sc_w}`` from derive_inputs
    outputs only.

    ``inputs``/``aoi`` are what :func:`derive_inputs` returns; the result is a
    label plus the diagnostic scalars behind it. ``C3_good`` requires landmark agreement
    — ``ceil(saddle)`` and ``floor(peak)`` on the ROUND_GRID land on the same
    multiple, so the window pins a unique candidate — guarded by
    ``snapback >= SNAP_TRUST`` (a narrow-but-spurious window in an unresolved
    spectrum has near-zero snapback and must not be called good) and by
    ``sc_r > SC_R_TRUST`` (basin tightness, saddle/peak: landmark agreement
    compares the window to a fixed grid, so a window that is loose relative to
    its position can still pass at small indices). Everything else is a cut on
    ``snapback``; ``sc_w`` (relative basin width) is recorded for reference but
    does not enter the decision. ``sc_snapback`` is NaN only when there is no usable
    AOI. Reporting a prediction is left to the caller: the label says whether
    the AOI window pins the true dim (``C3``), roughly brackets it (``C2``),
    or only lower-bounds it.
    """
    def _cls(label, **kw):
        rec = {"spectrum_class": label, "sc_snapback": float("nan"),
               "sc_r": float("nan"), "sc_w": float("nan")}
        rec.update(kw)
        return rec

    saddle_g, peak_g = aoi
    if saddle_g <= 0 or peak_g <= saddle_g:
        return _cls("C0_not_predictive")

    slopes, s_off = inputs["slopes"]
    slopes = slopes.detach().cpu().numpy() if isinstance(slopes, torch.Tensor) \
        else np.asarray(slopes)
    n_sl = len(slopes)
    if n_sl < 3:
        return _cls("C0_not_predictive")

    sm_t = _smooth(torch.from_numpy(np.asarray(slopes, dtype=np.float64)),
                   min(SMOOTH_WINDOW, n_sl))
    sm = sm_t.detach().cpu().numpy() if isinstance(sm_t, torch.Tensor) \
        else np.asarray(sm_t)

    s = int(np.clip(saddle_g - 1 - s_off, 0, n_sl - 1))
    p = int(np.clip(peak_g - 1 - s_off, 0, n_sl - 1))
    width = max(p - s, 1)

    saddle_val = float(sm[s])
    post_win = sm[p:min(n_sl, p + width)]
    post_level = float(np.median(post_win)) if post_win.size else saddle_val
    snapback = (post_level - saddle_val) / max(-saddle_val, 1e-12)

    # C3 needs the two grid landmarks to agree on a single multiple AND a
    # recovered plateau backing it up. Agreement alone is not enough: a narrow
    # window in a still-falling spectrum (tiny snapback) is noise, not a pin.
    # The two landmarks must round inward (saddle up, peak down) onto the same
    # multiple for the window to pin a unique candidate. The estimate itself
    # is floored (see _floor_grid).
    ceil_saddle = -(-saddle_g // ROUND_GRID) * ROUND_GRID
    floor_peak = (peak_g // ROUND_GRID) * ROUND_GRID
    sc_r = saddle_g / peak_g
    if ceil_saddle == floor_peak and snapback >= SNAP_TRUST and sc_r > SC_R_TRUST:
        label = "C3_good"
    elif snapback < SNAP_TRUST:
        label = "C1_underestimate"
    else:
        label = "C2_marginal"

    # sc_w is a diagnostic only; it is just 1/sc_r - 1.
    return _cls(label, sc_snapback=snapback,
                sc_r=sc_r,
                sc_w=(peak_g - saddle_g) / saddle_g)


# --------------------------- plotting --------------------------------------
def _fmt_sc(v) -> str:
    """Format a spectrum-class diagnostic; the scalars are NaN for C0."""
    v = float(v)
    return "n/a" if v != v else f"{v:.3g}"


def _fmt_delta(value) -> str:
    return "n/a" if value is None else f"{value:+.3g}"


def extended_aoi(aoi: tuple, n: int) -> tuple[int, int]:
    """The AOI window extended left and right by half its width, 1-based inclusive.

    ``[saddle_g, peak_g]`` becomes ``[saddle_g - w/2, peak_g + w/2]`` with
    ``w = peak_g - saddle_g`` (half-width rounded up, so a window of width 1
    still grows by one index on each side), clipped to ``[1, n]``. ``(-1, -1)``
    when the AOI has no window (``peak_g <= 0``), which is what :func:`_pred`
    reports for a C0 spectrum.
    """
    saddle_g, peak_g = aoi
    if saddle_g <= 0 or peak_g <= saddle_g or n < 2:
        return -1, -1
    half = (peak_g - saddle_g + 1) // 2
    return max(saddle_g - half, 1), min(peak_g + half, n)


def gap_rank(E, lo: int = 1000, hi: int | None = None) -> tuple[int, float]:
    """Number of eigenvalues above the largest successive log-drop, searched
    over counts in ``[lo, hi)``.

    ``E`` is the descending spectrum as a 1-D tensor or numpy array. Returns
    ``(count, drop)`` where ``drop`` is the natural-log size of the drop
    between ``E[count-1]`` and ``E[count]``; a drop below ``GAP_RANK_MIN_DROP``
    (~ln(100)) means there is no cliff. ``(-1, nan)`` when the search range is
    empty. Non-positive eigenvalues (numerical noise at the tail) are clamped
    to the dtype's smallest positive normal before the log.
    """
    if isinstance(E, np.ndarray):
        E = torch.from_numpy(E)
    E = E.detach().to(torch.float64)
    n = E.numel()
    hi = n if hi is None else min(hi, n)
    lo = max(int(lo), 1)
    if hi - lo < 1 or n < 2:
        return -1, float("nan")
    logs = torch.log(E.clamp_min(torch.finfo(E.dtype).tiny))
    drops = logs[:-1] - logs[1:]                     # drops[j] is between E[j] and E[j+1]
    j = int(torch.argmax(drops[lo - 1:hi - 1]).item()) + lo - 1
    return j + 1, float(drops[j].item())


def gap_rank_in_aoi(E, aoi: tuple) -> tuple[int, float]:
    """:func:`gap_rank` searched over the extended AOI, the same window the
    two-segment fit uses. ``(-1, nan)`` for a C0 spectrum with no window."""
    n = E.numel() if isinstance(E, torch.Tensor) else len(E)
    lo_g, hi_g = extended_aoi(aoi, n)
    if lo_g <= 0:
        return -1, float("nan")
    # extended_aoi is 1-based inclusive; gap_rank's hi is exclusive.
    return gap_rank(E, lo_g, hi_g + 1)


def _ols_sse(n, sx, sy, sxx, syy, sxy):
    """Residual sum of squares of the least-squares line, from moment sums.

    Vectorised over ``n`` so every candidate breakpoint is scored at once.
    A degenerate segment (fewer than 2 points, or no spread in x) falls back to
    the variance about the mean, which is what a horizontal fit would leave.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        syy_c = syy - sy * sy / n
        sxx_c = sxx - sx * sx / n
        sxy_c = sxy - sx * sy / n
        sse = np.where(sxx_c > 0, syy_c - sxy_c * sxy_c / sxx_c, syy_c)
    sse = np.where(np.isfinite(sse), sse, np.inf)
    return np.maximum(sse, 0.0)


def two_segment_fit(sig, off: int, lo_g: int, hi_g: int) -> dict | None:
    """Least-squares two-segment fit of ``sig`` over ``[lo_g, hi_g]`` (1-based,
    inclusive).

    One segment grows from the left edge of the window and the other from the
    right edge; the breakpoint where they meet is the split that minimises the
    combined residual sum of squares. Both segments include the breakpoint
    sample, so the two lines are fitted over overlapping-by-one ranges rather
    than being constrained to intersect there -- the vertical offset between
    them at the break is itself a read on how sharp the knee is.

    Returns ``{"break_g", "left", "right", "sse", "lo_g", "hi_g"}`` where each
    of ``left``/``right`` is ``(slope, intercept)`` in global-index units, or
    ``None`` when the window cannot hold two segments of ``SEG_FIT_MIN_PTS``.
    """
    v = np.asarray(sig, dtype=np.float64).ravel()
    if v.size == 0 or lo_g <= 0 or hi_g <= lo_g:
        return None
    lo = max(int(lo_g) - 1 - off, 0)               # 0-based local, inclusive
    hi = min(int(hi_g) - 1 - off, v.size - 1)
    y = v[lo:hi + 1]
    x = np.arange(lo + off + 1, hi + off + 2, dtype=np.float64)  # global index
    n = y.size
    if n < 2 * SEG_FIT_MIN_PTS - 1 or not np.all(np.isfinite(y)):
        return None

    z = np.zeros(1)
    cn  = np.concatenate((z, np.arange(1, n + 1, dtype=np.float64)))
    cx  = np.concatenate((z, np.cumsum(x)))
    cy  = np.concatenate((z, np.cumsum(y)))
    cxx = np.concatenate((z, np.cumsum(x * x)))
    cyy = np.concatenate((z, np.cumsum(y * y)))
    cxy = np.concatenate((z, np.cumsum(x * y)))

    # Candidate breakpoints as 0-based local indices; the left segment is
    # [0, b] and the right [b, n - 1], so each keeps SEG_FIT_MIN_PTS samples.
    b = np.arange(SEG_FIT_MIN_PTS - 1, n - SEG_FIT_MIN_PTS + 1)
    if b.size == 0:
        return None
    left = _ols_sse(cn[b + 1], cx[b + 1], cy[b + 1],
                    cxx[b + 1], cyy[b + 1], cxy[b + 1])
    right = _ols_sse(cn[n] - cn[b], cx[n] - cx[b], cy[n] - cy[b],
                     cxx[n] - cxx[b], cyy[n] - cyy[b], cxy[n] - cxy[b])
    total = left + right
    if not np.any(np.isfinite(total)):
        return None
    k = int(np.argmin(total))
    bk = int(b[k])

    def _line(xs, ys):
        if xs.size < 2:
            return 0.0, float(ys[0]) if ys.size else 0.0
        m, c = np.polyfit(xs, ys, 1)
        return float(m), float(c)

    return {
        "break_g": int(x[bk]),
        "left":  _line(x[:bk + 1], y[:bk + 1]),
        "right": _line(x[bk:], y[bk:]),
        "sse": float(total[k]),
        "lo_g": int(x[0]),
        "hi_g": int(x[-1]),
    }


def _draw_two_segment_fit(ax, sig, off: int, aoi: tuple, n_ev: int) -> None:
    """Shade the extended AOI and draw the fit with a grid-rounded breakpoint.

    Called once per arctan panel with that panel's own signal, so each input is
    fitted on its own curve. Must run before :func:`_decorate`, which builds
    the legend.
    """
    lo_g, hi_g = extended_aoi(aoi, n_ev)

    if lo_g <= 0:
        return
    ax.axvspan(lo_g, hi_g, color="tab:gray", alpha=0.10,
               label=f"AOI+width [{lo_g},{hi_g}]")
    fit = two_segment_fit(sig, off, lo_g, hi_g)
    if fit is None:
        return
    color, ls = SEG_FIT_STYLE
    # Snap the plotted breakpoint to the nearest grid multiple (ties round up).
    bk = ((fit["break_g"] + ROUND_GRID //2) // ROUND_GRID) * ROUND_GRID
    for (m, c), (x0, x1), lab in (
        (fit["left"],  (fit["lo_g"], bk),       f"2-seg fit, break={bk}"),
        (fit["right"], (bk, fit["hi_g"]), None),
    ):
        xs = np.array([x0, x1], dtype=np.float64)
        ax.plot(xs, m * xs + c, color=color, ls=ls, lw=2.0, alpha=0.9, label=lab)
    ax.axvline(bk, color=color, lw=1.2, ls=":", alpha=0.8)


def plot_one(stem: str, E: np.ndarray, inputs: dict, aoi: tuple,
             hidden_dim: int, out_path: Path,
             preds: dict,
             sclass: dict | None = None,
             gap: tuple[int, float] | None = None) -> None:
    """``gap`` is the ``(count, drop)`` pair from :func:`gap_rank_in_aoi``;
    it is drawn on the raw panel only, since it is read off raw ``E``."""
    if isinstance(E, torch.Tensor):
        E = E.detach().cpu().numpy()
    inputs = {
        k: (sig.detach().cpu().numpy() if isinstance(sig, torch.Tensor) else sig, off)
        for k, (sig, off) in inputs.items()
    }
    # raw spectrum, the atan signal the detector runs on, and its slopes.
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    axes = axes.ravel()
    for ax in axes:
        ax.tick_params(axis="both", labelsize=16)
    i_peak = int(preds["peak_floor"][0])
    d_hat = _floor_grid(i_peak) if i_peak > 0 else -1
    d_hat_str = f"d_hat={d_hat}" if d_hat > 0 else ""
    sc_label = sclass["spectrum_class"] if sclass else None
    sc_color = SPECTRUM_CLASS_COLORS.get(sc_label, "black")
    sc_str = ""
    if sclass:
        sc_str = (f"\nspectrum_class={sc_label}  "
                  f"snapback={_fmt_sc(sclass['sc_snapback'])}  "
                  f"sad/pk={_fmt_sc(sclass['sc_r'])}")
    gap_k, gap_drop = gap if gap is not None else (-1, float("nan"))
    if gap_k > 0:
        cliff = "cliff" if gap_drop >= GAP_RANK_MIN_DROP else "no cliff"
        sc_str += f"  gap_rank={gap_k} (drop {gap_drop:.2f}, {cliff})"
    fig.suptitle(
        f"{stem}\ntrue={hidden_dim}\n"
        f"{d_hat_str}{sc_str}",
        fontsize=18, y=0.99,
    )

    def _decorate(ax):
        if hidden_dim > 0:
            ax.axvline(hidden_dim, color="black", lw=1.4, alpha=0.85,
                       label=f"true={hidden_dim}")
        # d_hat is an eigenvalue index, so it means the same thing on every
        # panel and is drawn on all of them.
        if d_hat > 0:
            color, ls = D_HAT_STYLE
            ax.axvline(d_hat, color=color, lw=1.3, ls=ls, alpha=0.9,
                       label=f"d_hat={d_hat}")
        ax.set_xlabel("index", fontsize=17)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=14, loc="best", ncol=2)
        ax.title.set_fontsize(19)

    ax = axes[0]
    n_ev = len(E)
    x_full = np.arange(1, n_ev + 1)
    ax.plot(x_full, E, lw=1.0, color="tab:blue")
    ax.set_yscale("log")
    ax.set_ylabel("eigenvalue", fontsize=17)
    ax.set_title(INPUT_PANEL_TITLES["raw"])
    # gap_rank is searched over the same extended AOI the two-segment fit
    # uses, so shade that window here too and mark the largest log-drop.
    lo_g, hi_g = extended_aoi(aoi, n_ev)
    if lo_g > 0:
        ax.axvspan(lo_g, hi_g, color="tab:gray", alpha=0.10,
                   label=f"AOI+width [{lo_g},{hi_g}]")
    if gap_k > 0:
        color, ls = GAP_RANK_STYLE
        ax.axvline(gap_k, color=color, lw=1.3, ls=ls, alpha=0.9,
                   label=f"gap_rank={gap_k} (drop {gap_drop:.2f})")
        if gap_k < n_ev:
            ax.plot([gap_k, gap_k + 1], [E[gap_k - 1], E[gap_k]], "o",
                    color=color, ms=5, alpha=0.9)
    _decorate(ax)

    ax = axes[1]
    a_sig = inputs["arctan"][0]
    a_sig = a_sig.detach().cpu().numpy() if isinstance(a_sig, torch.Tensor) else np.asarray(a_sig)
    ax.plot(x_full, a_sig, lw=1.0, color="tab:blue")
    ax.set_ylabel("arctan(E)", fontsize=17)
    ax.set_title(INPUT_PANEL_TITLES["arctan"])
    _draw_two_segment_fit(ax, a_sig, 0, aoi, n_ev)
    _decorate(ax)

    def _plot_slopes_panel(ax):
        slopes_sig, slopes_off = inputs["slopes"]
        if len(slopes_sig) > 0:
            x_slopes = np.arange(slopes_off + 1, slopes_off + 1 + len(slopes_sig))
            slopes_plot = _smooth_np(slopes_sig, SMOOTH_WINDOW)
            ax.plot(x_slopes, slopes_plot, lw=1.0, color="tab:purple")
            ax.axhline(0.0, color="black", lw=0.6, alpha=0.5)
            # Symmetric-log y: the slopes are almost entirely negative, so a plain
            # log axis would drop the whole curve. linthresh at the 5th-percentile
            # magnitude (rounded down to a decade so ticks land on powers of 10)
            # keeps the linear band to near-zero values and the dip log-scaled.
            finite = slopes_plot[np.isfinite(slopes_plot)]
            mags = np.abs(finite)
            mags = mags[mags > 0]
            if mags.size:
                linthresh = 10.0 ** np.floor(np.log10(max(float(np.percentile(mags, 5)), 1e-12)))
            else:
                linthresh = 1e-12
            ax.set_yscale("symlog", linthresh=linthresh, linscale=0.5)
            # Without explicit limits the positive half of the symlog axis is
            # empty decades. Clamp the top to just past the largest positive
            # value (or the linear band if none).
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                bottom = -abs(lo) * 3.0 if lo < 0 else -linthresh
                top = hi * 3.0 if hi > linthresh else linthresh
                ax.set_ylim(bottom, top)
            # Only the AOI basin is drawn; the individual landmarks are read
            # off the shaded span and its label.
            saddle_g, peak_g = aoi
            if saddle_g > 0 and peak_g > saddle_g:
                # Tint the AOI by the predictiveness class — the class is read off
                # this basin (relative width + snapback past the peak).
                ax.axvspan(saddle_g, peak_g,
                           color=sc_color if sclass else "tab:orange", alpha=0.10,
                           label=f"AOI [{saddle_g},{peak_g}]")
        ax.set_ylabel("d arctan(E) / d idx", fontsize=17)
        ax.set_title(INPUT_PANEL_TITLES["slopes"] + " (symlog y)")
        _decorate(ax)
        if sclass:
            ax.text(
                0.02, 0.03,
                f"{sc_label}",
                transform=ax.transAxes, fontsize=14, color=sc_color,
                va="bottom", ha="left", zorder=5,
                bbox=dict(boxstyle="round", facecolor="white",
                          edgecolor=sc_color, alpha=0.85),
            )

    _plot_slopes_panel(axes[2])

    fig.tight_layout(rect=(0, 0, 1, 0.93 if sclass else 0.94))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _rel_err(pred: int, td: int) -> float:
    if td <= 0:
        return float("nan")
    return abs(pred - td) / td


# --------------------------- sampling --------------------------------------
@torch.no_grad()
def _prepare_distributions(logits: torch.Tensor, temperature: float,
                           ctm_temperature: float, device: str,
                           logprob_dtype: torch.dtype = torch.bfloat16):
    """Compute sampling probabilities on the device and cache CTM log-probs on CPU.

    Both softmaxes run on the sampling device in bounded row chunks. The CPU
    cache is stored in ``logprob_dtype`` (bfloat16 by default) before conversion to the float64 CTM.
    No original logits are needed after this function returns.
    """
    sample_probs = log_probs_cpu = data = tempered = probs = lp = None
    try:
        sample_probs = torch.empty(logits.shape, dtype=torch.float32, device=device)
        log_probs_cpu = torch.empty(logits.shape, dtype=logprob_dtype, device="cpu")
        for s in range(0, logits.shape[0], PREPARE_PROMPT_CHUNK):
            e = min(s + PREPARE_PROMPT_CHUNK, logits.shape[0])
            data = logits[s:e].to(device=device, dtype=torch.float32)
            tempered = data / temperature if temperature != 1.0 else data
            probs = F.softmax(tempered, dim=-1)
            sample_probs[s:e].copy_(probs)
            probs = None
            if ctm_temperature != temperature:
                tempered = data / ctm_temperature if ctm_temperature != 1.0 else data
            lp = F.log_softmax(tempered, dim=-1)
            log_probs_cpu[s:e].copy_(lp.to(dtype=logprob_dtype))
            data = tempered = lp = None
        return sample_probs, log_probs_cpu
    finally:
        # Also release partially prepared outputs when a chunk allocation fails.
        logits = sample_probs = log_probs_cpu = data = tempered = probs = lp = None


def _sat_stats(sat: torch.Tensor) -> dict:
    if sat.numel() == 0:
        nan = float("nan")
        return {"sat_min": nan, "sat_median": nan, "sat_mean": nan, "sat_max": nan}
    return {
        "sat_min": float(sat.min()), "sat_median": float(sat.median()),
        "sat_mean": float(sat.mean()), "sat_max": float(sat.max()),
    }


def _step_changes(previous: dict | None, current: dict) -> dict:
    """Direct changes; missing AOIs or non-finite values have no comparison."""
    changes = {"delta_peak": None, "delta_sc_r": None, "delta_snapback": None}
    if previous is None or previous["i_peak"] <= 0 or current["i_peak"] <= 0:
        return changes
    for delta, metric in (("delta_peak", "i_peak"), ("delta_sc_r", "sc_r"),
                          ("delta_snapback", "sc_snapback")):
        before, after = previous[metric], current[metric]
        if math.isfinite(before) and math.isfinite(after):
            changes[delta] = after - before
    return changes


def _step_quotas(budget: int, n_steps: int, n_p: int) -> tuple[int, int]:
    """``(step_samples, draws_per_prompt)``: the budget split into ``n_steps``
    equal quotas, each spread evenly over the ``n_p`` prompts. Integer
    rounding leaves ``budget - n_steps * n_p * draws_per_prompt`` unspent."""
    per_prompt = int(budget) // (int(n_steps) * int(n_p))
    return per_prompt * int(n_p), per_prompt


@torch.no_grad()
def run_sampling(
    logits: torch.Tensor | Path,
    *,
    output_dir: Path,
    output_stem: str,
    temperature: float,
    ctm_temperature: float | None,
    samples_per_iter: int,
    seed: int,
    device: str,
    gram_dtype: torch.dtype,
    budget: int,
    max_prompts: int,
    n_steps: int = DEFAULT_N_STEPS,
    eval_every: int = DEFAULT_EVAL_EVERY,
    logprob_dtype: torch.dtype = LOGPROB_DTYPES[DEFAULT_LOGPROB_DTYPE],
    plot_dir: Path | None = None,
) -> dict:
    """Draw the same number of samples from every prompt at every step,
    evaluate the CTM spectrum after each evaluated step, and save the run
    trajectory. Returns the run's result dict."""
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive; got {n_steps}")
    if eval_every <= 0:
        raise ValueError(f"eval_every must be positive; got {eval_every}")
    if max_prompts <= 0:
        raise ValueError(f"max_prompts must be positive; got {max_prompts}")
    if budget <= 0:
        raise ValueError(f"budget must be positive; got {budget}")
    if samples_per_iter <= 0:
        raise ValueError(f"samples_per_iter must be positive; got {samples_per_iter}")
    if logprob_dtype not in LOGPROB_DTYPES.values():
        raise ValueError(f"logprob_dtype must be one of {sorted(LOGPROB_DTYPES)}; "
                         f"got {logprob_dtype}")

    run_tag = output_stem
    target_dir = output_dir / run_tag
    target_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = plot_dir or (target_dir / PLOTS_SUBDIR)
    plot_dir.mkdir(parents=True, exist_ok=True)
    traj_path = target_dir / f"{run_tag}{TRAJECTORY_JSON_SUFFIX}"
    true_dim = true_dim_for(output_stem)
    sample_probs = log_probs_cpu = presence = sat = draws = incumbent = g_sample = None
    try:
        if isinstance(logits, Path):
            logits = load_logits(logits, max_prompts)
        ctm_T = float(temperature if ctm_temperature is None else ctm_temperature)
        print(f"  CTM log-probs at temperature {ctm_T:g}"
              + ("" if ctm_temperature is None else f" (--ctm-temperature; sampling at {temperature:g})"))
        sample_probs, log_probs_cpu = _prepare_distributions(
            logits, temperature, ctm_T, device, logprob_dtype)
        logits = None
        print(f"  CTM log-probs cached on CPU: {tuple(log_probs_cpu.shape)} {log_probs_cpu.dtype} "
              f"({log_probs_cpu.numel() * log_probs_cpu.element_size() / 2**30:.1f} GiB)")

        pool, vocab_size = int(sample_probs.shape[0]), int(sample_probs.shape[1])
        n_p = min(int(max_prompts), pool)
        if max_prompts > pool:
            print(f"  n_max={max_prompts} exceeds the {pool}-prompt pool; using all {pool}")
        budget = int(budget)
        step_samples, per_prompt = _step_quotas(budget, n_steps, n_p)
        if per_prompt <= 0:
            raise ValueError(f"budget {budget} is too small for {n_steps} steps x {n_p} prompts: "
                             f"fewer than one draw per prompt per step")
        planned_total = step_samples * n_steps
        print(f"  sampling schedule: {n_steps} steps x {step_samples:,} samples = "
              f"{per_prompt:,} draws per prompt per step over all {n_p} prompts "
              f"({planned_total:,} = {planned_total / budget:.2%} of the budget; "
              f"{budget - planned_total:,} lost to rounding)")

        presence = torch.zeros(n_p, vocab_size, dtype=torch.bool, device=device)
        sat = torch.zeros(n_p, dtype=torch.float64, device=device)
        draws = torch.zeros(n_p, dtype=torch.int64, device=device)
        g_sample = torch.Generator(device=device).manual_seed(seed + 1)
        prompt_chunk = SAMPLE_PROMPT_CHUNK
        all_rows = torch.arange(n_p, device=device)

        total = 0
        previous_ev: dict | None = None
        incumbent: tuple[torch.Tensor, torch.Tensor] | None = None
        shrink_guard_active = False
        steps: list[dict] = []
        t_start = time.time()
        config = {
            "budget": budget, "n_p": int(n_p), "n_steps": int(n_steps),
            "true_dim": int(true_dim), "temperature": float(temperature),
            "seed": int(seed), "pred_smooth": bool(PRED_SMOOTH),
        }

        def write_trajectory() -> None:
            with open(traj_path, "w") as f:
                json.dump({"run_tag": run_tag, "config": config, "steps": steps}, f, indent=2)

        def draw_step(step: int) -> dict:
            """Give every prompt ``per_prompt`` draws, in minibatches of at most
            ``samples_per_iter`` draws per prompt, chunked over rows so
            multinomial's temporaries stay bounded."""
            nonlocal total
            before = _sat_stats(sat)
            print(f"[sampling] step {step}/{n_steps}: {per_prompt:,} draws for each of the "
                  f"{n_p} prompts ({step_samples:,} samples); "
                  f"sat min={before['sat_min']:.4f} med={before['sat_median']:.4f}")
            t0 = time.time()
            spent = 0
            minibatches = 0
            left = per_prompt
            while left > 0:
                draw = min(samples_per_iter, left)
                for c0 in range(0, n_p, prompt_chunk):
                    rows = all_rows[c0:c0 + prompt_chunk]
                    probs_chunk = sample_probs.index_select(0, rows)
                    chunk_samples = torch.multinomial(
                        probs_chunk, num_samples=draw, replacement=True, generator=g_sample)
                    presence[rows.unsqueeze(1), chunk_samples] = True
                    # Saturation = mass of the observed tokens under the sampling
                    # distribution, reduced in bounded vocabulary chunks.
                    probs_chunk.mul_(presence.index_select(0, rows))
                    sat_chunk = torch.zeros(int(rows.numel()), dtype=torch.float64, device=device)
                    for v0 in range(0, vocab_size, SAT_VOCAB_CHUNK):
                        sat_chunk += probs_chunk[:, v0:v0 + SAT_VOCAB_CHUNK].sum(dim=-1).to(torch.float64)
                    sat[rows] = sat_chunk
                    del chunk_samples, probs_chunk, sat_chunk
                draws.add_(draw)
                spent += draw * n_p
                total += draw * n_p
                left -= draw
                minibatches += 1
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            seconds = time.time() - t0
            after = _sat_stats(sat)
            print(f"  step {step} sampled {spent:,} in {minibatches} minibatches over {n_p} prompts: "
                  f"sat min {before['sat_min']:.4f} -> {after['sat_min']:.4f}, "
                  f"med {before['sat_median']:.4f} -> {after['sat_median']:.4f} [{seconds:.1f}s]")
            return {"samples_spent": int(spent), "sample_seconds": seconds}

        def empty_eval() -> dict:
            nan = float("nan")
            return {
                "rows": 0, "cols": 0, "i_peak": -1, "d_hat": -1, "aoi_saddle": -1, "aoi_peak": -1,
                "spectrum_class": "C0_not_predictive", "sc_snapback": nan, "sc_r": nan,
                "rel_err": nan, "evaluated": False,
            }

        def evaluate(step: int) -> dict:
            """Estimation half: cut the CTM
            as the maximum edge biclique, decompose its Gram, read the AOI
            peak and spectrum class off the eigenvalues, and plot the step."""
            nonlocal incumbent, shrink_guard_active
            rec = empty_eval()
            rec["evaluated"] = True
            print(f"  biclique search over n_p={n_p} prompts at total={total}: "
                  f"min_non_nan_per_col={MIN_NON_NAN_PER_COL}")
            t_prune = time.time()
            guard_now = shrink_guard_active and step > SHRINK_GUARD_MIN_STEP
            try:
                rows, cols, res = prune_common_rows_cols(
                    presence, min_non_nan_per_col=MIN_NON_NAN_PER_COL,
                    incumbent=incumbent, shrink_guard_active=guard_now)
            except EmptyCTM as exc:
                print(f"  empty CTM ({exc}): no spectrum, no peak at this step")
                return rec
            except torch.OutOfMemoryError:
                if device.startswith("cuda"):
                    print(f"  CUDA OOM in the biclique search at n_p={n_p}: "
                          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated, "
                          f"{torch.cuda.memory_reserved() / 2**30:.1f} GiB reserved", flush=True)
                raise
            sort_seconds = time.time() - t_prune
            rec["rows"], rec["cols"] = int(res.n_rows), int(res.n_cols)
            print(f"    biclique: {res}  [won_by={res.info['won_by']}]")
            incumbent = (rows.clone(), cols.clone())
            n_rows = int(rows.numel())

            ctm = torch.empty(n_rows, int(cols.numel()), dtype=torch.float64, device=device)
            rows_cpu, cols_cpu = rows.cpu(), cols.cpu()
            for c0 in range(0, n_rows, prompt_chunk):
                e = min(c0 + prompt_chunk, n_rows)
                lp = log_probs_cpu[rows_cpu[c0:e, None], cols_cpu[None, :]]
                ctm[c0:e].copy_(lp)
                del lp
            t_gram = time.time()
            gram_M = smaller_side_gram(ctm.to(gram_dtype))
            gram_seconds = time.time() - t_gram
            t_eig = time.time()
            E = torch.linalg.eigvalsh(gram_M.to(device=DEVICE)).flip(0)
            eig_seconds = time.time() - t_eig
            inputs, aoi = derive_inputs(E)
            sclass = classify_spectrum(inputs, aoi)
            preds = run_predictors(aoi)
            gap_k, gap_drop = gap_rank_in_aoi(E, aoi)
            i_peak = int(preds["peak_floor"][0])
            d_hat = _floor_grid(i_peak)
            E_np = E.detach().cpu().numpy()
            nan = float("nan")
            rec.update({
                "i_peak": i_peak, "d_hat": int(d_hat),
                "aoi_saddle": int(aoi[0]), "aoi_peak": int(aoi[1]),
                "spectrum_class": sclass["spectrum_class"],
                "sc_snapback": float(sclass["sc_snapback"]),
                "sc_r": float(sclass["sc_r"]),
                "rel_err": float(_rel_err(d_hat, true_dim)) if d_hat > 0 else nan,
            })
            shrink_guard_active = sclass["spectrum_class"].startswith("C2")
            print(f"  ctm=({rec['rows']},{rec['cols']}) gram={tuple(gram_M.shape)} "
                  f"peak={i_peak} (floor {d_hat}) class={sclass['spectrum_class']} "
                  f"snapback={_fmt_sc(sclass['sc_snapback'])} "
                  f"gap_rank={gap_k} (drop {_fmt_sc(gap_drop)}) true={true_dim} "
                  f"[sort {sort_seconds:.1f}s gram {gram_seconds:.1f}s eig {eig_seconds:.1f}s]")
            sat_min = float(sat.min())
            key = f"{run_tag}_step{step:03d}_sampling_{n_p}p_sat{sat_min:.3f}_{total}"
            title = f"{run_tag}  step {step} (sampling)  n_p={n_p} ctm=({rec['rows']},{rec['cols']})"
            plot_one(title, E_np, inputs, aoi, true_dim, plot_dir / f"{key}.png",
                       preds, sclass=sclass, gap=(gap_k, gap_drop))
            del ctm, gram_M, E, inputs, rows, cols, res
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            return rec

        print(f"[sampling] pool={pool} prompts x {vocab_size} vocab  budget={budget:,}; "
              f"{n_steps} steps x {step_samples:,} samples ({per_prompt:,} per prompt per step)  "
              f"n_p={n_p} eval_every={eval_every} grid={ROUND_GRID} "
              f"snap_trust={SNAP_TRUST:g} true_dim={true_dim}")

        ev: dict = empty_eval()
        for step in range(1, n_steps + 1):
            t_step = time.time()
            act = draw_step(step)
            is_last = step == n_steps
            if is_last or step % eval_every == 0:
                ev = evaluate(step)
            else:
                # Not an evaluation step: nothing is recorded for it, and the
                # previous spectrum is only reused for the progress line below.
                ev = dict(ev)
                ev["evaluated"] = False
                print(f"  step {step}: evaluation skipped (every {eval_every} steps)")
            changes = _step_changes(previous_ev, ev) if ev["evaluated"] else \
                {"delta_peak": None, "delta_sc_r": None, "delta_snapback": None}
            if ev["evaluated"]:
                previous_ev = ev
                steps.append({"step": step, "draws_per_prompt_total": int(per_prompt * step),
                              **{k: ev[k] for k in STEP_RESULT_FIELDS}})
                write_trajectory()
            print(f"  step {step}/{n_steps} done: n_p={n_p} total={total:,} "
                  f"({total / budget:.1%} of budget) spent={act['samples_spent']:,} "
                  f"peak={ev['i_peak']} ({_fmt_delta(changes['delta_peak'])}) "
                  f"sc_r={_fmt_sc(ev['sc_r'])} class={ev['spectrum_class']} "
                  f"snapback={_fmt_sc(ev['sc_snapback'])} "
                  f"[{time.time() - t_step:.1f}s: sample {act['sample_seconds']:.1f}s]")
            if is_last:
                print(f"  {step} steps played, {total:,} of {budget:,} samples drawn; stopping")

        d_hat, i_peak = int(ev["d_hat"]), int(ev["i_peak"])
        print(f"\n[sampling] done: d_hat={d_hat} (raw peak={i_peak}, true={true_dim}, "
              f"rel_err={ev['rel_err']:.3f}) class={ev['spectrum_class']} "
              f"snapback={_fmt_sc(ev['sc_snapback'])} "
              f"n_p={n_p} total_samples={total:,} ({total / budget:.1%} of budget) "
              f"steps={len(steps)} elapsed={time.time() - t_start:.1f}s")
        print(f"  wrote trajectory to {traj_path}")

        return {
            "trajectory_path": traj_path, "true_dim": int(true_dim), "d_hat": d_hat,
            "rel_err": float(ev["rel_err"]), "spectrum_class": ev["spectrum_class"],
            "sc_snapback": float(ev["sc_snapback"]),
        }
    finally:
        logits = sample_probs = log_probs_cpu = presence = sat = draws = None
        incumbent = g_sample = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logits_path", type=Path, nargs="?", default=DEFAULT_LOGITS_PATH,
                   help="A cached logits .pt file, or a directory of them")
    p.add_argument("output_dir", type=Path, nargs="?", default=DEFAULT_OUTPUT_DIR,
                   help="Directory under which per-run trajectories and plots are written")
    p.add_argument("--pattern", type=str, default=DEFAULT_PATTERN,
                   help="Glob applied inside logits_path when it names a directory")
    p.add_argument("--output-stem", type=str, default=None,
                   help="Run name; defaults to the logits file stem minus a trailing "
                        "'_logits'. Single-file runs only.")
    # --- sampling distribution ---
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--ctm-temperature", type=float, default=None,
                   help="Temperature of the CTM log-probs (default: the sampling --temperature)")
    p.add_argument("--samples-per-iter", type=int, default=DEFAULT_SAMPLES_PER_ITER,
                   help="Largest number of draws per prompt per multinomial call; a step's "
                        "per-prompt draws are split into minibatches of this size")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--gram-dtype", choices=list(GRAM_DTYPE_CHOICES), default=DEFAULT_GRAM_DTYPE)
    p.add_argument("--logprob-dtype", choices=list(LOGPROB_DTYPES), default=DEFAULT_LOGPROB_DTYPE,
                   help="Storage precision of the CPU log-prob cache the CTM is gathered from")
    # --- budget allocation ---
    p.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                   help="N_max: total sample budget summed over all prompts")
    p.add_argument("--max-prompts", type=int, default=DEFAULT_MAX_PROMPTS,
                   help="n_max: prompts in the set. Also the "
                        "read cap on the cache (only the first n_max rows are read)")
    p.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS,
                   help="K: equal budget slices; every step gives every prompt "
                        "budget // (K * n_p) draws")
    p.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY,
                   help="Run the biclique search, spectrum and plot every k-th step "
                        "(the last step is always evaluated); 1 = every step")
    # --- outputs ---
    p.add_argument("--plot-dir", type=Path, default=None,
                   help="Directory for the per-step PNGs; default <output_dir>/<run_tag>/plots. "
                        "Single-file runs only.")
    p.add_argument("--summary-path", type=Path, default=None,
                   help=f"Summary JSON path; default <output_dir>/{SUMMARY_JSON_NAME}")
    # --- shared common-token block / predictor knobs ---
    p.add_argument("--min-non-nan-per-col", type=int, default=MIN_NON_NAN_PER_COL,
                   help="Support floor: tokens seen by fewer prompts than this are "
                        "bulk-dropped before the biclique search")
    p.add_argument("--sweep-passes", type=int, default=SWEEP_PASSES,
                   help="Sweep ordering passes in the biclique search")
    p.add_argument("--shrink-guard-min-step", type=int, default=SHRINK_GUARD_MIN_STEP,
                   help="Steps at the start of a run whose searches ignore the CTM no-shrink guard")
    p.add_argument("--no-pred-smooth", dest="pred_smooth", action="store_false",
                   default=PRED_SMOOTH,
                   help="Skip the SMOOTH_WINDOW smoothing of the slopes in the saddle/peak search")
    return p.parse_args()


def run_one(args, logits_path: Path, *, output_stem: str, device: str,
            gram_dtype: torch.dtype, logprob_dtype: torch.dtype) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return run_sampling(
        logits_path,
        output_dir=args.output_dir,
        output_stem=output_stem,
        temperature=args.temperature,
        ctm_temperature=args.ctm_temperature,
        samples_per_iter=args.samples_per_iter,
        seed=args.seed,
        device=device,
        gram_dtype=gram_dtype,
        logprob_dtype=logprob_dtype,
        budget=int(args.budget),
        max_prompts=args.max_prompts,
        n_steps=args.n_steps,
        eval_every=args.eval_every,
        plot_dir=args.plot_dir,
    )


def summary_record(output_stem: str, result: dict) -> dict:
    return {
        "stem": output_stem, "true_dim": int(result["true_dim"]), "d_hat": int(result["d_hat"]),
        "rel_err": float(result["rel_err"]), "spectrum_class": result["spectrum_class"],
        "sc_snapback": float(result["sc_snapback"]),
        "trajectory": str(result["trajectory_path"]),
    }


def print_summary(records: list[dict], failures: list) -> None:
    if records:
        print("\n" + "=" * 100)
        print(f"=== SAMPLING SUMMARY over {len(records)} file(s) ===")
        print("=" * 100)
        hdr = (f"  {'stem':>22s} | {'true':>6s} {'d_hat':>6s} {'err':>6s} "
               f"{'class':>18s} {'snap':>6s}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in records:
            err = r["rel_err"]
            print(f"  {r['stem']:>22s} | {r['true_dim']:>6d} {r['d_hat']:>6d} "
                  f"{('--' if err != err else f'{err:.3f}'):>6s} "
                  f"{r['spectrum_class']:>18s} {_fmt_sc(r['sc_snapback']):>6s}")
        valid = [r["rel_err"] for r in records if r["rel_err"] == r["rel_err"]]
        if valid:
            print(f"  mean rel_err over {len(valid)} file(s) with an estimate: "
                  f"{sum(valid) / len(valid):.3f}")
    for stem, err in failures:
        print(f"  FAILED {stem}: {err}")


def write_summary_json(records: list[dict], failures: list, args, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "budget": int(args.budget), "max_prompts": args.max_prompts,
            "n_steps": int(args.n_steps), "temperature": args.temperature,
            "seed": args.seed, "pred_smooth": bool(PRED_SMOOTH),
        },
        "results": records,
        "failures": [{"stem": s, "error": e} for s, e in failures],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote summary to {out_path}")


def main():
    # The estimator functions read these module globals at call time.
    global PRED_SMOOTH, MIN_NON_NAN_PER_COL, SWEEP_PASSES, SHRINK_GUARD_MIN_STEP
    args = parse_args()
    PRED_SMOOTH = args.pred_smooth
    MIN_NON_NAN_PER_COL = args.min_non_nan_per_col
    SWEEP_PASSES = max(1, args.sweep_passes)
    SHRINK_GUARD_MIN_STEP = max(0, int(args.shrink_guard_min_step))
    device = SAMPLING_DEVICE
    gram_dtype = torch.float64 if args.gram_dtype == "float64" else torch.float32
    logprob_dtype = LOGPROB_DTYPES[args.logprob_dtype]
    if args.n_steps <= 0 or args.max_prompts <= 0 or args.eval_every <= 0:
        raise SystemExit(f"--n-steps {args.n_steps}, --max-prompts {args.max_prompts} and "
                         f"--eval-every {args.eval_every} must be positive")
    if _step_quotas(int(args.budget), args.n_steps, args.max_prompts)[1] <= 0:
        raise SystemExit(f"--budget {int(args.budget):,} gives fewer than one draw per prompt "
                         f"per step at {args.n_steps} steps x {args.max_prompts} prompts")

    logits_files = resolve_logits_files(args.logits_path, args.pattern)
    if len(logits_files) > 1:
        for flag, value in (("--output-stem", args.output_stem), ("--plot-dir", args.plot_dir)):
            if value is not None:
                raise SystemExit(f"{flag} applies to a single logits file, but "
                                 f"{args.logits_path} matched {len(logits_files)} files")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_path or (args.output_dir / SUMMARY_JSON_NAME)
    records: list[dict] = []
    failures: list = []
    for i, logits_path in enumerate(logits_files, 1):
        output_stem = args.output_stem or stem_from_logits_path(logits_path)
        print(f"\n{'=' * 78}\n[{i}/{len(logits_files)}] {output_stem}  <-  {logits_path}\n{'=' * 78}")
        t0 = time.time()
        try:
            result = run_one(args, logits_path, output_stem=output_stem, device=device,
                             gram_dtype=gram_dtype, logprob_dtype=logprob_dtype)
        except Exception as exc:
            traceback.print_exc()
            failures.append((output_stem, f"{type(exc).__name__}: {exc}"))
            print(f"[{output_stem}] FAILED after {time.time() - t0:.1f}s")
            traceback.clear_frames(exc.__traceback__)
        else:
            records.append(summary_record(output_stem, result))
            print(f"[{output_stem}] done in {time.time() - t0:.1f}s")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_summary_json(records, failures, args, summary_path)

    print_summary(records, failures)
    done = len(logits_files) - len(failures)
    print(f"\n{'=' * 78}\nProcessed {done}/{len(logits_files)} logits files")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
