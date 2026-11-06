# Sampling

Stage 3 of `nightvision_common_set`. This stage estimates each model's hidden
dimension from the next-token logits saved by `../collect_logits/`, without
loading the model. Tokens are sampled from the saved distributions under a
fixed total budget, which is spread evenly over all prompts.

## Terminology

- **Saturation**: the share of a prompt's probability mass that its sampled
  tokens cover. It is the sum of the sampling probabilities of every token the
  prompt has drawn at least once.
- **Presence matrix**: a prompts x tokens table recording which prompt has
  drawn which token.
- **Common-token block / biclique**: a set of prompts and a set of tokens in
  which every prompt has drawn every token. That is an all-observed rectangle
  of the presence matrix. The search maximizes its area (rows x columns).
- **CTM (common-token matrix)**: the log-probabilities of the block's tokens
  under the block's prompts. Its Gram matrix is taken on the smaller side.
- **Spectrum**: the eigenvalues of the CTM's Gram matrix, in descending order.
  The analysis takes their arctan and computes a least-squares slope over a
  sliding window of about 100 indices. It then finds a *saddle* (a local
  minimum of the slope) and the *peak* that follows it (the first index where
  the slope has recovered 90% of the way back). The peak index is the raw
  estimate.
- **AOI (area of interest)**: the saddle-to-peak window of the spectrum.
- **Snapback**: how far the slope recovers after the saddle, relative to the
  saddle's depth. A high snapback indicates a sharp, trustworthy peak.

## Files

| File | Description |
|---|---|
| `run_sampling.py` | The whole pipeline: sampling loop, biclique search, spectrum analysis, plots, summary |
| `run_sampling.sh` | Bash wrapper that runs the script over every logits file, once per `LOGPROB_DTYPES` entry |
| `results/` | Output root (not tracked in git) |

The `CONFIGURATION` block near the top of `run_sampling.py` holds every default
and tuning constant.

## How it works

1. **Load** the first `--max-prompts` rows of a `(prompts, vocab)` logits
   tensor, convert them to sampling probabilities at `--temperature`, and
   cache the log-probabilities on CPU in `--logprob-dtype` so the logits can
   be released.
2. **Sample.** The `--budget` is split into `--n-steps` equal quotas and every
   step gives every prompt the same `budget // (n_steps * n_p)` draws,
   regardless of its saturation, its membership in the block, or what the
   spectrum says. The run plays all `--n-steps` steps and stops when the budget
   is spent (integer rounding leaves a small remainder unspent).
3. **Common-token block.** After each step, or every `--eval-every` steps with
   the last step always included, the presence matrix is pruned to its largest
   all-observed biclique via a column-support sweep (`--sweep-passes`) plus
   incumbent and peel passes. The cached log-probabilities of that block form
   the CTM. This evaluation is the expensive part at full `n_p`, which is what
   `--eval-every` is for.
4. **Spectrum.** The eigenvalues of the CTM's Gram matrix (in `--gram-dtype`)
   give the area-of-interest landmarks, a spectrum class (`C3_good`,
   `C2_marginal`, `C1_underestimate`, `C0_not_predictive`), a snapback score
   and the peak index. The estimate `d_hat` is the raw peak floored onto a
   grid of `ROUND_GRID` (128).

The true hidden size is looked up in the repository's `model_configs.json`
(found by searching upward from the script directory) through the
`_STEM_TO_HF` table. It is used only to report the error (`rel_err`). Stems
missing from the table report `true_dim = -1`.

## Usage

```bash
python run_sampling.py [logits_path] [output_dir] [options]
```

`logits_path` is a single `.pt` file or a directory of them matched by
`--pattern` (default `*.pt`); the default is `../collect_logits/results`.
Each file is processed as its own run named after the file stem minus
`_logits`. `output_dir` defaults to `results/`.

### Sampling and allocation

| Flag | Default | Description |
|---|---|---|
| `--budget` | `2e10` | Total samples summed over all prompts |
| `--max-prompts` | `20000` | Prompts in the set; also how many rows of the cache are read |
| `--n-steps` | `100` | Equal budget slices |
| `--eval-every` | `1` | Evaluate (biclique search, spectrum, plot) every k-th step; the last step is always evaluated |
| `--samples-per-iter` | `10000` | Largest number of draws per prompt per multinomial call; a step's per-prompt draws are split into minibatches of this size |
| `--temperature` | `2.0` | Sampling temperature applied to the logits |
| `--ctm-temperature` | sampling temperature | Temperature of the CTM log-probs (`1.0` = untempered) |
| `--seed` | `42` | RNG seed |

### Precision, block search and prediction

| Flag | Default | Description |
|---|---|---|
| `--gram-dtype` | `float64` | Precision of the Gram product and eigendecomposition |
| `--logprob-dtype` | `bfloat16` | Storage precision of the CPU log-prob cache (`float32` doubles its memory) |
| `--min-non-nan-per-col` | `2` | Tokens seen by fewer prompts are dropped before the biclique search |
| `--sweep-passes` | `2` | Column-ordering passes in the biclique search |
| `--shrink-guard-min-step` | `10` | Steps at the start of a run that ignore the block no-shrink guard |
| `--no-pred-smooth` | off | Run the saddle/peak search on raw slopes instead of slopes smoothed with a ~100-index moving average (the wrapper passes this by default); the snapback is always read off the smoothed slopes |

### Outputs

| Flag | Default | Description |
|---|---|---|
| `--pattern` | `*.pt` | Glob applied inside `logits_path` when it names a directory |
| `--output-stem` | file stem minus `_logits` | Run name; single-file runs only |
| `--plot-dir` | `<output_dir>/<run>/plots` | Per-step PNGs; single-file runs only |
| `--summary-path` | `<output_dir>/sampling_summary.json` | One record per processed file |

## Output format

For each logits file the script writes `<output_dir>/<run>/`:

- `<run>_sampling.json`: the trajectory. Top level: `run_tag`, `config`
  (`budget`, `n_p`, `n_steps`, `true_dim`, `temperature`, `seed`, `pred_smooth`;
  the published files in `results/commonset_results/` keep only the first four) and `steps`, one record per evaluated step: `step`,
  `draws_per_prompt_total`, `d_hat`, `aoi_saddle`, `aoi_peak`, `spectrum_class`,
  `sc_snapback` and `rel_err`. A step with no estimate has `d_hat = -1` and
  `rel_err = NaN`. The headline estimate of a run is its last step; no earlier
  estimate is carried forward. The file is rewritten after every evaluated step,
  so a killed run keeps its partial trajectory, and a finished run has its last
  step at `step == n_steps`. Other diagnostics (block shape, saturation, largest
  gap, timings) go to the log only.
- `plots/<run>_step<NNN>_sampling_<n_p>p_sat<min sat>_<samples>.png`: one figure
  per evaluated step showing the raw spectrum, its arctan transform and the
  windowed slopes with the AOI, the estimate and the true dimension marked.

`<output_dir>/sampling_summary.json` collects each run's last-step `d_hat`,
`rel_err`, `spectrum_class` and `sc_snapback`, plus the shared `config` and a
`failures` list, and is rewritten after each
file so a sweep killed midway keeps its results. A table of the same numbers
is printed at the end; the process exits non-zero if any file failed.

## Wrapper script

```bash
bash experiments/nightvision_common_set/sampling/run_sampling.sh [extra run_sampling.py args...]
```

Runs the script with the options used for the published results. It locates
the repository root from its own location, so it can be started from any
directory, and needs one GPU. The wrapper sweeps `$LOGPROB_DTYPES`
into `$OUT_ROOT/<RUN_NAME>_<dtype>/` with `--temperature 2.0 --seed 42` and
`--no-pred-smooth` unless `PRED_SMOOTH=true`; extra arguments are appended
(for example `--samples-per-iter 2000` on a GPU with less memory).

| Variable | Default | Description |
|---|---|---|
| `LOGITS` | `../collect_logits/results` | Logits file or directory |
| `PATTERN` | `*.pt` | Glob inside `LOGITS` |
| `OUT_ROOT` | `sampling/results` | Parent of the per-dtype output directories |
| `RUN_NAME` | `full_run` | Output directory prefix |
| `LOGPROB_DTYPES` | `bfloat16` | Space-separated `--logprob-dtype` values to sweep |
| `SWEEP_PASSES` | `2` | `--sweep-passes` |
| `MAX_PROMPTS` | `20000` | `--max-prompts` |
| `BUDGET` | `20000000000` | `--budget` |
| `N_STEPS` | `10` | `--n-steps` |
| `EVAL_EVERY` | `1` | `--eval-every` |
| `PRED_SMOOTH` | `false` | `true` keeps slope smoothing on |
| `PROJECT_DIR`, `SAMPLING_DIR`, `PYTHON` | derived | Repository root, this directory, and interpreter |
