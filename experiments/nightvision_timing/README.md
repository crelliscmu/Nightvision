# NightVision Timing Attack

Phase 2 of NightVision: estimate a model's depth `L` and parameter count `P`
from end-to-end prefill latency, following the timing attack in the paper
(Section 4.2, "Phase 2: NightVision Timing Attack", and Appendix B.3, "Timing
Attack Details").
The reported results use 14 dense models (see "Models excluded" below).

## Files

| File | Description |
|---|---|
| `collect_prefill.py` | Serve one model with vLLM and record per-request latencies over the sequence-length sweep |
| `collect_all_prefill.sh` | Bash loop: `collect_prefill.py` for every model in this directory's `model_configs.json` (the 14 timing models), serially on one GPU |
| `fit_timing.py` | Fit the prefill scaling relations leave-one-out and print/save the fit's one estimated quantity per model (`L_hat` or `P_hat`, see `--fit`) |
| `fit_timing_cross.py` | Fit across two platforms, in both directions: the reference relation on one, the target measurements on the other. Writes `results/timing_fits_cross/` |
| `fit_timing_budgets.py` | The `layers` and `params` fits with the `d_hat` of each sampling budget, skipping any budget where a model has no `d_hat`. Writes `results/timing_fits_each_budget/` |
| `../../results/timing_results_prefill/` | `<model>_timing.json` per model from the prefill collection on the H100, written by `collect_prefill.py` and read by the fit scripts |
| `../../results/timing_results_prefill_h200/` | The same measurements of the 14 timing models on the H200 |
| `../../results/timing_fits/` | `timing_fit_<fit>[_dhat].json` written by `fit_timing.py` (its `scaling_law_<fit>_prefill.png` plots are not tracked in git) |
| `../../results/timing_fits_cross/` | `timing_cross_<ref>_to_<tgt>_<fit>[_dhat].json` written by `fit_timing_cross.py` |
| `plot_timing.py` | Plot the trimmed-mean latency against prompt length for one timing file |
| `plot_combined.py` | Plot the latency curves of several timing files on one figure |

## Stage 1: prefill timing collection

`collect_prefill.py <HF model>` reads the model's `config.json` from the
HuggingFace cache, builds the sequence-length sweep and serves the model with
`api.vllm_server.VLLMServer`:

- 24 uniformly spaced lengths from 1 to 100,000 tokens, the same for every
  model. Models whose context window is shorter have `max_position_embeddings`
  raised in the served config, so the window does not limit the sweep. Outputs
  past the trained context are meaningless, but prefill compute, and so latency,
  does not depend on them.
- The crossover `l* = floor((d + d_kv + 1.5 d_ffn) * 0.6)` with
  `d_kv = n_kv * d / n_heads` is recorded for reference. It does not set the sweep.
- Prompts of `l` tokens are built by repeating a fixed passage, truncating with
  the model's tokenizer, and replacing the first tokens with a random nonce so
  no two requests share a prefix. The actual `prompt_tokens` reported by the
  server (which includes the chat template) is recorded and used by the fit.
- For `l = 1` and each sweep length: 2 warmup requests, then 15 sequential
  batch-size-one chat requests with `max_tokens=1`, each timed client-side
  with `time.perf_counter()`.
- The server runs with prefix caching disabled (so repeated prompts are
  always re-prefilled) and `max_num_seqs=1` with `max_model_len` just above
  the longest prompt, so the whole prompt is prefilled in one step.

Output: `results/timing_results_prefill/<org>_<model>_timing.json` with the architecture fields, the
sweep, and the raw latencies of every request at every length.

```bash
source .venv/bin/activate
python experiments/nightvision_timing/collect_prefill.py Qwen/Qwen3-8B
bash experiments/nightvision_timing/collect_all_prefill.sh   # every dense model
```

Both need a GPU node; on a Slurm cluster run them inside a job. `collect_all_prefill.sh`
skips models whose timing file already exists, so an interrupted loop can be
rerun, and a failing model does not stop the rest. Environment overrides:
`OUTPUT_DIR` (default `results/timing_results_prefill/`), `VENV` (default `<repo root>/.venv`).

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `results/timing_results_prefill/` | Where the JSON is written |
| `--num-trials` | `15` | Timed requests per length |
| `--min-len`, `--max-len`, `--num-lens` | `1`, `100000`, `24` | Sweep bounds and number of lengths |
| `--ignore-max-pos` | on | Allow `--max-len` beyond `max_position_embeddings` by overriding it in the served config |
| `--gpu-mem` | `0.9` | vLLM GPU memory utilization |

## Stages 2-3: fitting and depth estimation

```bash
python experiments/nightvision_timing/fit_timing.py [--fit FIT] [--commonset-dir DIR]
```

Every model is the target once, with all other models as the reference set
(leave-one-out). The measured time `T` is the `max_tokens=1` latency less the
`l = 1` baseline. Each length is reduced to its 10%-trimmed mean and standard
error `std / sqrt(N_kept)`, and every fit is weighted least squares with weights
`1 / sigma`.

**`--fit layers` and `--fit params`** pick the unknown `Q` of the scaling relation
and its feature `q(d)`: `layers` (the paper's relation) is `Q = L`, `q(d) = l^2 d`;
`params` is `Q = P_ne`, `q(d) = l^2 / d`, since block parameters scale as `L d^2`.
`P_ne = P - E(V, d)` is the non-embedding active parameter count, with
`E(V, d) = (1 or 2) V d` the embedding plus unembedding. Then

1. Q-known fit over the references: `T ~ beta * Q * q(d) + alpha`.
2. Q-unknown fit over the references with true `d`: `T ~ gamma_ref * q(d) + zeta`.
3. The same fit with the estimated `d_hat`: `gamma_tgt`; transfer factor `k = gamma_tgt / gamma_ref`.
4. Per-model fit of the target: `T_g ~ gamma_g * q(d_hat_g) + zeta_g`.
5. `Q_hat_g = gamma_g / (k * beta)`; with `params`, `P_hat = P_ne_hat + E(V, d_hat)`.

`--fit all` runs `layers` and `params` on both the true `d` and `d_hat`.

Models excluded by default (`--exclude-ref`, `--exclude-pred`; the `EXCLUDED`
list in `fit_timing.py`) leave the 14 timing models: the dense validation models
above 1B parameters whose every layer uses global multi-head attention (MHA or
GQA). That rules out MoE models, models below 1B parameters, Qwen3.5 (Gated
DeltaNet layers) and OLMo-3 (sliding-window layers), and the development models
are held out. Among the Granite 3.x models, which repeat the same architectures
across releases, only 3.3 is kept so duplicates do not skew the fits.

Mean |relative error| over the 14 timing models on the H100, from `--fit all`:

| `--fit` | Quantity | True `d` | `d_hat` |
|---|---|---|---|
| `layers` | `L` | 7.5% | 36.4% |
| `params` | `P` | 12.4% | 25.2% |

| Flag | Default | Description |
|---|---|---|
| `--fit` | `all` | `layers`, `params` or `all` |
| `--prefill-dir` | `results/timing_results_prefill/` | Directory of `*_timing.json` files |
| `--out-dir` | `results/timing_fits/` | Where the fit JSON and scaling plots go |
| `--keep-longest` | `0` (all) | Longest lengths kept per model |
| `--commonset-dir` | `results/commonset_results/` | Sampling runs `<stem>_sampling.json`; each model's `d_hat` is the estimate of the final step. Every estimated or reference model needs one |
| `--exclude-ref` | every model outside the 14 | Name substrings of models never used as references (they are still estimated) |
| `--exclude-pred` | every model outside the 14 | Name substrings of models not estimated (they may still be references) |

Outputs `results/timing_fits/timing_fit_<fit>.json`, or `timing_fit_<fit>_dhat.json` when
the fit used `d_hat` (`--fit layers` or `--fit params` on its own uses `d_hat`;
`--fit all` writes both). Each model's `estimates.prefill` holds the estimate, its
relative error and the fitted coefficients; `P` is `active_parameters` from this
directory's `model_configs.json`. `results/timing_fits/scaling_law_<fit>_prefill.png` plots the
measured time of every reference model against the fit's feature with the global
fit, using the true `d`.

## Cross-platform calibration

`fit_timing_cross.py` transfers the scaling relation between two platforms. For
a held-out target `g`, the reference platform gives `beta` and `gamma_ref` (true
`d`), the other models on the target platform give `gamma_tgt` (`d_hat`), and
`Q_hat_g = gamma_g / (k * beta)` with `k = gamma_tgt / gamma_ref`. It runs both
directions, for depth and parameter count, with the true `d` and with `d_hat`,
read from `results/commonset_results/` as in `fit_timing.py` (`--commonset-dir`).
Only models measured on both platforms are used.

```bash
python experiments/nightvision_timing/fit_timing_cross.py
python experiments/nightvision_timing/fit_timing_cross.py --platforms a=DIR_A b=DIR_B
```

With both platforms set to the same directory it reproduces the prefill
estimates of `fit_timing.py`. Outputs go to `results/timing_fits_cross/`:
`timing_cross_<ref>_to_<tgt>_<fit>[_dhat].json`.

Mean |relative error| over the 14 timing models:

| Quantity | `d` | H100 reference, H200 target | H200 reference, H100 target |
|---|---|---|---|
| `L` | true | 9.7% | 8.1% |
| `L` | `d_hat` | 35.7% | 38.5% |
| `P` | true | 13.2% | 14.6% |
| `P` | `d_hat` | 28.3% | 24.0% |
