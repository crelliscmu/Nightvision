# nightvision_common_set

A pipeline of experiments that estimates each model's hidden dimension from its
next-token output distributions. Every stage of the pipeline uses the same
prompt set for a given model, hence "common set". Each stage reads its input
from a fixed location in this directory and writes its output to another. Any
stage can therefore be run on its own once the stage before it has produced
its output. Each subdirectory has its own README with the full command-line
options and output format.

## Background

A language model's last layer turns a `hidden_size`-dimensional vector into a
logit for every vocabulary token. Stack the next-token log-probabilities of many
prompts into a matrix. Its rank is limited by roughly `hidden_size`, even though
it has one column per vocabulary token. The pipeline exploits this limit:

1. It finds prompts whose next-token distribution is spread out. These are
   prompts where even the most likely next token has low probability, so
   sampling from them yields many distinct tokens.
2. It records each model's full next-token logits for those prompts.
3. It simulates sampling from those distributions. Each prompt only reveals
   the tokens it has actually drawn, as a black-box API returning sampled
   tokens would. The estimator keeps the largest group of prompts and tokens
   in which every prompt has drawn every token (the *common-token block*).
   It then estimates the hidden dimension from the eigenvalue spectrum of that
   block's log-probability matrix.

## Layout

```
nightvision_common_set/
├── prompts/                       # one prompt set per model: <stem>.json = {"prompts": [...]}
│   └── prompt_stats.txt           # probability percentiles of the 39 prompt sets
├── prompt_generation/             # stage 1: search for low-probability prompts
│   ├── main.py, search.py         # vLLM-driven greedy search
│   ├── postprocess.py             # search results -> ../prompts/<stem>.json + token counts + stats
│   ├── run_prompt_search.sh       # runs main.py for every model with the per-model options used
│   └── results/                   # written by the search and postprocess.py (not tracked in git)
│       ├── <N>_char/              # search outputs, <stem>_<N>tokens.json
│       ├── prompt_set_summary.json    # per-stem average prompt tokens + probability stats
│       ├── avg_prompt_tokens.json     # {"models": [{"model", "architecture", "avg_prompt_tokens"}]};
│       │                              # the published copy is ../../results/avg_prompt_tokens.json
│       ├── probability_summary.txt    # the same stats as a table
│       └── postprocess_report.json    # report of the last postprocess.py run
├── collect_logits/                # stage 2: next-token logits for every prompt set
│   ├── collect.py                 # ../prompts/<stem>.json -> results/<stem>_logits.pt
│   ├── collect_all.sh             # runs collect.py for every prompt file
│   └── results/                   # one (prompts, vocab) .pt per model (created on first run)
└── sampling/                      # stage 3: hidden-dimension estimate
    ├── run_sampling.py            # ../collect_logits/results/*.pt -> results/
    ├── run_sampling.sh            # runs the script with the options used for the published results
    └── results/<run>_<dtype>/     # (created on first run)
        ├── <stem>/<stem>_sampling.json, plots/*.png
        └── sampling_summary.json
```

In the stage-3 output directory, `<dtype>` is the precision used to store the
cached log-probabilities (`--logprob-dtype`, `bfloat16` by default). The
wrapper can run several precisions side by side.

## Running the pipeline

All stages run in the repository's virtual environment (run
`source .venv/bin/activate` at the repository root; see the root `README.md`
for installation). They expect the model tokenizers and weights to be in the
HuggingFace hub cache (`HF_HUB_CACHE`; `download_models.sh` at the repository
root fills it). Each stage has a bash wrapper that runs its Python script with
the options used for the published results. The wrappers find the repository root from their own
location, so they can be started from any directory. The commands below are
written relative to this directory. Every stage runs on a single GPU. The
default settings assume an 80 GB H100; the "Hardware assumptions" section of
the root `README.md` lists what to lower on smaller GPUs.

1. **Search for prompts** (`prompt_generation/`). Run one search per model
   and prompt length:
   `bash prompt_generation/run_prompt_search.sh 5` and
   `bash prompt_generation/run_prompt_search.sh 8` search every model in turn
   (or only the stems listed after the length). Each search writes the 20000
   lowest-probability prompts it found to
   `prompt_generation/results/<N>_char/`. The 8-character results shipped
   with the current prompt sets combine four separate search runs per model.
2. **Build the prompt sets** (`prompt_generation/postprocess.py`). Run
   `python prompt_generation/postprocess.py`. For each model, it merges the
   results from `results/5_char` and `results/8_char` and removes duplicates
   (by string and by token ids). It then keeps the 80000 lowest-probability
   prompts and writes them to `prompts/<stem>.json`. An existing prompt file
   is kept unless `--overwrite` is passed. The script also records each
   prompt set's average token count (after applying the chat template) and
   probability percentiles in `prompt_generation/results/`. It loads
   tokenizers only, not model weights, and can run on CPU.
3. **Collect logits** (`collect_logits/`). Run
   `bash collect_logits/collect_all.sh`. It runs `collect.py` on every
   `prompts/<stem>.json` and writes `collect_logits/results/<stem>_logits.pt`.
   That file is a `(prompts, vocab)` bfloat16 tensor of next-token logits for
   the first 20000 prompts of the set. Files that already exist are skipped,
   so an interrupted run can be restarted.
4. **Sampling** (`sampling/`). Run `bash sampling/run_sampling.sh`. It
   estimates each model's hidden dimension from the collected logits without
   loading the model. All prompts (`MAX_PROMPTS`, 20000 by default) are used
   from the first step, and every step gives every prompt the same number of
   draws. Per model, it writes a trajectory JSON and per-step plots. It also
   writes `sampling_summary.json` under `sampling/results/<run>_<dtype>/`.

The stage-3 wrapper accepts environment overrides (`LOGITS`, `PATTERN`,
`OUT_ROOT`, `RUN_NAME`, `LOGPROB_DTYPES`, `SWEEP_PASSES`, `MAX_PROMPTS`,
`PRED_SMOOTH`, ...). It passes any extra command-line arguments through to the
Python script. `sampling/README.md` lists every override.

## Model stems

Prompt files, logits files and result folders are named by a short model
*stem*, such as `qwen3_8b`, `granite_3.1_2b` or `smol2_135m`. There are
currently 39 models. Four lists must be kept in sync when a model is added or
removed:

- the per-model options table at the top of
  `prompt_generation/run_prompt_search.sh` (stem, HuggingFace name and search
  settings)
- `MODELS` in `collect_logits/collect_all.sh` (stem -> HuggingFace name)
- `_STEM_TO_HF` in `sampling/run_sampling.py` (stem -> HuggingFace name)
- `../../model_configs.json`, keyed by HuggingFace name, with the model's
  `architecture` and `hidden_size`. The estimator uses `hidden_size` as the
  true dimension when it reports error. `postprocess.py` uses `architecture`
  in its tables, and `download_models.sh` downloads every model listed there.

`postprocess.py` reads the model name from the search results themselves, so
it needs no stem mapping.
