# Prompt Generation

Stage 1 of `nightvision_common_set`. This directory contains two things:

- A greedy hill-climbing search for short prompts (5 or 8 characters) whose next-token distribution is as flat as possible. The search minimizes the log-probability of the model's single most likely next token, called the *argmax logprob* below. A low argmax logprob means the probability mass is spread over many tokens, so sampling from the prompt reveals many distinct tokens. The later stages rely on this.
- A post-processing step that merges the search results into one prompt set per model in `../prompts/`, which every later stage reads. It also describes each set: the average token count after the chat template is applied, and probability percentiles.

## How the search works

1. A vLLM server is started locally for the target model (via `api.vllm_server`).
2. **Seed iterations.** The *seed* is the current best prompt. In the first iteration, a fresh batch of `batch_size` candidates is generated, split equally across the enabled character groups plus a synthetic `mixed` group that draws each character from a randomly chosen concrete group (character groups are described below). Later seed iterations are half *1-character permutations* of the best prompt so far (the prompt with one character replaced) and half fresh draws. The seed phase runs for up to 40 iterations (`SearchConfig.seed_iterations`, not exposed on the CLI) and ends early as soon as any prompt's probability drops below 2% (`SearchConfig.seed_early_exit_probability`).
3. Each candidate is sent to the model as a single user message and evaluated with `max_tokens=1`, `temperature=0`, `logprobs=True`. The argmax logprob of the next token is recorded.
4. **Permutation iterations.** `--iterations - 1` rounds follow, in which `batch_size` 1-character permutations of the current seed are evaluated (split equally across groups) and the seed is replaced whenever a lower-logprob prompt appears. A seed whose permutation space is exhausted (the deduplicated batch stays under 50% of `batch_size` after 5 resampling retries) is swapped for a random prompt from the lowest-5% pool of everything evaluated so far.
5. **Final selection.** Every evaluated prompt is sorted by logprob ascending and the bottom `target_count` is saved to JSON.

### Deduplication and resampling

Each iteration requests 1.5× `batch_size` candidates to absorb dedup losses. If the unique batch falls below 50% of `batch_size`, excess candidates from earlier iterations are drained first and then more are sampled (up to 5 retries).

### Character groups

Prompts are built from a wide set of Unicode character groups (math, emoji-adjacent, CJK, Greek, Arabic, Hebrew, braille, ancient scripts, control characters, homoglyphs, invisible/zero-width, RTL/bidi attack sequences, etc.) — see `CHAR_GROUPS` in `search.py` for the full list. By default all groups are enabled; pass `--enabled-groups` to restrict to a subset.

## Files

| File | Description |
|---|---|
| `main.py` | CLI entry point — starts the vLLM server and runs the search |
| `search.py` | Core search logic: character groups, prompt generation, evaluation, persistence |
| `postprocess.py` | Merge raw results per model, dedup (string + token ids), select the lowest-logprob N, write `../prompts/<stem>.json`; then count average prompt tokens and tabulate probability percentiles |
| `run_prompt_search.sh` | Bash wrapper that runs `main.py` with the per-model options used to build the prompt sets in `../prompts/` |
| `results/<N>_char/` | Search outputs for `N`-character prompts, one `<stem>_<N>tokens.json` per model (`<N>` is the prompt length in characters). Each 8-character file combines four separate search runs for the model and uses the merged format (see `--merged-dir`), without per-run source information. Not tracked in git |
| `results/postprocess_report.json` | Per-run report of the last `postprocess.py` run (inputs, per-stem counts, outputs) |
| `results/prompt_set_summary.json` | Per-stem average prompt tokens and logprob/probability percentiles, accumulated across `postprocess.py` runs |
| `results/avg_prompt_tokens.json`, `results/probability_summary.txt` | The same numbers regenerated as a `{"models": [...]}` list and as a text table |
| `../prompts/` | Prompt sets read by `collect_logits` (and, through the collected logits, by the sampling stages) |

## Usage

### Running the search

```bash
python main.py [OPTIONS]
```

#### Model & server options

| Flag | Default | Description |
|---|---|---|
| `--model` | `ibm-granite/granite-3.1-1b-a400m-instruct` | HuggingFace model name |
| `--port` | auto | Port for the vLLM server |
| `--tensor-parallel-size` | `1` | Number of GPUs for tensor parallelism |
| `--gpu-memory-utilization` | `0.9` | Fraction of GPU memory to allocate |
| `--max-model-len` | `128` | Maximum model context length |
| `--max-num-seqs` | None | Maximum number of sequences per batch |

#### Search parameters

| Flag | Default | Description |
|---|---|---|
| `--max-prompt-chars` | `5` | Character length for generated prompts |
| `--variable-length` | off | Random length from 1 to `max-prompt-chars` instead of fixed length |
| `--iterations` | `50` | Total iterations. The seed phase (up to 40 internal iterations, see above) counts as one, so `--iterations - 1` permutation iterations follow it |
| `--target` | `10000` | Number of prompts to keep in the final output |
| `--batch-size` | `20000` | Number of candidate prompts evaluated per iteration |
| `--enabled-groups` | all | Space-separated subset of character group names to use |

#### Output

| Flag | Default | Description |
|---|---|---|
| `--output` | `results/<N>_char/<name>_<N>tokens.json` (next to `main.py`) | Output JSON path for the raw results. `<name>` is the last part of the HuggingFace name, lowercased with `-` replaced by `_` (e.g. `qwen3_8b`). This matches the model stem for some models but not all, so `run_prompt_search.sh` always passes an explicit path |
| `--no-progress` | off | Disable the progress bar |

#### Model auto-detection

If the model name contains `Qwen3` or `SmolLM3`, the model's built-in "thinking" (reasoning) mode is disabled through `chat_template_kwargs={"enable_thinking": False}`. This test also matches Qwen3.5 and the Qwen3-30B variants.

### Building and describing the prompt sets

For every model stem it finds under the input directories, `postprocess.py`:

1. merges the result files (raw `<stem>_<N>tokens.json` searches or merged `<stem>.json` files) and drops duplicate prompt strings (lowest logprob wins);
2. drops prompts whose token-id sequence duplicates a lower-logprob prompt;
3. keeps the lowest-logprob `--top-n` and writes `../prompts/<stem>.json` as `{"prompts": [...]}` (an existing file is kept unless `--overwrite` is given);
4. formats the first `--count-prompts` prompts of the file on disk with the model's chat template (the model-specific wrapper that turns a user message into the token sequence the model was trained on), exactly as `collect_logits/collect.py` does, and averages their token counts;
5. records the logprob/probability percentiles of the selection.

Steps 4–5 accumulate per stem in `results/prompt_set_summary.json`, from which `results/avg_prompt_tokens.json` and `results/probability_summary.txt` are regenerated after every run (a `--models` run updates only its stems).

```bash
python postprocess.py                                   # results/5_char + results/8_char -> ../prompts (top 80000 each)
python postprocess.py -d results/8_char                 # only one result folder
python postprocess.py --models qwen3_0.6b smol2_135m    # only these stems
python postprocess.py --overwrite --no-token-dedup      # replace existing files; skip tokenizer dedup
python postprocess.py --no-token-count                  # skip the chat-template token count
python postprocess.py --merged-dir results/merged       # also keep full entries + summary stats
python postprocess.py --summarize results/8_char        # only tabulate a result folder's probability stats
```

The default settings are the ones used to produce the prompt sets in `../prompts/` and the token counts in `results/avg_prompt_tokens.json`.

| Flag | Default | Description |
|---|---|---|
| `-d, --dirs` | `results/5_char results/8_char` | Result folders to read (top-level `*.json` only) |
| `-o, --prompts-dir` | `../prompts` | Where `<stem>.json` prompt files are written |
| `-n, --top-n` | `80000` | Prompts kept per model (`0` = all) |
| `--models` | all | Restrict to these stems |
| `--no-token-dedup` | off | Skip the tokenizer-based dedup (string dedup always runs) |
| `--no-token-count` | off | Skip the average prompt token count |
| `--count-prompts` | `20000` | Prompts (from the top of the set) averaged for the token count; `0` = all |
| `--count-chunk` | `512` | Prompts rendered/tokenized per call when counting |
| `--merged-dir` | none | Also write full-format JSONs (config, summary, results incl. `num_tokens`/`token_ids`) |
| `--overwrite` | off | Replace prompt files that already exist (counts and stats are refreshed either way) |
| `--hub` | env | HF hub cache for the tokenizers. Sets `HF_HUB_CACHE` only if it is not already set |
| `--trust-remote-code` | off | Pass `trust_remote_code=True` when loading tokenizers |
| `--report` | `results/postprocess_report.json` | Per-run processing report |
| `--summary-json` | `results/prompt_set_summary.json` | Accumulated per-stem counts and stats |
| `--avg-tokens-json` | `results/avg_prompt_tokens.json` | `{"models": [{"model", "architecture", "avg_prompt_tokens"}]}` regenerated from the summary |
| `--probability-txt` | `results/probability_summary.txt` | Probability table regenerated from the summary |
| `--summarize DIR` | none | Only tabulate `summary.probability` of every JSON in `DIR` to `DIR/<name>_probability_summary.txt` (`--summarize-output` to change) |

The model stem is the result filename without its `_<N>tokens` suffix (`qwen3_4b_8tokens.json` → `qwen3_4b`). It must match the stem → HuggingFace mapping used by `collect_logits/collect_all.sh` and `sampling/run_sampling.py`. The architecture column comes from `../../../model_configs.json`.

## Examples

Search for 8-character prompts on Granite 3.1-1B and keep the lowest-logprob 20000:

```bash
python main.py \
  --model ibm-granite/granite-3.1-1b-a400m-instruct \
  --max-prompt-chars 8 \
  --target 20000 \
  --output results/8_char/granite_3.1_1b_a400m_8tokens.json
```

Restrict the search to a subset of character groups:

```bash
python main.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --enabled-groups math emoji_adjacent invisible bidi_attack \
  --target 5000 \
  --output results/8_char/smol2_135m_8tokens.json
```

## Running every model

```bash
bash experiments/nightvision_common_set/prompt_generation/run_prompt_search.sh <5|8> [stem ...] [-- extra main.py args...]
```

`run_prompt_search.sh` runs `main.py` with the options used to build the prompt sets in `../prompts/`, for one prompt length (5 or 8 characters) and the listed model stems, one model at a time; with no stems it runs all 39 models. It locates the repository root from its own location, so it can be started from any directory. Raw results are written to `results/<N>_char/<stem>_<N>tokens.json`, and models whose result file already exists are skipped, so an interrupted run can be restarted. A model that fails does not stop the loop; failures are listed at the end and the script exits non-zero.

```bash
bash experiments/nightvision_common_set/prompt_generation/run_prompt_search.sh 8                       # all models, 8 characters
bash experiments/nightvision_common_set/prompt_generation/run_prompt_search.sh 5 qwen3_8b smol2_135m   # two models, 5 characters
bash experiments/nightvision_common_set/prompt_generation/run_prompt_search.sh 8 qwen3_8b -- --max-num-seqs 256 --batch-size 5000
```

Every search runs on one GPU with `--tensor-parallel-size 1`, `--target 20000` and `--iterations 40`. The remaining options are set per model in a table at the top of the script: `--max-model-len` (64 to 512), `--max-num-seqs` (512 to 2048), `--batch-size` (20000, or 10000 for two models at 8 characters) and `--gpu-memory-utilization` (0.9, or 0.75 for `olmo_3_7b`). These values were chosen for an 80 GB H100. On a smaller GPU, lower `--max-num-seqs` and `--batch-size` by passing them after `--`; later flags override the table.

| Variable | Default | Description |
|---|---|---|
| `PROJECT_DIR` | derived from the script location | Repository root |
| `RESULTS_DIR` | `prompt_generation/results` | Parent of the `<N>_char/` output directories |
| `PYTHON` | `$PROJECT_DIR/.venv/bin/python` | Interpreter |

## Output format

The search results JSON contains:

- `config` — search parameters (`model`, `max_prompt_chars`, `variable_length`, `iterations`, `target`)
- `summary` — count and logprob/probability percentile stats over the saved prompts
- `results` — list of objects with `prompt`, `token`, `logprob`, `probability`, and `group`

The `postprocess.py` prompt file (`../prompts/<stem>.json`) contains:

- `prompts` — ordered list of prompt strings (lowest logprob first)
