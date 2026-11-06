# Collect Logits

Stage 2 of `nightvision_common_set`. For every prompt set in
`../prompts/<stem>.json`, this stage runs the matching HuggingFace model once
over each prompt and saves the model's next-token logits. The saved tensors
are the only input to `../sampling/`, which never loads a model itself.

## Files

| File | Description |
|---|---|
| `collect.py` | Load one prompt set and one model, write `<name>_logits.pt` |
| `collect_all.sh` | Bash loop: `collect.py` for every `../prompts/*.json`, serially on one GPU |
| `results/` | Output directory, one `<stem>_logits.pt` per model (not tracked in git) |

## How it works

`collect.py` formats every prompt as a single user message with the model's
chat template (`add_generation_prompt=True`, `enable_thinking=False`). It
tokenizes each batch with left padding and keeps the logits at the last
position, which are the logits for the model's first response token. Models without a
padding token use their end-of-sequence token for padding. The model is loaded in bfloat16 and
run under `torch.no_grad()`; the logits are accumulated in float32 and cast to
`--save-dtype` at save time. The file is written to a `.tmp` name and renamed,
so a partially written file is never mistaken for a finished one. If no GPU
is available the script runs on CPU, which is impractically slow for real
runs.

The output is a 2-D tensor of shape `(prompts, vocab_size)` where row `i` is
the next-token logit vector for prompt `i` of the input set, in the same order
as the prompt file. Only the first `--num-prompts-cap` prompts (default 20000)
are collected, which is also the default `MAX_PROMPTS` of the sampling stages.

## Usage

```bash
python collect.py <prompts file> <output name> <HF model> [options]
```

A bare prompts filename that does not exist as a path is looked up in
`../prompts/`, so `python collect.py qwen3_8b.json qwen3_8b Qwen/Qwen3-8B`
writes `results/qwen3_8b_logits.pt`. A `.txt` file with one prompt per line is
also accepted.

| Argument / flag | Default | Description |
|---|---|---|
| `prompt_filename` | *(required)* | `.json` (`{"prompts": [...]}`) or `.txt` prompts file, or a bare name in `../prompts/` |
| `output_filename` | *(required)* | Base name of the output; saved as `<output-dir>/<output_filename>_logits.pt` |
| `model_name` | *(required)* | HuggingFace model name |
| `--output-dir` | `results/` | Where the `.pt` is written; a fast local disk helps for large runs |
| `--batch-size` | `200` | Prompts per forward pass (sized for an 80 GB H100; lower it on smaller GPUs) |
| `--num-prompts-cap` | `20000` | Only collect the first N prompts (`0` or negative = all) |
| `--save-dtype` | `bfloat16` | dtype of the saved tensor (`float32`, `bfloat16`, `float16`) |

## Running every prompt set

```bash
bash experiments/nightvision_common_set/collect_logits/collect_all.sh [extra collect.py args...]
```

Run it with `bash` on a machine with one GPU. It locates the repository root
from its own location, so it can be started from any directory. It loops over every `.json` under
`$PROMPTS_DIR` in sorted order, looks the stem up in its `MODELS` table
(stem -> HuggingFace name; keep it in sync with `_STEM_TO_HF` in
`../sampling/run_sampling.py`), and runs `collect.py` for
each. Stems without a mapping are skipped with a warning. Extra arguments are
passed through to every `collect.py` call.

- Files whose logits already exist are skipped, so if the loop is
  interrupted just rerun it and it resumes at the first unfinished model.
- A failing model (for example an OOM) does not abort the loop; it is reported
  at the end and the script exits non-zero if anything failed.
- Prompt sets may live in subdirectories of `$PROMPTS_DIR`
  (`variant/qwen3_8b.json`). Each subdirectory gets its own output
  subdirectory and its name is appended to the filename:
  `$OUTPUT_DIR/variant/qwen3_8b_variant_logits.pt`. Nested subdirectory
  names are joined with `_`.

Environment overrides:

| Variable | Default | Description |
|---|---|---|
| `PROJECT_DIR` | derived from the script location | Repository root |
| `PROMPTS_DIR` | `../prompts` | Directory of prompt sets |
| `OUTPUT_DIR` | `collect_logits/results` | Where the logits land (what the sampling stages read by default) |
| `PYTHON` | `$PROJECT_DIR/.venv/bin/python` | Interpreter |
