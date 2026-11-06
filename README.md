# NightVision

NightVision infers a language model's architecture from what a restrictive API still exposes: the log-probability of each sampled token and the end-to-end response time. This repository holds the code and results for both main experiments in the paper *Black-Box Inference of LLM Architectural Properties with Restrictive API Access*:

| Experiment | Estimates | Signal | Mean relative error |
|---|---|---|---|
| Common-set prompting ([`nightvision_common_set`](experiments/nightvision_common_set/README.md)) | Hidden dimension `d` | Sampled next tokens and their log-probabilities | 20.9% over 31 models, with 12 exact recoveries |
| Timing ([`nightvision_timing`](experiments/nightvision_timing/README.md)) | Depth `L` and parameter count `P` | Prefill latency over a sweep of prompt lengths | `L` 36.4% and `P` 25.2% over 14 dense models, using the estimated `d` |

**Common-set prompting.** A transformer's final layer maps a `d`-dimensional vector to a logit for every token in the vocabulary. Stack the next-token log-probabilities from many prompts into one matrix and its rank is bounded by roughly `d`, even though the vocabulary is far larger. The experiment builds such matrices from sampled tokens and reads `d` off their eigenvalue spectrum. To keep runs cheap and repeatable, it runs each model locally once to record its full next-token logits, then simulates token sampling from them.

**Timing.** On long prompts, prefill time grows roughly linearly in `L * l^2 * d`, where `l` is the prompt length, with a slope shared across models. The experiment serves each model with vLLM, times it over the sweep, and fits this relation against the other models as references with known architectures. Each model's own fitted slope then gives its depth, and a parallel fit gives its parameter count. These fits take `d` as an input, which is where the two experiments connect: the timing estimates above use the `d` estimated by common-set prompting. The fits can also run with the true `d` (see [Depth and parameter count](#depth-and-parameter-count)).

Both experiments run on a shared **model-serving toolkit** (`api/`, `examples/`). It serves HuggingFace models through [vLLM](https://github.com/vllm-project/vllm), with an OpenAI-compatible API and token-level logprob output. The vLLM server runs inside a [Singularity](https://sylabs.io/singularity/) container, so the environment is portable and reproducible.

## Project structure

```
NightVision/
├── api/
│   ├── __init__.py
│   ├── vllm_server.py          # vLLM server lifecycle management (Singularity-based)
│   └── chat_completions_client.py  # OpenAI-compatible client with batch support
├── examples/
│   ├── single_token.py         # CLI for single-token logprob analysis
│   └── multi_token.py          # CLI for multi-token generation with logprobs
├── experiments/
│   ├── nightvision_common_set/ # hidden dimension: prompt search -> prompt sets -> logits -> sampling
│   │                           # (see experiments/nightvision_common_set/README.md)
│   ├── nightvision_timing/     # depth / parameter count: latency sweep -> scaling-relation fit
│   │                           # (see experiments/nightvision_timing/README.md)
│   └── plotting/
│       └── generate_plots.py   # paper figures from the files in results/
├── results/
│   ├── commonset_results/      # hidden-dimension sampling trajectories for the 31 validation models
│   ├── timing_results_prefill/ # prefill latency sweeps of the 14 timing models on the H100
│   ├── timing_results_prefill_h200/  # the same sweeps on the H200
│   ├── qwen_3.5_timing/        # prefill latency sweeps of the Qwen3.5 models (global-fit figure only)
│   ├── timing_fits/            # fit_timing.py outputs
│   ├── timing_fits_cross/      # fit_timing_cross.py outputs
│   ├── timing_fits_each_budget/  # fit_timing_budgets.py outputs
│   ├── avg_prompt_tokens.json  # average prompt tokens per model (for the cost axes)
│   └── figs/                   # PDF figures written by generate_plots.py
├── model_configs.json          # architecture, hidden/layer/head/FFN sizes, context, vocab, parameter count per model
├── download_models.sh          # downloads every model in model_configs.json to the HF cache
├── pyproject.toml              # project metadata and dependencies
├── uv.lock                     # locked dependency versions
├── install.sh                  # environment setup script
├── LICENSE
└── README.md
```

## Overview

- **`api/vllm_server.py`** — Manages a vLLM server running inside a Singularity container (start, health-check, stop). It can be used as a context manager and cleans up orphaned server processes left on its port.
- **`api/chat_completions_client.py`** — Extended OpenAI-compatible client with single and batch chat completion support, including logprob extraction and full API parameter pass-through.
- **`examples/single_token.py`** — CLI for single-token inference: generates one token and displays top-10 logprobs.
- **`examples/multi_token.py`** — CLI for multi-token inference: generates a sequence and displays per-position top-k logprobs.
- **`experiments/nightvision_common_set/`** — the hidden-dimension experiment, in three stages:
  - `prompt_generation` searches for prompts whose most likely next token has low probability, builds one shared prompt set per model, and records token counts and probability statistics for each set.
  - `collect_logits` runs each model once over its prompt set and caches the next-token logits.
  - `sampling` estimates each model's hidden dimension from the cached logits, with the sample budget spread evenly over all prompts.
- **`experiments/nightvision_timing/`** — the timing experiment, in two stages:
  - `collect_prefill.py` records per-request latencies over a sequence-length sweep for one model; `collect_all_prefill.sh` runs it for all 14 timing models.
  - `fit_timing.py` fits the prefill scaling relations leave-one-out and estimates depth or parameter count. `fit_timing_cross.py` does the same across two platforms (H100 and H200).
  - The fits take each model's estimated `d` from the final sampling step of its `results/commonset_results/<stem>_sampling.json`.
- **`results/commonset_results/`** — `<stem>_sampling.json` for each of the 31 validation models, holding the run config and the per-step estimates. Of the 39 models in `model_configs.json`, the other 8 have no file here.
- **`model_configs.json`** — one entry per model, keyed by HuggingFace repository id: `architecture` (`Dense` or `MoE`), `model_type`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `max_position_embeddings`, `vocab_size`, `tie_word_embeddings`, `total_parameters` (parameters stored in the safetensors files, so tied embeddings count once) and `active_parameters` (which differs from `total_parameters` only for MoE models). The sampling stage uses `hidden_size` as the true value when it reports estimation error. The timing fits use `num_hidden_layers` from each result file and `active_parameters` from `experiments/nightvision_timing/model_configs.json`, which covers the 14 timing models.

## Compute environment

The hidden-dimension experiment and the H100 timing runs used the Pittsburgh Supercomputing Center (PSC) [Bridges-2](https://www.psc.edu/resources/bridges-2/) system. Every stage (prompt search, logit collection, sampling, timing collection) ran on 1x NVIDIA H100-SXM5-80GB GPU. The 14 timing models were also measured on an NVIDIA H200 on the National Center for Supercomputing Applications (NCSA) [Delta](https://delta.ncsa.illinois.edu/) system to test cross-platform calibration.

## Installation

> **Prerequisites:**
> - [Singularity](https://sylabs.io/singularity/) or [Apptainer](https://apptainer.org/), available as the `singularity` command. The vLLM server runs from the official `vllm/vllm-openai:v0.17.0` Docker image, converted to a Singularity `.sif` file.
> - CUDA 12.6.1
> - GCC
> - Python 3.12
>
> On a module-based cluster, load these before running the script (e.g. `module load gcc cuda/12.6.1`).

The included `install.sh` script handles the full setup:

```bash
bash install.sh
```

This will:

1. Install [uv](https://github.com/astral-sh/uv) if it is not already on `PATH`
2. Pull the official vLLM container image (`docker://vllm/vllm-openai:v0.17.0`) to `$PROJECT/containers/vllm.sif`, or `./containers/vllm.sif` if `$PROJECT` is unset (skipped if already present)
3. Create a Python 3.12 virtual environment using [uv](https://github.com/astral-sh/uv)
4. Install PyTorch, depending on `$HOSTNAME`: the `cu126` build on `login-gpu` (which also appends `module load gcc cuda/12.6.1` to `.venv/bin/activate`), or the `cu132` build on `delta.ncsa.illinois.edu`. On any other host this step is skipped, and PyTorch comes from the default index as a dependency of `accelerate`, so install a build that matches your CUDA version if you need a specific one
5. Install the project in editable mode, pulling runtime dependencies from `pyproject.toml` (openai, httpx, requests, tqdm, numpy, wandb, huggingface-hub, transformers 5.x, safetensors, matplotlib, accelerate)

The script exits with an error if the `singularity` command is not found.

After installation, activate the environment with:

```bash
source .venv/bin/activate
```

### Environment variables

| Variable | Used for |
|---|---|
| `PROJECT` | Parent directory of `containers/vllm.sif`. If unset, the container is expected at `./containers/vllm.sif`. |
| `HF_HUB_CACHE` (or `HF_HOME`) | HuggingFace hub cache. Models are downloaded here, and the same directory is bind-mounted into the vLLM container. It must point to the same location for downloading and for running. |
| `HF_TOKEN` | Needed to download gated models such as `meta-llama/*`. Running `hf auth login` once also works. |

The vLLM container also bind-mounts `~/.cache/vllm` and `/tmp`. Create `~/.cache/vllm` before the first run if it does not exist.

### Downloading the models

`download_models.sh` runs `hf download` for every model listed in `model_configs.json` (39 models, the largest about 30B parameters). The downloads take a substantial amount of disk space. Run it from the repository root with the environment active:

```bash
source .venv/bin/activate
bash download_models.sh
```

## Hardware assumptions

The experiment scripts assume a single GPU comparable to an NVIDIA H100 with 80 GB of memory, the hardware every stage was run on (see [Compute environment](#compute-environment)). Every stage runs on one GPU; no multi-GPU or multi-node setup is required.

On a smaller GPU, scale down the memory-heavy settings before running:

- **Prompt search** (`prompt_generation/main.py`): lower `--max-num-seqs` (concurrent vLLM sequences) and `--batch-size` (candidates per iteration), and if needed `--gpu-memory-utilization`. Larger models may need `--tensor-parallel-size` > 1 across several smaller GPUs.
- **Logit collection** (`collect_logits/collect.py`): lower `--batch-size` (prompts per forward pass).
- **Sampling**: lower `--samples-per-iter` (draws per prompt per sampling call). `--max-prompts` also scales GPU memory use, but changing it changes the experiment.
- **Timing collection** (`nightvision_timing/collect_prefill.py`): lower `--gpu-mem` to leave memory free. `--max-len` (100,000 tokens by default) bounds the longest prompt, but changing it changes the sweep.

The wrapper scripts pass any extra command-line arguments through to the Python scripts, so these settings can be changed without editing the wrappers.

## Usage

### Running the experiments

Run these from the repository root with the environment active, on a GPU node, after downloading the models. Each experiment's README covers the options and output formats: [`nightvision_common_set`](experiments/nightvision_common_set/README.md) and [`nightvision_timing`](experiments/nightvision_timing/README.md).

#### Hidden dimension

```bash
cd experiments/nightvision_common_set
bash prompt_generation/run_prompt_search.sh 5     # search prompts of each length
bash prompt_generation/run_prompt_search.sh 8
python prompt_generation/postprocess.py           # build prompts/<stem>.json
bash collect_logits/collect_all.sh                # cache next-token logits
bash sampling/run_sampling.sh                     # estimate hidden dimensions
```

The prompt sets in `prompts/` are checked in, so you can start at `collect_all.sh`.

#### Depth and parameter count

```bash
bash experiments/nightvision_timing/collect_all_prefill.sh   # latency sweep for the 14 timing models
python experiments/nightvision_timing/fit_timing.py --fit all
python experiments/nightvision_timing/fit_timing_cross.py    # H100 <-> H200 calibration
```

The fits only need the JSON files in `results/timing_results_prefill/` and `results/timing_results_prefill_h200/`, so they run on CPU without collecting again. They also read the hidden-dimension estimates from `results/commonset_results/`, which is checked in, so you do not have to run the hidden-dimension experiment first.

`--fit all` runs depth and parameter-count fits with the true `d` and the estimated `d`. The [timing README](experiments/nightvision_timing/README.md) has the errors for each.

### Single-token logprob analysis

Generate one token and inspect the probability distribution:

```bash
python examples/single_token.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --prompt "The capital of France is" \
  --tp 1 \
  --max-model-len 4096
```

### Multi-token generation with logprobs

Generate a full sequence and see per-position logprobs:

```bash
python examples/multi_token.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --prompt "Explain quantum computing in one sentence." \
  --max-tokens 50 \
  --top-k-logprobs 10 \
  --tp 1 \
  --max-model-len 4096
```

### CLI options

Both scripts share common flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace model name or path |
| `--prompt` | `"Who are you?"` | User prompt to evaluate |
| `--system-message` | `None` | Optional system message |
| `--tp` | `1` | Tensor parallel size (number of GPUs) |
| `--gpu-mem` | `0.9` | Fraction of GPU memory to use |
| `--max-model-len` | `4096` | Maximum model context length |
| `--port` | *auto* | Server port (randomly selected if omitted) |

`examples/multi_token.py` adds:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-tokens` | `50` | Number of tokens to generate |
| `--temperature` | `0.0` | Sampling temperature |
| `--top-k-logprobs` | `10` | Top logprobs per position (max 20) |

## Programmatic usage

```python
from api.vllm_server import VLLMServer, find_port
from api.chat_completions_client import ChatCompletionsClient

server = VLLMServer(
    model="meta-llama/Llama-3.1-8B-Instruct",
    port=find_port(),
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    # container_image="path/to/custom/vllm.sif",  # default: $PROJECT/containers/vllm.sif
    # hf_cache_dir="path/to/hf/cache",             # default: $HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub
    # max_model_len=4096,                           # default: vLLM default
    # max_num_seqs=256,                             # default: vLLM default
    # flash_attn_version=2,                         # default: vLLM default
)

with server:
    client = ChatCompletionsClient(
        api_key="EMPTY",
        base_url=server.base_url,
        default_model="meta-llama/Llama-3.1-8B-Instruct",
    )

    # Single request with logprobs
    result = client.evaluate_single_chat(
        messages="What is 2+2?",
        max_tokens=1,
        logprobs=True,
        top_logprobs=10,
    )

    # Batch requests (threaded)
    results = client.evaluate_batch(
        batch_messages=["Prompt 1", "Prompt 2", "Prompt 3"],
        max_tokens=50,
    )
```

### VLLMServer parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | *(required)* | HuggingFace model name or path |
| `host` | `"0.0.0.0"` | Server host address |
| `port` | `8000` | Server port |
| `tensor_parallel_size` | `1` | Number of GPUs for tensor parallelism |
| `gpu_memory_utilization` | `0.9` | Fraction of GPU memory to use |
| `max_model_len` | `None` | Maximum context length (vLLM's default if omitted) |
| `max_num_seqs` | `None` | Maximum sequences per batch (vLLM's default if omitted). If both this and `max_model_len` are set, `--max-num-batched-tokens` is set to their product |
| `flash_attn_version` | `None` | Flash attention version (`2` or `3`; vLLM's default if omitted) |
| `container_image` | `$PROJECT/containers/vllm.sif` | Path to Singularity `.sif` image |
| `hf_cache_dir` | `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub` | HuggingFace hub cache directory (bind-mounted into container) |
| `log_dir` | `"logs/vllm"` | Directory for server logs |
| `extra_args` | `None` | Additional raw vLLM CLI arguments, e.g. `["--no-enable-prefix-caching"]` |
| `extra_env` | `None` | Additional environment variables exported inside the container |
| `hf_overrides` | `None` | Values to override in the served HF config (vLLM `--hf-overrides`); only top-level keys are replaced |
| `extend_context` | `False` | If `max_model_len` exceeds the model's native context, add the overrides that let vLLM serve it anyway |

## Models and licenses

The paper evaluates 37 open-weight models from the HuggingFace Hub: 31 validation models and 6 development models used only to design the method. `model_configs.json` and `download_models.sh` also cover two smaller SmolLM2 models: SmolLM2-135M is used for the empirical comparison with the sample-complexity bound (paper Appendix F.3), and SmolLM2-360M is not used in any reported result. We use each model under the license on its model page and do not redistribute any weights; `download_models.sh` fetches them from the Hub. The Llama models are gated, so you must accept Meta's license on their model pages before downloading them. The licenses below are as listed on each model page on 2026-11-06.

| Model | Role | License |
|---|---|---|
| [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Validation | [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE) |
| [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | Development | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`Qwen/Qwen3-30B-A3B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) | Development | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`ibm-granite/granite-3.1-1b-a400m-instruct`](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct) | Development | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`ibm-granite/granite-3.0-1b-a400m-instruct`](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`ibm-granite/granite-3.0-3b-a800m-instruct`](https://huggingface.co/ibm-granite/granite-3.0-3b-a800m-instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`ibm-granite/granite-3.1-3b-a800m-instruct`](https://huggingface.co/ibm-granite/granite-3.1-3b-a800m-instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`ibm-granite/granite-3.0-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.0-2b-instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`ibm-granite/granite-3.1-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct) | Development | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`ibm-granite/granite-3.2-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.2-2b-instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`ibm-granite/granite-3.3-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`ibm-granite/granite-3.0-8b-instruct`](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`ibm-granite/granite-3.1-8b-instruct`](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`ibm-granite/granite-3.2-8b-instruct`](https://huggingface.co/ibm-granite/granite-3.2-8b-instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`ibm-granite/granite-3.3-8b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`ibm-granite/granite-4.0-350m`](https://huggingface.co/ibm-granite/granite-4.0-350m) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`ibm-granite/granite-4.0-micro`](https://huggingface.co/ibm-granite/granite-4.0-micro) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | Validation | [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/LICENSE.txt) |
| [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | Validation | [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE.txt) |
| [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Validation | [Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE) |
| [`allenai/OLMo-2-0425-1B-Instruct`](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`allenai/OLMo-2-1124-7B-Instruct`](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`allenai/OLMoE-1B-7B-0924-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`allenai/Olmo-3-7B-Instruct`](https://huggingface.co/allenai/Olmo-3-7B-Instruct) | Validation | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | Development | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`HuggingFaceTB/SmolLM3-3B`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | Development | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Theory comparison only | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | Not used in reported results | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |

## Acknowledgements

This work used Bridges-2 [1] at Pittsburgh Supercomputing Center and Delta at the National Center for Supercomputing Applications through allocation [CIS251219] from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

This research used the Delta advanced computing and data resource which is supported by the National Science Foundation (award OAC 2005572) and the State of Illinois. Delta is a joint effort of the University of Illinois Urbana-Champaign and its National Center for Supercomputing Applications.

[1] Brown, S. T., Buitrago, P., Hanna, E., Sanielevici, S., Scibek, R., & Nystrom, N. A. (2021). Bridges-2: A Platform for Rapidly-Evolving and Data Intensive Research. In *Practice and Experience in Advanced Research Computing* (PEARC '21), pp. 1–4. [doi:10.1145/3437359.3465593](https://doi.org/10.1145/3437359.3465593)
