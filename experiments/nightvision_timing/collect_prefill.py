#!/usr/bin/env python3
"""Collect prefill-dominated end-to-end latencies for one model (timing attack, stage 1).

Serves the model with vLLM, then for every input length in the paper's
sequence-length sweep sends batch-size-one chat requests that generate a
single token and records the client-side latency of each. An l=1 baseline is
measured in the same session. Raw per-request latencies are written to
results/timing_results_prefill/<model>_timing.json; fit_timing.py turns them into depth estimates.
"""

import argparse
import importlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.chat_completions_client import ChatCompletionsClient  # noqa: E402
from api.vllm_server import VLLMServer, find_port  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parents[1] / "results" / "timing_results_prefill"

# Sequence-length sweep (paper, "Sequence-length sweep").
NUM_LENGTHS = 24
MIN_LEN = 1             # shortest prompt in tokens
MAX_LEN = 100000        # longest prompt in tokens
THROUGHPUT_RATIO = 0.6  # rho_thr: empirical attention-to-MLP throughput ratio
POS_MARGIN = 200        # keep l_hi <= p_max - POS_MARGIN
IGNORE_MAX_POS = True   # let max_len exceed p_max by overriding it in the served config
# Timing protocol.
NUM_WARMUP = 2
NUM_TRIALS = 15
# Repeated to build prompts of arbitrary token length.
PASSAGE = (
    "The quick brown fox jumps over the lazy dog while the sun sets slowly "
    "behind the distant mountains, painting the sky in shades of orange and "
    "purple as the evening breeze carries the scent of pine through the valley. "
)


def load_tokenizer(model_name: str):
    """AutoTokenizer, falling back to the class named in the hub's tokenizer_config.json.

    transformers 5.x routes model types registered as a generic backend (e.g.
    granitemoe) to TokenizersBackend regardless of the hub's tokenizer_class.
    That backend only builds from tokenizer.json / tokenizer.model, so repos
    that ship only vocab.json + merges.txt (ibm-granite/granite-3.0-3b-a800m-
    instruct) fail with "Couldn't instantiate the backend tokenizer". The
    declared class (GPT2Tokenizer) loads those files fine.
    """
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except ValueError as e:
        with open(hf_hub_download(model_name, "tokenizer_config.json"), encoding="utf-8") as f:
            cls_name = json.load(f).get("tokenizer_class")
        if not cls_name:
            raise
        print(f"WARNING: AutoTokenizer failed ({str(e).splitlines()[0]}); "
              f"retrying with {cls_name} from tokenizer_config.json")
        return getattr(importlib.import_module("transformers"), cls_name).from_pretrained(model_name)


def load_arch(model_name: str) -> dict:
    """Architecture fields the sweep needs, from the model's HF config.json."""
    with open(hf_hub_download(model_name, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = cfg.get("text_config", cfg)  # composite configs (e.g. Qwen3.5) nest the LM config
    n_heads = cfg["num_attention_heads"]
    return {
        "hidden_size": cfg["hidden_size"],
        "num_hidden_layers": cfg["num_hidden_layers"],
        "num_attention_heads": n_heads,
        "num_key_value_heads": cfg.get("num_key_value_heads", n_heads),
        "intermediate_size": cfg["intermediate_size"],
        "max_position_embeddings": cfg["max_position_embeddings"],
        "vocab_size": cfg["vocab_size"],
    }


def sequence_lengths(arch: dict, min_len: int = MIN_LEN, max_len: int = MAX_LEN,
                     num_lens: int = NUM_LENGTHS,
                     ignore_max_pos: bool = False) -> tuple[list[int], int]:
    """(lengths, l_star): num_lens evenly spaced lengths in [min_len, max_len].

    A model's context window does not limit the sweep: with ignore_max_pos, max_len
    may exceed max_position_embeddings and the served config is overridden to fit
    it. l_star is the projection-to-attention crossover, recorded for reference.
    """
    d = arch["hidden_size"]
    d_kv = arch["num_key_value_heads"] * d // arch["num_attention_heads"]
    l_star = int((d + d_kv + 1.5 * arch["intermediate_size"]) * THROUGHPUT_RATIO)
    l_max_pos = arch["max_position_embeddings"] - POS_MARGIN
    if max_len > l_max_pos and not ignore_max_pos:
        raise ValueError(f"max_len={max_len} exceeds max_position_embeddings - {POS_MARGIN} = {l_max_pos}")
    if not 1 <= min_len <= max_len:
        raise ValueError(f"need 1 <= min_len <= max_len, got [{min_len}, {max_len}]")
    if num_lens < 1:
        raise ValueError(f"num_lens must be >= 1, got {num_lens}")
    if num_lens == 1:
        return [max_len], l_star
    lengths = {min_len + (max_len - min_len) * j // (num_lens - 1) for j in range(num_lens)}
    return sorted(lengths), l_star


def filler_ids(tokenizer, num_tokens: int) -> list[int]:
    """`num_tokens` token ids made by repeating PASSAGE and truncating."""
    passage_ids = tokenizer(PASSAGE, add_special_tokens=False)["input_ids"]
    reps = num_tokens // len(passage_ids) + 1
    return tokenizer(PASSAGE * reps, add_special_tokens=False)["input_ids"][:num_tokens]


def build_prompt(tokenizer, filler: list[int]) -> str:
    """Text of len(filler) tokens that starts with a fresh random nonce.

    The nonce makes every request's prompt differ from its first token, so no
    request can be served from a prefix cache populated by an earlier
    same-length request. It replaces leading filler tokens to keep the length.
    """
    nonce_ids = tokenizer(f"{uuid.uuid4().hex}\n", add_special_tokens=False)["input_ids"]
    return tokenizer.decode((nonce_ids + filler)[:len(filler)])


def time_requests(client, tokenizer, filler: list[int], n: int) -> tuple[list[float], int]:
    """Latencies of n sequential single-token requests and the prompt token count.

    Each request gets its own randomly prefixed prompt, built outside the timed region.
    """
    latencies, prompt_tokens = [], None
    for _ in range(n):
        prompt = build_prompt(tokenizer, filler)
        t0 = time.perf_counter()
        result = client.evaluate_single_chat(
            prompt, max_tokens=1, temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        latencies.append(time.perf_counter() - t0)
        if not result.success:
            raise RuntimeError(f"request failed: {result.error}")
        prompt_tokens = result.usage["prompt_tokens"]
    return latencies, prompt_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_name", help="HuggingFace model name (e.g. Qwen/Qwen3-8B)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Where <model>_timing.json is written (default: {OUTPUT_DIR})")
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS,
                        help=f"Timed requests per length (default: {NUM_TRIALS})")
    parser.add_argument("--min-len", type=int, default=MIN_LEN,
                        help=f"Shortest prompt length in tokens (default: {MIN_LEN})")
    parser.add_argument("--max-len", type=int, default=MAX_LEN,
                        help=f"Longest prompt length in tokens (default: {MAX_LEN})")
    parser.add_argument("--num-lens", type=int, default=NUM_LENGTHS,
                        help=f"Number of evenly spaced lengths (default: {NUM_LENGTHS})")
    parser.add_argument("--ignore-max-pos", action=argparse.BooleanOptionalAction,
                        default=IGNORE_MAX_POS,
                        help="Allow --max-len beyond the model's max_position_embeddings by "
                             "overriding it in the served config (RoPE/ALiBi models only). "
                             "Outputs past the trained context are meaningless, but prefill "
                             "compute, and so latency, does not depend on it. "
                             f"(default: {IGNORE_MAX_POS}; --no-ignore-max-pos to enforce p_max)")
    parser.add_argument("--gpu-mem", type=float, default=0.9, help="vLLM GPU memory utilization")
    return parser.parse_args()


def main():
    args = parse_args()
    arch = load_arch(args.model_name)
    lengths, l_star = sequence_lengths(arch, args.min_len, args.max_len, args.num_lens,
                                       args.ignore_max_pos)
    max_model_len = lengths[-1] + POS_MARGIN
    print(f"{args.model_name}: d={arch['hidden_size']} L={arch['num_hidden_layers']} "
          f"l*={l_star} sweep={lengths[0]}..{lengths[-1]} ({len(lengths)} points)")

    tokenizer = load_tokenizer(args.model_name)
    # Prefill the whole prompt in one scheduler step (max_num_seqs=1 makes the
    # server's --max-num-batched-tokens equal max_model_len) and never serve a
    # repeated prompt from the prefix cache (each prompt also gets a random
    # nonce prefix; see build_prompt).
    # extend_context lets max_model_len exceed the model's max_position_embeddings
    # by patching the served config (RoPE cache size, YaRN factor); see
    # api.vllm_server.context_overrides.
    server = VLLMServer(
        model=args.model_name,
        port=find_port(),
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=max_model_len,
        max_num_seqs=1,
        extra_args=["--no-enable-prefix-caching"],
        extend_context=True,
        # Server logs go to their own directory, one file per model.
        log_dir="logs/vllm_prefill",
    )
    if server.hf_overrides:
        print(f"WARNING: overriding served config: {json.dumps(server.hf_overrides)}")
    record = {
        "model": args.model_name,
        "arch": arch,
        "l_star": l_star,
        "hf_overrides": server.hf_overrides,
        "hostname": os.uname().nodename,
        "num_trials": args.num_trials,
        "measurements": [],  # one entry per length, l=1 baseline first
    }
    with server:
        client = ChatCompletionsClient(api_key="EMPTY", base_url=server.base_url,
                                       default_model=args.model_name, timeout=300.0)
        for length in [1] + lengths:
            filler = filler_ids(tokenizer, length)
            time_requests(client, tokenizer, filler, NUM_WARMUP)
            latencies, prompt_tokens = time_requests(client, tokenizer, filler, args.num_trials)
            record["measurements"].append(
                {"length": length, "prompt_tokens": prompt_tokens, "latencies": latencies})
            print(f"  l={length:6d} (prompt_tokens={prompt_tokens:6d}) "
                  f"mean={sum(latencies) / len(latencies):.4f}s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.model_name.replace('/', '_')}_timing.json"
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    os.replace(tmp, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
