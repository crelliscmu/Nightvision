#!/usr/bin/env python3
"""Collect next-token logits across prompts for a given model.

Reads a prompt set from the unified prompt directory (../prompts/<stem>.json,
written by prompt_generation/postprocess.py), renders every prompt through the
model's chat template as a single user message, and saves the next-token
logits as a 2-D (prompts, vocab) tensor to results/<name>_logits.pt. Those
tensors are the input of sampling/run_sampling.py.
Nothing outside this file is imported.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent


def load_tokenizer(model_name):
    """AutoTokenizer, falling back to the class named in tokenizer_config.json.

    transformers 5 maps some model types (e.g. granitemoe) to a generic fast
    backend that needs tokenizer.json. ibm-granite/granite-3.0-3b-a800m-instruct
    only ships vocab.json/merges.txt, but its tokenizer_config.json names
    GPT2Tokenizer, which rebuilds the BPE model from those files."""
    import transformers
    from transformers.utils import cached_file

    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as auto_err:
        try:
            with open(cached_file(model_name, "tokenizer_config.json"), encoding="utf-8") as f:
                cls = getattr(transformers, json.load(f).get("tokenizer_class") or "", None)
        except Exception:
            cls = None
        if cls is None or cls is AutoTokenizer:
            raise
        print(f"AutoTokenizer failed ({type(auto_err).__name__}); using {cls.__name__}")
        return cls.from_pretrained(model_name)

# Configuration
# Collected logits land here; sampling reads them from this directory by default.
OUTPUT_DIR = SCRIPT_DIR / "results"
# Unified prompt directory shared by every nightvision_common_set experiment.
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"
BATCH_SIZE = 200
NUM_PROMPTS_CAP = 20000
SAVE_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def load_prompts(path: Path) -> list[str]:
    """Read prompts from a .json ({'prompts': [...]}) or .txt (one per line) file."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["prompts"]
    if suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    raise ValueError(f"Unsupported prompts file format: {suffix}")


def collect_logits(model, tokenizer, prompts: list[str], batch_size: int) -> torch.Tensor:
    """Next-token logits for every prompt, one chat-templated batch at a time."""
    model.eval()
    input_device = model.device
    vocab_size = model.config.vocab_size
    print(f"Vocab size: {vocab_size}")
    num_prompts = len(prompts)
    all_logits = torch.zeros((num_prompts, vocab_size))
    num_batches = (num_prompts + batch_size - 1) // batch_size
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}

    with torch.no_grad():
        for batch_start in tqdm(range(0, num_prompts, batch_size), desc="Batches", total=num_batches):
            batch_end = min(batch_start + batch_size, num_prompts)
            batch_prompts = prompts[batch_start:batch_end]

            templated = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    **kwargs
                )
                for p in batch_prompts
            ]
            inputs = tokenizer(
                templated, return_tensors="pt", padding=True, truncation=True
            ).to(input_device)
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            next_token_logits = outputs.logits[:, -1, :].float().cpu()
            all_logits[batch_start:batch_end, :] = next_token_logits
            del outputs, next_token_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return all_logits


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect next-token logits across prompts for a given model."
    )
    parser.add_argument(
        "prompt_filename",
        type=str,
        help="Path to a .json ({'prompts': [...]}) or .txt (one per line) prompts file. "
             f"A bare filename that does not exist is looked up in {PROMPTS_DIR} "
             "(e.g. qwen3_8b.json).",
    )
    parser.add_argument(
        "output_filename",
        type=str,
        help="Base name for the output .pt file (e.g. my_run). "
             "Saved as <output-dir>/<output_filename>_logits.pt.",
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="HuggingFace model name (e.g. Qwen/Qwen3-30B-A3B-Instruct-2507).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory the logits .pt is written to (default: {OUTPUT_DIR}). "
             "Point this at node-local storage for large runs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Prompts per forward pass (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--num-prompts-cap",
        type=int,
        default=NUM_PROMPTS_CAP,
        help=f"Only collect logits for the first N prompts (default: {NUM_PROMPTS_CAP}; "
             "0 or negative = all).",
    )
    parser.add_argument(
        "--save-dtype",
        choices=sorted(SAVE_DTYPES),
        default="bfloat16",
        help="dtype of the saved tensor (default: bfloat16). Collection always happens "
             "in float32; other dtypes are a cast at save time.",
    )
    return parser.parse_args()


def resolve_prompts_file(arg: str) -> Path:
    """Accept either a path to a prompts file or a bare filename in the unified prompt dir."""
    candidate = Path(arg)
    if candidate.exists():
        return candidate
    unified = PROMPTS_DIR / arg
    if unified.exists():
        return unified
    raise FileNotFoundError(f"Prompts file not found: {arg} (also tried {unified})")


def main():
    """Load the prompts and model, run inference, and save the logits tensor."""
    args = parse_args()

    prompts_file = resolve_prompts_file(args.prompt_filename)
    output_dir = args.output_dir
    save_dtype = SAVE_DTYPES[args.save_dtype]
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading prompts from {prompts_file}")
    prompts = load_prompts(prompts_file)
    print(f"Loaded {len(prompts)} prompts")
    if args.num_prompts_cap is not None and args.num_prompts_cap > 0:
        prompts = prompts[: args.num_prompts_cap]
        print(f"Capped to {len(prompts)} prompts")

    print(f"Loading model {args.model_name}")
    tokenizer = load_tokenizer(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16)
    model = model.to(device)
    logits = collect_logits(model, tokenizer, prompts, args.batch_size)

    model.to("cpu")
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if logits.dtype != save_dtype:
        logits = logits.to(save_dtype)

    # Write to a temp name and rename so a partially written file is never
    # mistaken for a finished one.
    print("Saving logits...")
    output_file = output_dir / f"{args.output_filename}_logits.pt"
    tmp_file = output_dir / f"{args.output_filename}_logits.pt.tmp"
    torch.save(logits, tmp_file)
    os.replace(tmp_file, output_file)
    print(f"Saved {output_file} with shape {tuple(logits.shape)} dtype {logits.dtype}")

    print("Done!")


if __name__ == "__main__":
    main()
