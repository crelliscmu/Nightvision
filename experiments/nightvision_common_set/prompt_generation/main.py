#!/usr/bin/env python3
"""CLI entry point for random search for prompts that minimize model output logprob for the next token."""
import argparse
import logging
from pathlib import Path

from api.vllm_server import VLLMServer, find_port
from search import SearchConfig, run_search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpx HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)

API_KEY = "EMPTY"
DEFAULT_MAX_MODEL_LEN = 128
SCRIPT_DIR = Path(__file__).resolve().parent
# Raw search outputs land here; postprocess.py turns them into ../prompts/<stem>.json.
RESULTS_DIR = SCRIPT_DIR / "results"


def default_output_path(model: str, max_prompt_chars: int) -> Path:
    """results/<N>_char/<model>_<N>tokens.json next to this script.

    run_prompt_search.sh passes an explicit --output using the shared model
    stem naming (e.g. qwen3_0.6b_8tokens.json); this fallback sanitizes the HF
    model name instead so an ad-hoc run still lands in the results tree.
    """
    safe_model = model.split("/")[-1].lower().replace("-", "_")
    return RESULTS_DIR / f"{max_prompt_chars}_char" / f"{safe_model}_{max_prompt_chars}tokens.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find prompts that minimize model output logprob for the next token"
    )

    # Model and server settings
    parser.add_argument(
        "--model",
        type=str,
        default="ibm-granite/granite-3.1-1b-a400m-instruct",
        help="Model name (default: ibm-granite/granite-3.1-1b-a400m-instruct)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for vLLM server (default: auto-select available port)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (default: 0.9)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=f"Maximum model context length (default: {DEFAULT_MAX_MODEL_LEN})"
    )

    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Maximum number of sequences to generate per batch (default: None)"
    )

    # Search parameters
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=5,
        help="Maximum character length for generated prompts (default: 5)"
    )
    parser.add_argument(
        "--variable-length",
        action="store_true",
        help="Enable variable-length prompts (random length from 1 to max-prompt-chars). "
             "Default is fixed-length at exactly max-prompt-chars."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of search iterations (iter 1 = fresh batch; "
             "iters 2..N = 1-char permutations of the best prompt seen so far). "
             "Default: 50"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=10000,
        help="Target number of prompts to find (default: 10000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20000,
        help="Number of prompts to generate per iteration (default: 20000)"
    )

    # Output settings
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path for the raw results "
             "(default: results/<N>_char/<model>_<N>tokens.json next to this script)"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )

    parser.add_argument(
        "--enabled-groups",
        type=str,
        nargs="+",
        default=None,
        help="Only use these character groups (space-separated). If not set, all groups are used."
    )

    return parser.parse_args()


def run_main(args, port: int):
    """Main function to run the search."""
    base_url = f"http://localhost:{port}/v1"

    # Auto-detect models that need thinking disabled
    extra_body = None
    max_gen_tokens = 1
    logprob_token_index = 0
    if "Qwen3" in args.model or "SmolLM3" in args.model:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        logger.info("Qwen3/SmolLM3 model detected — disabling thinking mode")
    #elif "granite-3.1-3b-a800m" in args.model.lower():
    #    max_gen_tokens = 2
    #    logprob_token_index = -1
    #    logger.info("granite model detected — generating %d tokens, "
    #                    "extracting logprobs from last token", max_gen_tokens)

    # Create search configuration
    config = SearchConfig(
        model=args.model,
        max_prompt_chars=args.max_prompt_chars,
        variable_length=args.variable_length,
        max_iterations=args.iterations,
        target_count=args.target,
        batch_size=args.batch_size,
        output_file=args.output,
        base_url=base_url,
        quiet=args.no_progress,
        extra_body=extra_body,
        max_gen_tokens=max_gen_tokens,
        logprob_token_index=logprob_token_index,
        enabled_groups=args.enabled_groups,
    )

    # Run the search
    results = run_search(config)

    logger.info(f"Search complete. Saved {len(results)} prompts to {args.output}")

    return results


def main():
    args = parse_args()
    if args.output is None:
        args.output = str(default_output_path(args.model, args.max_prompt_chars))

    port = args.port if args.port is not None else find_port()

    # Start local vLLM server
    server = VLLMServer(
        model=args.model,
        port=port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )

    with server:
        logger.info(f"vLLM server running at http://localhost:{server.port}")

        # Run search
        run_main(args, server.port)


if __name__ == "__main__":
    main()
