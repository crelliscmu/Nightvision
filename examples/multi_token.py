"""Run multi-token inference via vLLM and print per-token logprobs."""

import argparse
import logging
import math

from api.vllm_server import VLLMServer, find_port
from api.chat_completions_client import ChatCompletionsClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Multi-token logprob inference with vLLM")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name/path")
    parser.add_argument("--prompt", type=str, default="Who are you?", help="User prompt to evaluate")
    parser.add_argument("--system-message", type=str, default=None, help="Optional system message")
    parser.add_argument("--max-tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-k-logprobs", type=int, default=10, help="Number of top logprobs per position")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-mem", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model context length")
    parser.add_argument("--port", type=int, default=None, help="Server port (auto-selected if omitted)")
    args = parser.parse_args()

    top_k = min(args.top_k_logprobs, 20)
    port = args.port or find_port()

    server = VLLMServer(
        model=args.model,
        port=port,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
    )

    with server:
        client = ChatCompletionsClient(
            api_key="EMPTY",
            base_url=server.base_url,
            default_model=args.model,
        )

        result = client.evaluate_single_chat(
            messages=args.prompt,
            system_message=args.system_message,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            logprobs=True,
            top_logprobs=top_k,
        )

        if not result.success:
            print(f"Error: {result.error}")
            return

        # Extract per-token logprobs from raw response
        choice = result.raw_response.choices[0]
        token_logprobs = choice.logprobs.content if choice.logprobs else []

        print(f"\nPrompt: {args.prompt}")
        print(f"Model:  {args.model}")
        print(f"Generated {len(token_logprobs)} tokens")
        print(f"\nFull completion: {result.response}")
        print("=" * 70)

        for pos, token_info in enumerate(token_logprobs):
            prob = math.exp(token_info.logprob)
            print(f"\n--- Position {pos} ---")
            print(f"  Chosen token: {token_info.token!r}  "
                  f"logprob: {token_info.logprob:.6f}  "
                  f"prob: {prob:.6f}")

            if token_info.top_logprobs:
                print(f"  Top {len(token_info.top_logprobs)} logprobs:")
                print(f"  {'Rank':<6}{'Token':<20}{'Logprob':<14}{'Probability':<12}")
                print(f"  {'-' * 50}")
                for rank, lp in enumerate(token_info.top_logprobs, start=1):
                    p = math.exp(lp.logprob)
                    marker = " <--" if lp.token == token_info.token else ""
                    print(f"  {rank:<6}{lp.token!r:<20}{lp.logprob:<14.6f}{p:<12.6f}{marker}")

        if result.usage:
            print(f"\nTokens — prompt: {result.usage['prompt_tokens']}, "
                  f"completion: {result.usage['completion_tokens']}, "
                  f"total: {result.usage['total_tokens']}")


if __name__ == "__main__":
    main()
