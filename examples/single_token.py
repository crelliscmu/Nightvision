"""Run single-token inference via vLLM and print top 10 logprobs."""

import argparse
import logging
import math

from api.vllm_server import VLLMServer, find_port
from api.chat_completions_client import ChatCompletionsClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Single-token logprob inference with vLLM")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name/path")
    parser.add_argument("--prompt", type=str, default="Who are you?", help="User prompt to evaluate")
    parser.add_argument("--system-message", type=str, default=None, help="Optional system message")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-mem", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model context length")
    parser.add_argument("--port", type=int, default=None, help="Server port (auto-selected if omitted)")
    args = parser.parse_args()

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
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=10,
        )

        if not result.success:
            print(f"Error: {result.error}")
            return

        print(f"\nPrompt: {args.prompt}")
        print(f"Model:  {args.model}")
        print("-" * 60)
        print(f"\nPredicted next token: {result.token!r}")
        print(f"Logprob:              {result.logprob:.6f}")
        print(f"Probability:          {math.exp(result.logprob):.6f}")
        print(f"\nTop 10 logprobs:")
        print(f"{'Rank':<6}{'Token':<20}{'Logprob':<14}{'Probability':<12}")
        print("-" * 52)
        for i, entry in enumerate(result.top_logprobs or [], start=1):
            prob = math.exp(entry["logprob"])
            print(f"{i:<6}{entry['token']!r:<20}{entry['logprob']:<14.6f}{prob:<12.6f}")

        if result.usage:
            print(f"\nTokens — prompt: {result.usage['prompt_tokens']}, "
                  f"completion: {result.usage['completion_tokens']}, "
                  f"total: {result.usage['total_tokens']}")


if __name__ == "__main__":
    main()
