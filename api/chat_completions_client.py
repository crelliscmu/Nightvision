"""Extended client for OpenAI-compatible chat completions with batch support."""

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from httpx import Timeout
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """Result from a chat completion request."""
    messages: list[dict[str, Any]]
    response: str
    finish_reason: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    usage: Optional[dict[str, int]] = None
    # Logprob fields (populated when logprobs=True)
    token: Optional[str] = None
    logprob: Optional[float] = None
    top_k_prob_sum: Optional[float] = None
    top_logprobs: Optional[list[dict[str, float]]] = None
    # Raw response for advanced use
    raw_response: Optional[ChatCompletion] = None


class ChatCompletionsClient:
    """Extended client for OpenAI-compatible chat completions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        default_max_logprobs: int = 20,
        default_max_num_seqs: int = 4096,
        timeout: float = 30.0,
    ):
        """Initialize chat completions client.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            base_url: Optional base URL for OpenAI-compatible endpoints.
            default_model: Default model to use if not specified per-request.
            default_max_logprobs: Default number of top logprobs to request (max 20).
            default_max_num_seqs: Default number of sequences to generate per batch (default: 4096).
            timeout: HTTP request timeout in seconds (default: 30.0).
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=Timeout(timeout, connect=5.0),
        )
        self.default_model = default_model
        self.default_max_logprobs = min(default_max_logprobs, 20)
        self.default_max_num_seqs = default_max_num_seqs
        
    def evaluate_single_chat(
        self,
        messages: Union[str, list[dict[str, Any]]],
        model: Optional[str] = None,
        # Generation parameters
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: int = 1,
        stop: Optional[Union[str, list[str]]] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        # Logprob parameters
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        top_k_for_sum: Optional[int] = None,
        # Response format
        response_format: Optional[dict[str, str]] = None,
        # Tool/function calling
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, dict[str, Any]]] = None,
        # Convenience shortcuts
        system_message: Optional[str] = None,
        # Pass-through for provider-specific params (e.g. vLLM, Qwen)
        extra_body: Optional[dict[str, Any]] = None,
        # Which generated token to extract logprobs from (0=first, -1=last)
        logprob_token_index: int = 0,
    ) -> ChatResult:
        """Execute a single chat completion with full API parameter support.

        Args:
            messages: Either a string (treated as a user message) or a list of
                      message dicts [{"role": ..., "content": ...}, ...].
            model: Model name. Falls back to default_model.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0-2).
            top_p: Nucleus sampling parameter.
            n: Number of completions to generate.
            stop: Stop sequence(s).
            frequency_penalty: Frequency penalty (-2 to 2).
            presence_penalty: Presence penalty (-2 to 2).
            seed: Random seed for deterministic generation.
            logprobs: Whether to return logprobs.
            top_logprobs: Number of top logprobs per token (max 20).
            top_k_for_sum: If provided, calculate sum of top K token probabilities.
            response_format: Response format specification (e.g. {"type": "json_object"}).
            tools: List of tool definitions for function calling.
            tool_choice: Tool choice strategy ("auto", "none", or specific tool).
            system_message: Convenience param — prepends a system message when
                            messages is a string.
            extra_body: Additional provider-specific body parameters.

        Returns:
            ChatResult with response content and optional logprob data.
        """
        resolved_model = model or self.default_model
        if not resolved_model:
            return ChatResult(
                messages=[], response="", success=False,
                error="No model specified and no default_model set.",
            )

        # Build messages list
        if isinstance(messages, str):
            msg_list: list[dict[str, Any]] = []
            if system_message:
                msg_list.append({"role": "system", "content": system_message})
            msg_list.append({"role": "user", "content": messages})
        else:
            msg_list = list(messages)
            if system_message:
                msg_list = [{"role": "system", "content": system_message}] + msg_list

        try:
            # Build kwargs, only including params that were explicitly provided
            kwargs: dict[str, Any] = {
                "model": resolved_model,
                "messages": msg_list,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
            if n != 1:
                kwargs["n"] = n
            if stop is not None:
                kwargs["stop"] = stop
            if frequency_penalty is not None:
                kwargs["frequency_penalty"] = frequency_penalty
            if presence_penalty is not None:
                kwargs["presence_penalty"] = presence_penalty
            if seed is not None:
                kwargs["seed"] = seed
            if logprobs is not None:
                kwargs["logprobs"] = logprobs
            if top_logprobs is not None:
                kwargs["top_logprobs"] = min(top_logprobs, 20)
            if response_format is not None:
                kwargs["response_format"] = response_format
            if tools is not None:
                kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            if extra_body is not None:
                kwargs["extra_body"] = extra_body

            response = self.client.chat.completions.create(**kwargs)

            if not response.choices:
                return ChatResult(
                    messages=msg_list, response="", success=False,
                    error="No choices in response",
                )

            choice = response.choices[0]
            content = choice.message.content or ""

            # Build usage dict
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                # vLLM (and OpenAI) report prefix-cache hits here; absent elsewhere.
                details = getattr(response.usage, "prompt_tokens_details", None)
                cached = getattr(details, "cached_tokens", None)
                if cached is not None:
                    usage["cached_tokens"] = cached

            # Extract logprob data if available
            token_str = None
            token_logprob = None
            top_k_prob_sum = None
            top_logprobs_list = None

            if choice.logprobs and choice.logprobs.content:
                target_token = choice.logprobs.content[logprob_token_index]
                token_str = target_token.token
                token_logprob = target_token.logprob

                if target_token.top_logprobs:
                    top_logprobs_list = [
                        {"token": lp.token, "logprob": lp.logprob}
                        for lp in target_token.top_logprobs
                    ]
                    if top_k_for_sum is not None:
                        probs = [math.exp(lp.logprob) for lp in target_token.top_logprobs[:top_k_for_sum]]
                        top_k_prob_sum = sum(probs)

            return ChatResult(
                messages=msg_list,
                response=content,
                finish_reason=choice.finish_reason,
                success=True,
                usage=usage,
                token=token_str,
                logprob=token_logprob,
                top_k_prob_sum=top_k_prob_sum,
                top_logprobs=top_logprobs_list,
                raw_response=response,
            )

        except Exception as e:
            logger.exception("Chat completion failed")
            return ChatResult(
                messages=msg_list, response="", success=False, error=str(e),
            )

    def evaluate_batch(
        self,
        batch_messages: list[Union[str, list[dict[str, Any]]]],
        model: Optional[str] = None,
        # Shared generation parameters applied to every request
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: int = 1,
        stop: Optional[Union[str, list[str]]] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        # Logprob parameters
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        top_k_for_sum: Optional[int] = None,
        # Response format
        response_format: Optional[dict[str, str]] = None,
        # Tool/function calling
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, dict[str, Any]]] = None,
        # Convenience
        system_message: Optional[str] = None,
        extra_body: Optional[dict[str, Any]] = None,
        # Which generated token to extract logprobs from (0=first, -1=last)
        logprob_token_index: int = 0,
        # Batch control
        show_progress: bool = True,
    ) -> list[ChatResult]:
        """Evaluate chat completions for a batch of message sets in parallel.

        Each item in batch_messages is either a plain string (user message) or
        a full messages list. All other parameters are shared across the batch.

        Args:
            batch_messages: List of message inputs (strings or message lists).
            model: Model name. Falls back to default_model.
            max_tokens: Maximum tokens to generate per request.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            n: Number of completions per request.
            stop: Stop sequence(s).
            frequency_penalty: Frequency penalty.
            presence_penalty: Presence penalty.
            seed: Random seed.
            logprobs: Whether to return logprobs.
            top_logprobs: Number of top logprobs per token.
            top_k_for_sum: If provided, calculate sum of top K probabilities.
            response_format: Response format specification.
            tools: Tool definitions for function calling.
            tool_choice: Tool choice strategy.
            system_message: System message prepended to each request.
            extra_body: Additional provider-specific body parameters.
            max_workers: Maximum parallel threads.
            show_progress: Whether to show tqdm progress bar.

        Returns:
            List of ChatResults in the same order as input.
        """
        # Shared kwargs for every call
        shared = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            top_k_for_sum=top_k_for_sum,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            system_message=system_message,
            extra_body=extra_body,
            logprob_token_index=logprob_token_index,
        )
        max_workers = self.default_max_num_seqs
        results: list[Optional[ChatResult]] = [None] * len(batch_messages)
        semaphore = threading.Semaphore(max_workers)

        def _guarded_call(msgs, **kw):
            with semaphore:
                return self.evaluate_single_chat(msgs, **kw)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_guarded_call, msgs, **shared): idx
                for idx, msgs in enumerate(batch_messages)
            }

            completed = as_completed(future_to_idx)
            if show_progress:
                completed = tqdm(completed, total=len(batch_messages), desc="Evaluating")

            for future in completed:
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results
