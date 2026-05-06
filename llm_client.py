"""Provider-agnostic LLM client using LiteLLM.

Wraps litellm.completion() with retry logic and response helpers
so the planner doesn't need to know about provider-specific formats.

Supports any provider LiteLLM supports (Anthropic, OpenAI, Ollama, etc.)
by setting the LLM_MODEL env var and the appropriate API key.

Skill-based defaults to Claude Haiku because the planner's decision
space is small (three skills) and the deterministic Python skills
absorb the per-step reasoning load. See methodology Section 2.1.3.
"""

import json
import logging
import os
import time

import litellm

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-haiku-4-5-20251001")

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True

# Allow tool-call blocks in message history even when `tools=None` on a call.
# Without this, LiteLLM refuses to transform a conversation that already has
# assistant tool_calls + tool results if the current turn has no tool schemas.
litellm.modify_params = True


def call_llm(messages, tools=None, model=None, max_tokens=4096, max_retries=3):
    """Call an LLM via LiteLLM with retry on rate-limit errors.

    Args:
        messages: OpenAI-format message list (including system message).
        tools: OpenAI-format tool definitions, or None.
        model: LiteLLM model string (e.g. "anthropic/claude-haiku-4-5-20251001").
               Defaults to LLM_MODEL env var.
        max_tokens: Maximum tokens in the response.
        max_retries: Number of retries on rate-limit errors.

    Returns:
        LiteLLM/OpenAI-format response object.
    """
    model = model or DEFAULT_MODEL
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model, messages=messages, max_tokens=max_tokens)
            if tools:
                kwargs["tools"] = tools
            return litellm.completion(**kwargs)
        except litellm.RateLimitError:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def wants_tool_use(response):
    """Check if the response wants to call tools."""
    return response.choices[0].finish_reason == "tool_calls"


def is_done(response):
    """Check if the response is a final turn (no more tool calls)."""
    return response.choices[0].finish_reason == "stop"


def get_tool_calls(response):
    """Extract tool calls from a response.

    Returns:
        List of (tool_call_id, function_name, arguments_dict) tuples.
    """
    tool_calls = response.choices[0].message.tool_calls or []
    result = []
    for tc in tool_calls:
        args = tc.function.arguments
        if isinstance(args, str):
            args = json.loads(args)
        result.append((tc.id, tc.function.name, args))
    return result


def get_text_content(response):
    """Extract text content from a response."""
    return response.choices[0].message.content or ""


def assistant_message(response):
    """Get the assistant message for appending to conversation history."""
    return response.choices[0].message


def tool_result_message(tool_call_id, content):
    """Format a tool result message (OpenAI format).

    `content` may be either a string (normal case) or a list of OpenAI-format
    content blocks (e.g. `[{"type": "text", ...}, {"type": "image_url", ...}]`)
    for tools that return images.
    """
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }
