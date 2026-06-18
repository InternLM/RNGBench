"""
Thin LLM-call wrapper: handles network-error retries, not logged by turn_logger.

Non-network errors (the model reply itself) are handled by the upper-level
runner (which records a turn).
"""

import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _is_network_error(exc: Exception) -> bool:
    """Heuristic: is this a network/server error (worth silently retrying)?"""
    import httpx
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
    network_types = (
        APIConnectionError, APITimeoutError, InternalServerError, RateLimitError,
        httpx.ConnectError, httpx.ReadError, httpx.TimeoutException,
    )
    return isinstance(exc, network_types)


def call_llm_with_retry(
    client,
    messages: List[Dict[str, Any]],
    max_network_retries: int = 3,
    backoff_base: float = 2.0,
) -> Dict[str, Any]:
    """Call client.chat(messages), silently retrying on network errors.

    Returns {"content": str, "reasoning": str | None}.
    Non-network errors (including parse failures of the model reply) are re-raised.
    """
    last_exc = None
    for attempt in range(max_network_retries + 1):
        try:
            return client.chat(messages)
        except Exception as e:
            if _is_network_error(e) and attempt < max_network_retries:
                wait = backoff_base ** attempt
                logger.warning(
                    f"Network error (attempt {attempt + 1}/{max_network_retries + 1}): "
                    f"{type(e).__name__}: {e}. Retrying in {wait}s..."
                )
                time.sleep(wait)
                last_exc = e
                continue
            raise
    raise last_exc  # should be unreachable
