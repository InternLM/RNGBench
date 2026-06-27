"""
LLM client.

Wraps OpenAI-compatible API calls. Supports text and multimodal messages.
Model-specific params (reasoning_effort, enable_thinking, etc.) are passed via
extra_params. For Gemini models, when generation_config is set it uses the
native generateContent endpoint. Supports image_size: every image's longest
edge is scaled proportionally to that value before sending.
"""

import base64
import copy
import io
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _is_network_error(exc: Exception) -> bool:
    """Heuristic: is this a transient network/server error worth retrying?

    Covers OpenAI-SDK connection/timeout/5xx/rate-limit errors and the raw httpx
    transport errors underneath them. Anything else (e.g. a 400 BadRequest, or a
    parse error) is a real failure and is re-raised immediately.
    """
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


class LLMClient:
    """OpenAI-compatible LLM client.

    Configured via environment variables or constructor args:
      OPENAI_API_BASE: API base URL
      OPENAI_API_KEY: default API key

    Args:
        model: model name
        label: label used for file naming (does not affect the API call)
        extra_params: extra params passed straight to the API (e.g. temperature, reasoning_effort)
        extra_body: extra_body params (e.g. Qwen's enable_thinking)
        api_key_env: env var name for a different API key (e.g. "GEMINI_API_KEY")
        api_key: API key set directly (takes precedence over api_key_env and env vars)
        api_base: custom API base URL, takes precedence over the OPENAI_API_BASE env var
        generation_config: Gemini native generationConfig dict; when set, uses the native endpoint
        safety_settings: Gemini native safetySettings list (optional)
        image_size: target px for the longest image edge; all images scaled proportionally (e.g. 448)
    """

    def __init__(
        self,
        model: str,
        label: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        api_key_env: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_bases: Optional[List[str]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        image_size: Optional[int] = None,
        image_detail: Optional[str] = None,
        proxy: Optional[str] = None,
        sample_seed_env: Optional[str] = None,
        max_network_retries: int = 30,
        retry_backoff: float = 60.0,   # wait a full ~TPM window before retrying (a 0.5s
                                       # storm just re-hits a saturated rate limit forever)
    ):
        self.model = model
        self.label = label or model
        self.extra_params = extra_params or {}
        self.extra_body = extra_body or {}
        self._api_key_env = api_key_env
        self._api_key = api_key
        self._api_base = api_base
        self._api_bases = list(api_bases) if api_bases else None  # round-robin pool
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.image_size = image_size
        self.image_detail = image_detail  # "low" | "high" | "auto" | None
        self._proxy = proxy  # None = direct connection (proxy off); a URL routes through that proxy
        self.sample_seed_env = sample_seed_env  # if a preset declares it, read seed from this env and inject into every request
        self.max_network_retries = max_network_retries  # transient-error retries before giving up
        self.retry_backoff = retry_backoff              # fixed pause (s) between network retries
        # Env overrides (defaults preserve original behavior). On flaky networks a
        # huge multi-image upload can hang the FULL per-request timeout before
        # failing, stalling the retry loop for ~timeout per attempt; lowering
        # LLM_HTTP_TIMEOUT makes a hung upload fail fast and retry on a fresh
        # connection (a healthy big call returns in well under ~90s).
        _to = os.environ.get("LLM_HTTP_TIMEOUT")
        self._http_timeout = float(_to) if _to else 1800.0
        _bo = os.environ.get("LLM_RETRY_BACKOFF")
        if _bo:
            self.retry_backoff = float(_bo)
        # LLM_MAX_NETWORK_RETRIES: raise this so a long bad network window pauses
        # rather than fails a run — the loop keeps retrying until the window
        # reopens instead of giving up after the default ~30 attempts.
        _mr = os.environ.get("LLM_MAX_NETWORK_RETRIES")
        if _mr:
            self.max_network_retries = int(_mr)
        self._client = None
        self._clients_pool = None  # built lazily when api_bases is set
        # Stagger initial round-robin offset by PID so multiple worker processes
        # don't lockstep onto the same endpoint (was: all start at 0 -> base[0]
        # gets every first call across all procs, others starve).
        self._clients_pool_idx = (os.getpid() if api_bases else 0)
        self._httpx_client = None

    def _get_api_key(self):
        if self._api_key:
            return self._api_key
        api_key = os.environ.get(self._api_key_env) if self._api_key_env else None
        return api_key or os.environ.get("OPENAI_API_KEY")

    def _get_api_base(self):
        return self._api_base or os.environ.get("OPENAI_API_BASE")

    def _build_openai_client(self, api_base: Optional[str]):
        import httpx
        from openai import OpenAI
        kwargs = {}
        if api_base:
            kwargs["base_url"] = api_base
        api_key = self._get_api_key()
        if api_key:
            kwargs["api_key"] = api_key
        # Disable HTTP keep-alive: every request opens a fresh connection. On a
        # flaky proxy path, large uploads that drop mid-flight leave half-broken
        # connections in the pool; a long-lived client then keeps reusing them and
        # fails instantly ("Connection error") at a systematically low success
        # rate, while fresh connections still work. Forcing one connection per
        # request stops a broken socket from poisoning later calls — the TCP/TLS
        # setup cost is negligible next to a 30–90s multi-MB upload. Opt out with
        # LLM_HTTP_KEEPALIVE=1.
        _limits = None
        if os.environ.get("LLM_HTTP_KEEPALIVE") not in ("1", "true", "yes"):
            _limits = httpx.Limits(max_keepalive_connections=0)
        kwargs["http_client"] = httpx.Client(proxy=self._proxy, trust_env=False,
                                             timeout=self._http_timeout, limits=_limits) \
            if _limits is not None else \
            httpx.Client(proxy=self._proxy, trust_env=False, timeout=self._http_timeout)
        kwargs["timeout"] = self._http_timeout  # per-request cap (SDK level); override via LLM_HTTP_TIMEOUT
        # Disable the SDK's own retry layer so our _retrying() wrapper is the sole
        # retry authority. Otherwise each of our attempts silently fans out into
        # SDK_max_retries extra full-timeout uploads (compounding to N×timeout per
        # logged attempt) — predictable single-shot attempts retry far better.
        kwargs["max_retries"] = 0
        return OpenAI(**kwargs)

    def _get_client(self):
        # Multiple endpoints: build pool once, then cycle per call.
        if self._api_bases:
            if self._clients_pool is None:
                self._clients_pool = [self._build_openai_client(b) for b in self._api_bases]
            client = self._clients_pool[self._clients_pool_idx % len(self._clients_pool)]
            self._clients_pool_idx += 1
            return client
        if self._client is None:
            self._client = self._build_openai_client(self._get_api_base())
        return self._client

    def _reset_clients(self):
        """Drop cached OpenAI/httpx clients so the next call builds fresh ones.
        Closes underlying httpx connections first to avoid socket leaks."""
        for c in [self._client, *(self._clients_pool or [])]:
            try:
                c and c.close()
            except Exception:
                pass
        try:
            if self._httpx_client is not None:
                self._httpx_client.close()
        except Exception:
            pass
        self._client = None
        self._clients_pool = None
        self._httpx_client = None

    def _get_httpx_client(self):
        if self._httpx_client is None:
            import httpx
            self._httpx_client = httpx.Client(proxy=self._proxy, trust_env=False, timeout=self._http_timeout)
        return self._httpx_client

    def _retrying(self, fn):
        """Run fn(), silently retrying transient network errors.

        Up to ``max_network_retries`` attempts with a fixed ``retry_backoff`` pause;
        on exhaustion (or any non-network error) the exception propagates so the
        caller's game loop can finalize the run with an error.
        """
        last_exc = None
        for attempt in range(self.max_network_retries + 1):
            try:
                return fn()
            except Exception as e:
                if _is_network_error(e) and attempt < self.max_network_retries:
                    _cause = repr(getattr(e, "__cause__", None))
                    logger.warning(
                        f"Network error (attempt {attempt + 1}/{self.max_network_retries + 1}): "
                        f"{type(e).__name__}: {e} | cause={_cause}. Retrying in {self.retry_backoff}s..."
                    )
                    # Rebuild the client so the retry uses a brand-new transport.
                    # A long-lived client accumulates broken transport/connection
                    # state on a flaky proxy path and then fails every request
                    # ("Server disconnected without sending a response"), while a
                    # freshly built client on the identical request still succeeds.
                    # Dropping the cached client forces clean state per retry.
                    self._reset_clients()
                    time.sleep(self.retry_backoff)
                    last_exc = e
                    continue
                raise
        raise last_exc  # unreachable

    def chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send the full messages array; returns {"content": str, "reasoning": str|None, "usage": dict|None}."""
        if self.image_size:
            messages = self._resize_images(messages)
        if self.image_detail:
            messages = self._apply_image_detail(messages)

        if self.generation_config:
            return self._chat_gemini_native(messages)

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        kwargs.update(self.extra_params)

        # Sampling seed (opt-in): only when a preset explicitly declares the env
        # var via sample_seed_env (e.g. "NON_MARKOV_SAMPLE_SEED") does LLMClient
        # inject the seed. This way non-OpenAI-compat paths (Gemini native, or
        # gateways that don't support seed) are unaffected by default.
        if self.sample_seed_env:
            _ss = os.environ.get(self.sample_seed_env)
            if _ss not in (None, ""):
                try:
                    kwargs["seed"] = int(_ss)
                except ValueError:
                    pass

        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        # Re-fetch the client inside the retried call: _retrying() may rebuild it
        # after a network failure, so capturing it once would reuse a closed client.
        response = self._retrying(lambda: self._get_client().chat.completions.create(**kwargs))
        msg = response.choices[0].message

        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        usage = getattr(response, "usage", None)
        usage_dict = usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else None

        # Return content exactly as the API gave it. Thinking models put their
        # reasoning in `reasoning_content`; that trace must NEVER be folded into
        # `content`, or it would be appended to the conversation history and
        # re-sent every round (token blow-up / TPM wall). An empty content is a
        # real "no answer" — the caller's parse-fail / forfeit path handles it.
        content = msg.content or ""

        return {
            "content": content,
            "reasoning": reasoning,
            "usage": usage_dict,
        }

    # ── Gemini native endpoint ───────────────────────────────────────────

    def _chat_gemini_native(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call via the Gemini native generateContent endpoint with full generationConfig support."""
        api_base = self._get_api_base() or ""
        # /boyue/v1 → /boyue/v1beta/models/{model}:generateContent
        base = re.sub(r"/v1/?$", "", api_base)
        url = f"{base}/v1beta/models/{self.model}:generateContent"

        api_key = self._get_api_key()

        # Convert OpenAI messages -> Gemini contents + systemInstruction
        system_parts = []
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_parts.append({"text": content})
                continue

            gemini_role = "model" if role == "assistant" else "user"
            parts = self._convert_content_to_parts(content)
            if parts:
                contents.append({"role": gemini_role, "parts": parts})

        # Build the request body
        body: Dict[str, Any] = {"contents": contents}
        if system_parts:
            body["systemInstruction"] = {"parts": system_parts}

        # generationConfig: user config wins; temperature defaults from extra_params
        gen_config = dict(self.generation_config)
        gen_config.setdefault("temperature", self.extra_params.get("temperature", 0.0))
        body["generationConfig"] = gen_config

        if self.safety_settings:
            body["safetySettings"] = self.safety_settings

        # Send the request (transient network errors retried by _retrying)
        http = self._get_httpx_client()
        resp = self._retrying(lambda: http.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            content=json.dumps(body, ensure_ascii=False),
        ))
        resp.raise_for_status()
        data = resp.json()

        # Parse the response
        content_text = ""
        reasoning_text = None
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if "thought" in part and part.get("thought"):
                    reasoning_text = (reasoning_text or "") + part.get("text", "")
                elif "text" in part:
                    content_text += part["text"]

        return {
            "content": content_text,
            "reasoning": reasoning_text,
        }

    @staticmethod
    def _convert_content_to_parts(content) -> List[Dict[str, Any]]:
        """Convert OpenAI content format to Gemini parts format."""
        if isinstance(content, str):
            return [{"text": content}] if content else []

        parts = []
        for item in content:
            item_type = item.get("type", "")
            if item_type == "text":
                parts.append({"text": item["text"]})
            elif item_type == "image_url":
                url = item["image_url"]["url"]
                # data:image/jpeg;base64,... → inlineData
                m = re.match(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
                if m:
                    parts.append({
                        "inlineData": {
                            "mimeType": m.group(1),
                            "data": m.group(2),
                        }
                    })
        return parts

    # ── Image detail injection ───────────────────────────────────────────

    def _apply_image_detail(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Inject a detail field into each image_url (Ark / OpenAI compatible)."""
        messages = copy.deepcopy(messages)
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") == "image_url":
                    iu = part["image_url"]
                    if "detail" not in iu:
                        iu["detail"] = self.image_detail
        return messages

    # ── Image resize ─────────────────────────────────────────────────────

    def _resize_images(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scale every image's longest edge in messages proportionally to image_size.

        Returns a deepcopy of messages; the original data is not mutated (traces
        are shared across players in duel mode).
        """
        from PIL import Image

        target = self.image_size
        messages = copy.deepcopy(messages)

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") != "image_url":
                    continue
                url = part["image_url"]["url"]
                m = re.match(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
                if not m:
                    continue
                img = Image.open(io.BytesIO(base64.b64decode(m.group(2))))
                if max(img.size) == target:
                    continue
                ratio = target / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                new_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                part["image_url"]["url"] = f"data:image/jpeg;base64,{new_b64}"

        return messages
