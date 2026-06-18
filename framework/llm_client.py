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
import os
import re
from typing import Any, Dict, List, Optional


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
        kwargs["http_client"] = httpx.Client(proxy=self._proxy, trust_env=False)
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

    def _get_httpx_client(self):
        if self._httpx_client is None:
            import httpx
            self._httpx_client = httpx.Client(proxy=self._proxy, trust_env=False, timeout=300)
        return self._httpx_client

    def chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send the full messages array; returns {"content": str, "reasoning": str|None, "usage": dict|None}."""
        if self.image_size:
            messages = self._resize_images(messages)
        if self.image_detail:
            messages = self._apply_image_detail(messages)

        if self.generation_config:
            return self._chat_gemini_native(messages)

        client = self._get_client()

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

        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        usage = getattr(response, "usage", None)
        usage_dict = usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else None

        # Some thinking models (e.g. lmdeploy base05) emit the final
        # `Thought:/Action:` only in reasoning_content and leave content empty.
        # Fall back to reasoning when content is blank so the parser still sees
        # the action (no-op for models that populate content normally).
        content = msg.content or ""
        if not content.strip() and reasoning:
            content = reasoning

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

        # Send the request
        http = self._get_httpx_client()
        resp = http.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            content=json.dumps(body, ensure_ascii=False),
        )
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
