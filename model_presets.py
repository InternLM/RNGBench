"""
Shared model presets for both games (Matching Pairs + 3D Maze).

Single source of truth at the repo root. Both `1_matching_pairs_new/modes/*`
and `2_3d_maze/run.py` import this module and bind to the shared
`framework.llm_client.LLMClient`. Adding/changing a model here takes effect in
both games at once.

A preset maps a short name -> endpoint + sampling. Supported fields:

  model            (str)   model name the server/provider expects
  api_base         (str)   OpenAI-compatible base URL. Note the path differs by
                           provider — OpenAI/vLLM/lmdeploy end in ``/v1``,
                           Volcengine Ark ends in ``/api/v3``. Omit to fall back
                           to the OPENAI_API_BASE env var.
  api_bases        (list)  several base URLs; the client load-balances across
                           them round-robin (handy for multiple vLLM replicas).
  api_key          (str)   inline key. Use "EMPTY" for no-auth local servers.
  api_key_env      (str)   name of an env var holding the key (preferred — keeps
                           secrets out of the code; loaded from the repo .env).
  extra_params     (dict)  merged into the chat/completions body. Use
                           ``max_tokens`` for OpenAI-compatible servers and
                           ``max_completion_tokens`` for OpenAI / Ark.
  extra_body       (dict)  non-standard body fields (top_k, repetition_penalty,
                           chat_template_kwargs / thinking toggles, ...).
  generation_config(dict)  Gemini-native generateContent config (no api_base).
  image_size       (int)   long side the board image is resized to before send.
  image_detail     (str)   OpenAI image "detail" hint (low/high).
  sample_seed_env  (str)   env var whose value is used as the sampler seed for
                           reproducibility (servers that support ``seed``).

NOTE: the endpoints below are illustrative placeholders (localhost / example
hosts). Point them at your own deployment and put real keys in ``.env``.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure the repo root is importable so `framework` resolves regardless of cwd.
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from framework.llm_client import LLMClient

# ── Model presets ───────────────────────────────────────────────────────────

MODEL_PRESETS = {
    # ── Template: any OpenAI-compatible server (vLLM / lmdeploy / SGLang / Ollama) ──
    # Copy this, point api_base at your server, and pass `--model my-model`.
    "local-openai-compatible": {
        "model": "your-served-model-name",
        "image_size": 512,
        "api_base": "http://localhost:8000/v1",   # OpenAI-compatible servers end in /v1
        "api_key": "EMPTY",                        # or use "api_key_env": "OPENAI_API_KEY"
        "extra_params": {"temperature": 0.8, "top_p": 0.95, "max_tokens": 512},
        "extra_body": {"top_k": 50},
        "sample_seed_env": "NON_MARKOV_SAMPLE_SEED",
    },

    # ── OpenAI cloud (or any OpenAI-compatible gateway) ──
    # Uses OPENAI_API_BASE / OPENAI_API_KEY from .env when api_base is omitted.
    # OpenAI models use `max_completion_tokens` (not `max_tokens`).
    "gpt-5.4": {
        "model": "gpt-5.4",
        "image_size": 512,
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "extra_params": {"max_completion_tokens": 16384},
    },
    # reasoning-effort sweep (none keeps a short cap; low/medium need a large budget
    # so the reasoning trace does not eat the final action). Supported efforts:
    # 'none', 'low', 'medium', 'high', 'xhigh'.
    "gpt-5.4-nothink": {
        "model": "gpt-5.4",
        "image_size": 512,
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "extra_params": {"reasoning_effort": "none", "max_completion_tokens": 512},
    },
    "gpt-5.4-lowthink": {
        "model": "gpt-5.4",
        "image_size": 512,
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "extra_params": {"reasoning_effort": "low", "max_completion_tokens": 32768},
    },
    "gpt-5.4-mediumthink": {
        "model": "gpt-5.4",
        "image_size": 512,
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "extra_params": {"reasoning_effort": "medium", "max_completion_tokens": 32768},
    },

    # ── Gemini (native generateContent endpoint; no api_base, uses generation_config) ──
    # Key read from the GEMINI_API_KEY env var. mediaResolution=LOW fixes ~280 tok/image.
    "gemini-3.1-pro": {
        "model": "gemini-3.1-pro",
        "api_key_env": "GEMINI_API_KEY",
        "generation_config": {
            "mediaResolution": "MEDIA_RESOLUTION_LOW",
            "maxOutputTokens": 512,
        },
        "extra_params": {},
    },
    "gemini-3.1-pro-thinking": {
        "model": "gemini-3.1-pro",
        "api_key_env": "GEMINI_API_KEY",
        "generation_config": {
            "mediaResolution": "MEDIA_RESOLUTION_LOW",
            "thinkingConfig": {"thinkingLevel": "LOW"},
            "maxOutputTokens": 16384,
        },
        "extra_params": {},
    },

    # ── Self-hosted Qwen3.5-397B (vLLM, OpenAI-compatible) ──
    # `api_bases` (plural) load-balances across replicas round-robin.
    # nothink is the default; the -think variant follows the model card's
    # recommended thinking-mode sampling (temp 1.0 / top_p 0.95 / top_k 20).
    "qwen3.5-397b": {
        "model": "Qwen3.5-397B-A17B",
        "image_size": 512,
        "api_bases": [
            "http://localhost:8000/v1",
            "http://localhost:8001/v1",
        ],
        "api_key": "EMPTY",
        "extra_params": {"temperature": 0.7, "top_p": 0.8, "max_tokens": 512, "presence_penalty": 1.5},
        "extra_body": {"top_k": 20, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": False}},
        "sample_seed_env": "NON_MARKOV_SAMPLE_SEED",
    },
    "qwen3.5-397b-think": {
        "model": "Qwen3.5-397B-A17B",
        "image_size": 512,
        "api_base": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "extra_params": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768, "presence_penalty": 1.5},
        "extra_body": {"top_k": 20, "min_p": 0, "repetition_penalty": 1.0},
        "sample_seed_env": "NON_MARKOV_SAMPLE_SEED",
    },

    # ── Self-hosted Qwen3.5-9B (vLLM) — the base for the fine-tuning study ──
    "qwen3.5-9b": {
        "model": "Qwen3.5-9B",
        "image_size": 512,
        "api_base": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "extra_params": {"temperature": 0.7, "top_p": 0.8, "max_tokens": 512, "presence_penalty": 1.5},
        "extra_body": {"top_k": 20, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": False}},
        "sample_seed_env": "NON_MARKOV_SAMPLE_SEED",
    },
    # Point this at a server hosting your fine-tuned checkpoint (e.g. rmix32k).
    "qwen3.5-9b-sft": {
        "model": "your-finetuned-checkpoint",
        "image_size": 512,
        "api_base": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "extra_params": {"temperature": 0.7, "top_p": 0.8, "max_tokens": 512, "presence_penalty": 1.5},
        "extra_body": {"top_k": 20, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": False}},
        "sample_seed_env": "NON_MARKOV_SAMPLE_SEED",
    },

    # ── Self-hosted Kimi-K2.5 (vLLM, OpenAI-compatible) ──
    "kimi-k2.5": {
        "model": "Kimi-K2.5",
        "image_size": 448,
        "api_base": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "extra_params": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 512},
        "extra_body": {"chat_template_kwargs": {"thinking": False}},
        "sample_seed_env": "NON_MARKOV_SAMPLE_SEED",
    },
    "kimi-k2.5-think": {
        "model": "Kimi-K2.5",
        "image_size": 448,
        "api_base": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "extra_params": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768},
        "sample_seed_env": "NON_MARKOV_SAMPLE_SEED",
    },

    # ── Volcengine Ark / Doubao Seed (NOTE: base path is /api/v3, not /v1) ──
    # `model` is your Ark endpoint id; key read from the ARK_API_KEY env var.
    "seed-2.0-lite": {
        "model": "your-ark-endpoint-id",
        "image_size": 672,
        "image_detail": "low",
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_env": "ARK_API_KEY",
        "extra_params": {"max_completion_tokens": 512},
        "extra_body": {"thinking": {"type": "disabled"}},
        "sample_seed_env": "NON_MARKOV_SAMPLE_SEED",
    },
    "seed-2.0-lite-think": {
        "model": "your-ark-endpoint-id",
        "image_size": 672,
        "image_detail": "low",
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_env": "ARK_API_KEY",
        "extra_params": {"max_completion_tokens": 32768},
        "extra_body": {"thinking": {"type": "enabled"}},
        "sample_seed_env": "NON_MARKOV_SAMPLE_SEED",
    },
}


def make_client(model_name: str, label: Optional[str] = None) -> LLMClient:
    """Build an LLMClient from a model name."""
    if model_name in MODEL_PRESETS:
        cfg = MODEL_PRESETS[model_name]
        return LLMClient(
            model=cfg["model"],
            label=label or model_name,
            extra_params=cfg.get("extra_params", {}),
            extra_body=cfg.get("extra_body", {}),
            api_key_env=cfg.get("api_key_env"),
            api_key=cfg.get("api_key"),
            api_base=cfg.get("api_base"),
            api_bases=cfg.get("api_bases"),
            generation_config=cfg.get("generation_config"),
            safety_settings=cfg.get("safety_settings"),
            image_size=cfg.get("image_size"),
            image_detail=cfg.get("image_detail"),
            proxy=cfg.get("proxy"),
            sample_seed_env=cfg.get("sample_seed_env"),
        )
    return LLMClient(model=model_name, label=label or model_name)


def parse_grid_size(s: str):
    """Parse a board size in '6x6' format."""
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid grid size: {s}. Expected format: 4x4")
    return int(parts[0]), int(parts[1])
