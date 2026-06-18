"""
Convert turn-log-style messages (where image_url.url is a relative path) into the
standard OpenAI messages LLMClient expects (where image_url.url is a data URL).

Convention: the storage and logging layers keep the original OpenAI shape
{"type": "image_url", "image_url": {"url": <path>}}; before calling the LLM,
materialize swaps non-data-URL paths for base64 data URLs.
"""

import copy
from typing import Any, Dict, List


def materialize(messages: List[Dict[str, Any]], image_store) -> List[Dict[str, Any]]:
    """Deep-copy messages, swapping every non-data-URL image_url.url for a base64 data URL."""
    out = copy.deepcopy(messages)
    for msg in out:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            # new format: image_url holds a path
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url and not url.startswith("data:"):
                    part["image_url"]["url"] = image_store.load_as_data_url(url)
            # legacy format: image_path
            elif part.get("type") == "image_path":
                data_url = image_store.load_as_data_url(part["path"])
                part.clear()
                part["type"] = "image_url"
                part["image_url"] = {"url": data_url}
    return out
