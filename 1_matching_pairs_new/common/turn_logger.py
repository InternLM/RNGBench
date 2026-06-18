"""
TurnLogger — records one turn per model reply (including retries, except network failures).

Writes game.json in real time, flushing after each log_call; finalize(result)
fills in the final result fields.
See the plan file for the JSON structure.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TurnLogger:
    """Record the full game: config + all turns + final result."""

    def __init__(
        self,
        output_path: Path,
        config: Dict[str, Any],
        *,
        resume_turns: Optional[List[Dict[str, Any]]] = None,
    ):
        """`resume_turns`: pre-existing turn dicts (from a partial game.json) to
        carry over. New log_call appends after them; turn_id continues monotonically."""
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if resume_turns:
            kept = [copy.deepcopy(t) for t in resume_turns]
            next_id = max((t.get("turn_id") or 0) for t in kept) + 1
        else:
            kept = []
            next_id = 1
        self._data: Dict[str, Any] = {
            "config": dict(config),
            "result": None,
            "turns": kept,
        }
        self._next_turn_id = next_id
        self._flush()

    def log_call(
        self,
        call_type: str,                                  # "flip_first" | "flip_second" | "retry"
        round_idx: int,
        player: Optional[str],
        messages_sent: List[Dict[str, Any]],             # stored in image_path format
        response: Dict[str, Any],                        # {"content", "reasoning"}
        parse: Dict[str, Any],                           # {"ok", "coord", "error"}
        env_result: Dict[str, Any],                      # {"applied", "revealed_face", "phase_after"}
        retry_of: Optional[int] = None,
    ) -> int:
        """Record one turn; returns the turn_id."""
        turn_id = self._next_turn_id
        self._next_turn_id += 1
        self._data["turns"].append({
            "turn_id": turn_id,
            "call_type": call_type,
            "retry_of": retry_of,
            "round": round_idx,
            "player": player,
            "messages_sent": copy.deepcopy(messages_sent),
            "response": dict(response),
            "parse": dict(parse),
            "env_result": dict(env_result),
        })
        self._flush()
        return turn_id

    def finalize(self, result: Dict[str, Any]) -> None:
        self._data["result"] = dict(result)
        self._flush()

    def _flush(self) -> None:
        tmp = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.output_path)

    @property
    def response_count(self) -> int:
        return len(self._data["turns"])
