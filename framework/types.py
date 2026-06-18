"""
Shared data types for the duel framework.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Observation:
    """Game observation (multimodal representation of the board state)."""
    text: str                           # text board
    image_url: Optional[str] = None     # base64 JPEG data URL (image mode)
    full_state: str = ""                # debug: full state visible


@dataclass
class StepResult:
    """Result of a single step."""
    phase: str          # game-defined phase name (e.g. "after_step_1", "after_resolve_match")
    info: dict = field(default_factory=dict)  # game-defined info


@dataclass
class TraceAttempt:
    """Record of a single LLM attempt."""
    attempt: int
    raw_output: str
    reasoning: Optional[str]
    parsed: Optional[str]       # parsed action (e.g. "aA")
    error: Optional[str]        # None = success


@dataclass
class TraceEntry:
    """Full record of one action step."""
    index: int                          # global action index (starts at 1)
    phase: str                          # game-defined phase
    player: int                         # 1 or 2
    action: Optional[str]               # the action finally executed
    used_fallback: bool
    attempts: List[TraceAttempt]
    scores: Dict[int, int]              # score snapshot after the action
    observation: Observation            # board observation after the action
    info: dict = field(default_factory=dict)  # game-defined info


@dataclass
class DuelResult:
    """Final result of a duel."""
    game_name: str                      # "matching_pairs", "tic_tac_toe", etc.
    scores: Dict[int, int]              # {1: score1, 2: score2}
    winner: int                         # 0=tie, 1=P1, 2=P2
    total_actions: int                  # total number of actions (successfully executed steps)
    traces: List[TraceEntry]
    total_llm_calls: int = 0            # actual number of LLM calls (incl. retries)
    game_config: dict = field(default_factory=dict)   # game-specific config
    model_config: dict = field(default_factory=dict)  # {1: {...}, 2: {...}}
    termination_reason: str = "game_over"  # "game_over" | "max_actions_reached" | "error"
    error: Optional[str] = None         # error message if aborted by exception
