"""
DuelGame abstract base class.

Any game that implements this interface plugs into DuelRunner.
Each game manages its own state machine (e.g. matching pairs' step_1/step_2);
the Runner does not need to know the details.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from framework.types import Observation, StepResult, TraceEntry


class DuelGame(ABC):
    """Abstract interface for a two-player duel game.

    Runner loop:
        while not game_over:
            messages = game.build_messages(player, mode, traces)
            raw = llm.chat(messages)
            action = game.parse_action(raw)
            error = game.validate_action(action)
            result = game.step(action)
            trace = ...
    """

    # ── Game state ──

    @abstractmethod
    def is_game_over(self) -> bool:
        """Whether the game is over."""
        ...

    @property
    @abstractmethod
    def current_player(self) -> int:
        """The player to act now (1 or 2)."""
        ...

    @abstractmethod
    def get_scores(self) -> Dict[int, int]:
        """Current scores {1: score1, 2: score2}."""
        ...

    @abstractmethod
    def get_winner(self) -> int:
        """Winner: 0=tie, 1=P1, 2=P2."""
        ...

    # ── Actions ──

    @abstractmethod
    def legal_actions(self) -> List[str]:
        """List of all currently legal action strings."""
        ...

    @abstractmethod
    def step(self, action: str) -> StepResult:
        """Execute an action and update the game state.

        Returns:
            StepResult with phase name and game-specific info
        """
        ...

    @abstractmethod
    def validate_action(self, action: str) -> Optional[str]:
        """Validate an action.

        Returns:
            None if legal, otherwise an error message string (used for the LLM retry hint)
        """
        ...

    # ── Observation ──

    @abstractmethod
    def get_observation(self, mode: str = "text") -> Observation:
        """Get the current board observation.

        Args:
            mode: "text" or "image"
        """
        ...

    # ── Prompt / Parse (game-defined) ──

    @abstractmethod
    def build_messages(
        self, player: int, mode: str, traces: List[TraceEntry]
    ) -> List[Dict[str, Any]]:
        """Build OpenAI Chat-format messages for the given player.

        Args:
            player: player number (1 or 2)
            mode: "text" or "image"
            traces: all previous trace records

        Returns:
            [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """
        ...

    @abstractmethod
    def parse_action(self, raw: str) -> Optional[str]:
        """Parse an action from the LLM's raw output.

        Returns:
            the parsed action string (e.g. "aA"), or None on failure
        """
        ...
