"""
Prompt fragments shared by the four modes.

Provides only string constants and small builder functions; each mode assembles
its own content parts.
"""

# ── System prompts ──────────────────────────────────────────────────────────

BOARD_LEGEND = """\
Board legend:
- [*] means a face-down card. Only face-down cards are valid choices.
- [ ] means an already matched/removed empty slot. Never choose an empty slot.
- Any visible card value/suit is currently revealed.
- In image boards, card backs are face-down cards; blank/empty slots are already
  matched/removed and cannot be selected.
"""


OUTPUT_FORMAT_COT = """\
Output format (strict — you MUST include both lines in every response):
Thought: <brief reasoning, at most a few sentences>
Action: <coordinate, e.g. cD>

IMPORTANT: Keep your Thought concise (1-5 sentences). Do NOT repeat prior history or
deliberate at length. State your reasoning briefly, then output the Action line
immediately. Responses missing the "Action:" line will fail to parse.
"""


OUTPUT_FORMAT_NO_COT = """\
Output format (strict — you MUST include this line in every response):
Action: <coordinate, e.g. cD>

IMPORTANT: Do not include reasoning. Responses missing the "Action:" line will fail
to parse.
"""


def output_format_block(cot_enabled: bool = True) -> str:
    return OUTPUT_FORMAT_COT if cot_enabled else OUTPUT_FORMAT_NO_COT


def action_format_hint(cot_enabled: bool = True) -> str:
    if cot_enabled:
        return "Thought: ...\nAction: <coordinate>"
    return "Action: <coordinate>"


SINGLE_SYSTEM_PROMPT_TEMPLATE = """\
You are playing a Memory Match card game. Your goal is to clear all card pairs from the board in as few turns as possible.

Rules:
- Cards are placed face-down on a grid.
- Each turn you flip two cards, one at a time. You see the board after each flip.
- If the two cards match, they are removed and you score 1 point. Otherwise they flip back face-down.
- Use your memory of previously revealed cards to find matches.

{board_legend}

Coordinate system:
- Rows: lowercase letters (a, b, c, ...)
- Columns: uppercase letters (A, B, C, ...)
- A coordinate is row + column, e.g. "cD" means row c, column D.

{output_format}

Retry / forfeit rule:
- If your reply cannot be parsed, or the coordinate is invalid (out of range, already
  matched/removed, or — for the second flip — the same position as the first flip),
  you get up to 2 retries. If all retries fail, the round is forfeited: any card
  already revealed this round is flipped back face-down, and the game moves on.
"""


DUEL_SYSTEM_PROMPT_TEMPLATE = """\
You are Player {player} in a competitive Memory Match card game against an opponent.

Rules:
- Each turn, the current player flips two cards (one at a time, you see the board after each flip).
- If the two cards match, they are removed, the player scores 1 point, and the player gets an extra turn.
- If they don't match, they are flipped back face-down and the turn passes to the opponent.
- You observe the board after every flip (yours and your opponent's). Remember what you and the opponent have seen.

{board_legend}

Coordinate system:
- Rows: lowercase letters (a, b, c, ...). Columns: uppercase letters (A, B, C, ...).
- A coordinate is row + column, e.g. "cD".

{output_format}

Retry / forfeit rule:
- If your reply cannot be parsed, or the coordinate is invalid (out of range, already
  matched/removed, or — for the second flip — the same position as the first flip),
  you get up to 2 retries. If all retries fail, the round is forfeited: any card
  already revealed this round is flipped back face-down and the turn passes to your
  opponent.
"""


NOACTION_HISTORY_NOTE = """\

History format note:
- Past rounds are shown as a short header (round number, match/no-match verdict, or
  forfeit reason) followed by the round's board snapshots. A completed round has two
  snapshots (first flip, both cards revealed) — the outcome (match or no match) is
  stated in the header text, not shown as an extra snapshot. A round that was
  forfeited after the first flip has one snapshot (first flip) — the revealed face
  is a piece of knowledge even though the position is now face-down again. A round
  that was forfeited before any flip has no snapshots. The current-board image at
  the end of the prompt always reflects the state right now (after all past rounds
  have been resolved). The specific coordinates of successful flips are not spelled
  out in text — infer them by comparing consecutive snapshots.
- If any flip required multiple attempts because the previous output was unparseable
  or the chosen card was invalid (out of range, already matched and removed, or the
  same as the already revealed first card), a short note records how many attempts
  were needed and describes each failure without naming the coordinate. Use these
  notes to avoid repeating the same mistakes.
"""


def build_single_system(noaction: bool = False, cot_enabled: bool = True) -> str:
    base = SINGLE_SYSTEM_PROMPT_TEMPLATE.format(
        board_legend=BOARD_LEGEND.rstrip(),
        output_format=output_format_block(cot_enabled).rstrip(),
    )
    return base + (NOACTION_HISTORY_NOTE if noaction else "")


def build_duel_system(player: str, noaction: bool = False, cot_enabled: bool = True) -> str:
    base = DUEL_SYSTEM_PROMPT_TEMPLATE.format(
        player=player,
        board_legend=BOARD_LEGEND.rstrip(),
        output_format=output_format_block(cot_enabled).rstrip(),
    )
    return base + (NOACTION_HISTORY_NOTE if noaction else "")


# ── Retry / error messages ──────────────────────────────────────────────────
# Judge-style: make clear the previous reply was not executed and the board
# is unchanged, regardless of the failure mode.

def retry_parse_fail(cot_enabled: bool = True) -> str:
    if cot_enabled:
        format_hint = (
            "Keep Thought to 1-2 sentences, then immediately write:\n"
            + action_format_hint(cot_enabled=True)
        )
    else:
        format_hint = "Do not include reasoning. Immediately write:\n" + action_format_hint(cot_enabled=False)
    return (
        "Your previous reply was not executed because no \"Action:\" line with a valid coordinate was found. "
        "The board is unchanged. You MUST include the Action line. "
        + format_hint
    )


RETRY_PARSE_FAIL = retry_parse_fail(cot_enabled=True)


# Generic (mode-agnostic) descriptions keyed by InvalidActionError.code.
# These describe what happened without repeating the coord, so noaction runners
# can reuse them without leaking the action.
_INVALID_DESC_GENERIC = {
    "out_of_range": "it selected a coordinate outside the grid",
    "already_removed": "it selected a card that was already matched and removed",
    "same_as_first": "it selected the already revealed first card",
}


def describe_invalid(code: str, fallback_reason: str = "invalid") -> str:
    """Coord-free human description of an invalid-action failure."""
    return _INVALID_DESC_GENERIC.get(code, fallback_reason)


def retry_invalid(code: str, reason: str, include_coord: bool, cot_enabled: bool = True) -> str:
    """Build a judge-style retry message.

    include_coord=True → use env's raw reason (mentions the coord); for normal mode
    where the model already said the coord in its last turn, repeating it is fine.
    include_coord=False → use a generic description (noaction mode: never surface action).
    """
    if include_coord:
        detail = reason
    else:
        detail = describe_invalid(code, fallback_reason="it was invalid")
    return (
        f"Your previous reply was not executed because {detail}. "
        f"The board is unchanged. Choose a valid face-down card.\n"
        f"{action_format_hint(cot_enabled)}"
    )


# Kept for backward compat with normal-mode runners (takes the raw reason).
def retry_invalid_coord(reason: str, cot_enabled: bool = True) -> str:
    return retry_invalid("invalid", reason, include_coord=True, cot_enabled=cot_enabled)


# ── Step instructions ──────────────────────────────────────────────────────

FLIP_FIRST_INSTRUCTION = "Choose a face-down card to flip (your first card this turn)."
FLIP_SECOND_INSTRUCTION = "Choose a different face-down card to flip (your second card this turn)."

# noaction-specific: don't spell out the first flip coord; the board shows it.
FLIP_SECOND_INSTRUCTION_NOACTION = (
    "One card is already revealed on the current board (your first flip this turn). "
    "Choose a different face-down card to flip as your second card."
)
