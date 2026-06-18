"""
DuelRunner — generic two-player duel loop.

Fully game-agnostic. All game-specific logic is delegated through the DuelGame
interface. Responsibilities: loop scheduling -> LLM calls -> retry/fallback ->
trace recording -> result output.
"""

import base64
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from framework.game import DuelGame
from framework.llm_client import LLMClient
from framework.types import (
    DuelResult,
    Observation,
    TraceAttempt,
    TraceEntry,
)

logger = logging.getLogger(__name__)


class DuelRunner:
    """Generic two-player duel runner.

    Usage:
        game = MatchingPairsDuel(...)  # or any DuelGame implementation
        runner = DuelRunner(game, clients, player_modes)
        result = runner.run()
    """

    def __init__(
        self,
        game: DuelGame,
        clients: Dict[int, LLMClient],
        player_modes: Dict[int, str],
        *,
        max_actions: int = 2000,
        max_retries: int = 2,
        rng_seed: int = 0,
        output_dir: str = "results",
        game_name: str = "unknown",
        dump_messages: bool = False,
    ):
        self.game = game
        self.clients = clients
        self.player_modes = player_modes
        self.max_actions = max_actions
        self.max_retries = max_retries
        self.rng = random.Random(rng_seed)
        self.output_dir = output_dir
        self.game_name = game_name
        self.dump_messages = dump_messages

        # Image output dir (initialized at run time)
        self._run_dir: Optional[Path] = None
        self._game_id: Optional[str] = None
        self._img_dir: Optional[Path] = None
        self._img_count = 0
        self._msg_dump_dir: Optional[Path] = None
        self._log_file = None

    def _init_output_dir(self, run_dir, game_id: str):
        """Initialize output dirs: images, logs, etc."""
        self._run_dir = Path(run_dir)
        self._game_id = game_id
        self._img_dir = self._run_dir / "images" / game_id
        self._img_dir.mkdir(parents=True, exist_ok=True)
        self._img_count = 0
        # log file
        log_dir = self._run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = open(log_dir / f"{game_id}.log", "w", encoding="utf-8")

    def _write_log(self, text: str):
        """Write to the log file."""
        if self._log_file:
            self._log_file.write(text + "\n")
            self._log_file.flush()

    def _save_observation_image(self, obs: Observation) -> Optional[str]:
        """Save the base64 image in an observation to disk, returning a relative path.
        Note: obs.image_url is not modified; the trace keeps the base64 in memory
        for later build_messages calls.
        """
        if not obs.image_url or not obs.image_url.startswith("data:"):
            return None
        if not self._img_dir:
            return None

        self._img_count += 1
        filename = f"action_{self._img_count:03d}.jpg"
        filepath = self._img_dir / filename

        try:
            b64_data = obs.image_url.split(",", 1)[1]
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(b64_data))
        except Exception as e:
            logger.warning(f"Failed to save image: {e}")
            return None

        return f"images/{self._game_id}/{filename}"

    def _dump_messages(self, action_count: int, player: int, messages: List[Dict]):
        """Dump the full LLM input messages to disk (base64 images saved as files)."""
        if not self.dump_messages or not self._run_dir:
            return
        if self._msg_dump_dir is None:
            self._msg_dump_dir = self._run_dir / "message_dumps" / self._game_id
            self._msg_dump_dir.mkdir(parents=True, exist_ok=True)

        dump_dir = self._msg_dump_dir / f"action_{action_count:03d}_P{player}"
        dump_dir.mkdir(exist_ok=True)

        img_idx = 0
        serialized = []
        for msg in messages:
            msg_copy = {"role": msg.get("role", "")}
            content = msg.get("content", "")

            if isinstance(content, str):
                msg_copy["content"] = content
            elif isinstance(content, list):
                parts_out = []
                for part in content:
                    if part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            img_idx += 1
                            img_file = f"img_{img_idx:03d}.jpg"
                            try:
                                b64_data = url.split(",", 1)[1]
                                with open(dump_dir / img_file, "wb") as f:
                                    f.write(base64.b64decode(b64_data))
                            except Exception:
                                pass
                            parts_out.append({"type": "image_url", "image_url": {"url": img_file}})
                        else:
                            parts_out.append(part)
                    else:
                        parts_out.append(part)
                msg_copy["content"] = parts_out
            else:
                msg_copy["content"] = content

            serialized.append(msg_copy)

        with open(dump_dir / "messages.json", "w") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)

    def run(self) -> DuelResult:
        """Run the full duel and return the result."""
        game = self.game
        traces: List[TraceEntry] = []
        action_count = 0
        llm_call_count = 0
        error_msg = None
        termination_reason = "game_over"

        logger.info(f"Duel start: {self.game_name}")

        try:
            any_image = any(m == "image" for m in self.player_modes.values())
            obs_mode = "image" if any_image else "text"

            while not game.is_game_over() and action_count < self.max_actions:
                player = game.current_player
                mode = self.player_modes[player]

                # 1. Build messages
                messages = game.build_messages(player, mode, traces)

                # 1.5 Dump the full input messages (for debugging)
                self._dump_messages(action_count + 1, player, messages)

                # 2. Choose action (with retry)
                choose_result = self._choose_action(player, messages)

                if choose_result is None:
                    # All retries failed -> forfeit this step
                    action_count += 1
                    forfeit_attempts = getattr(self, "_last_attempts", [])
                    llm_call_count += len(forfeit_attempts)
                    if hasattr(game, 'forfeit_action'):
                        result = game.forfeit_action()
                    else:
                        # Games without a forfeit method: just skip
                        continue
                    obs = game.get_observation(obs_mode)
                    img_path = self._save_observation_image(obs)
                    trace_info = dict(result.info) if result.info else {}
                    if img_path:
                        trace_info["_img_path"] = img_path
                    # Save the last attempt's action and error for build_messages
                    if forfeit_attempts:
                        last = forfeit_attempts[-1]
                        trace_info["_attempted_action"] = last.parsed
                        trace_info["_attempt_error"] = last.error
                        trace_info["_raw_output"] = last.raw_output
                    traces.append(TraceEntry(
                        index=action_count,
                        phase=result.phase,
                        player=player,
                        action=None,
                        used_fallback=False,
                        attempts=forfeit_attempts,
                        scores=game.get_scores(),
                        observation=obs,
                        info=trace_info,
                    ))
                    for att in forfeit_attempts:
                        self._write_log(f"  [P{player} ASSISTANT] {att.raw_output}")
                        if att.error:
                            self._write_log(f"  [ERROR] {att.error}")
                    self._write_log(f"Action #{action_count}: P{player} FORFEIT → {result.phase}")
                    logger.info(f"  Action #{action_count}: P{player} FORFEIT → {result.phase}")
                    continue

                action, used_fallback, attempts = choose_result
                llm_call_count += len(attempts)

                # 3. Execute
                result = game.step(action)
                action_count += 1

                # 4. Get observation & save image
                obs = game.get_observation(obs_mode)
                resolve_obs = None

                # 5. Build trace info
                trace_info = dict(result.info) if result.info else {}

                # For resolve phases: the trace's observation uses the reveal image
                # (both cards flipped); the post-resolve board state is stored as
                # _resolve_observation.
                is_resolve = result.phase in ("after_resolve_match", "after_resolve_miss")
                reveal_url = trace_info.get("_reveal_image_url")
                reveal_text = trace_info.get("_reveal_text", "")

                if is_resolve and (reveal_url or reveal_text):
                    # reveal image as the main observation (both cards flipped)
                    reveal_obs = Observation(
                        text=reveal_text,
                        image_url=reveal_url,
                        full_state=obs.full_state,
                    )
                    # save the reveal image to disk
                    reveal_img_path = self._save_observation_image(reveal_obs)
                    if reveal_img_path:
                        trace_info["_img_path"] = reveal_img_path

                    # post-resolve board (matched-removed / flipped-back) as an extra field
                    resolve_img_path = self._save_observation_image(obs)
                    if resolve_img_path:
                        trace_info["_resolve_img_path"] = resolve_img_path
                    trace_info["_resolve_text"] = obs.text
                    trace_info["_resolve_image_url"] = obs.image_url

                    # the trace's observation uses the reveal image
                    obs = reveal_obs
                else:
                    # step_1 normal: use obs directly
                    img_path = self._save_observation_image(obs)
                    if img_path:
                        trace_info["_img_path"] = img_path

                # 6. Record trace
                traces.append(TraceEntry(
                    index=action_count,
                    phase=result.phase,
                    player=player,
                    action=action,
                    used_fallback=used_fallback,
                    attempts=attempts,
                    scores=game.get_scores(),
                    observation=obs,
                    info=trace_info,
                ))

                # 7. Write log
                log_line = (
                    f"Action #{action_count}: P{player} {action} "
                    f"→ {result.phase} | scores={game.get_scores()}"
                )
                if used_fallback:
                    log_line += " [FALLBACK]"
                self._write_log(log_line)
                # record the model's raw output
                for att in attempts:
                    self._write_log(f"  [P{player} ASSISTANT] {att.raw_output}")
                    if att.error:
                        self._write_log(f"  [ERROR] {att.error}")
                # record the board text
                if obs.text:
                    self._write_log(f"  [BOARD]\n{obs.text}")

                logger.info(
                    f"  Action #{action_count}: P{player} {action} "
                    f"→ {result.phase} | scores={game.get_scores()}"
                    f"{' [FALLBACK]' if used_fallback else ''}"
                )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            termination_reason = "error"
            logger.error(f"Duel aborted at action #{action_count}: {error_msg}")
            self._write_log(f"ERROR: {error_msg}")

        if error_msg is None and not game.is_game_over() and action_count >= self.max_actions:
            termination_reason = "max_actions_reached"
            logger.warning(
                f"Duel stopped after reaching max_actions={self.max_actions} "
                f"before game over."
            )

        # Close log
        if self._log_file:
            scores = game.get_scores()
            winner_val = game.get_winner() if not error_msg else 0
            self._write_log(
                f"\n=== RESULT: P1={scores.get(1,0)}, P2={scores.get(2,0)}, "
                f"winner=P{winner_val}, actions={action_count}, "
                f"termination={termination_reason} ==="
            )
            self._log_file.close()
            self._log_file = None

        # Build result
        scores = game.get_scores()
        winner = game.get_winner() if not error_msg else 0

        duel_result = DuelResult(
            game_name=self.game_name,
            scores=scores,
            winner=winner,
            total_actions=action_count,
            total_llm_calls=llm_call_count,
            traces=traces,
            model_config={
                p: {"model": c.model, "label": c.label, "mode": self.player_modes.get(p, "text")}
                for p, c in self.clients.items()
            },
            termination_reason=termination_reason,
            error=error_msg,
        )

        logger.info(
            f"Duel end: scores={scores}, winner=P{winner}, "
            f"actions={action_count}, termination={termination_reason}"
            + (f", error={error_msg}" if error_msg else "")
        )

        return duel_result

    def _choose_action(
        self,
        player: int,
        messages: List[Dict[str, Any]],
    ) -> Optional[Tuple[str, bool, List[TraceAttempt]]]:
        """Have the LLM choose an action, with retries. Returns None if all fail."""
        game = self.game
        client = self.clients[player]
        attempts: List[TraceAttempt] = []

        for attempt_idx in range(self.max_retries + 1):
            resp = client.chat(messages)
            raw = resp["content"]
            reasoning = resp["reasoning"]

            # Try to parse
            action = game.parse_action(raw)
            if action is None:
                error = "Could not parse a valid action from your output."
                attempts.append(TraceAttempt(
                    attempt=attempt_idx, raw_output=raw,
                    reasoning=reasoning, parsed=None, error=error,
                ))
                if attempt_idx < self.max_retries:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": error + " Please try again."})
                logger.warning(f"  P{player} attempt {attempt_idx + 1}: parse failed")
                continue

            # Validate legality
            validation_error = game.validate_action(action)
            if validation_error:
                attempts.append(TraceAttempt(
                    attempt=attempt_idx, raw_output=raw,
                    reasoning=reasoning, parsed=action, error=validation_error,
                ))
                if attempt_idx < self.max_retries:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": validation_error})
                logger.warning(
                    f"  P{player} attempt {attempt_idx + 1}: "
                    f"{action} invalid: {validation_error}"
                )
                continue

            # Success
            attempts.append(TraceAttempt(
                attempt=attempt_idx, raw_output=raw,
                reasoning=reasoning, parsed=action, error=None,
            ))
            return action, False, attempts

        # All retries exhausted, return None (handled as forfeit by the runner)
        self._last_attempts = attempts
        logger.warning(f"  P{player}: all retries failed")
        return None


def save_duel_result(result: DuelResult, run_dir, game_id: str) -> Path:
    """Save the duel result to JSON."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{game_id}.json"
    data = _serialize_result(result)

    path = run_dir / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Duel result saved to {path}")
    return path


def _serialize_result(result: DuelResult) -> dict:
    """Serialize a DuelResult, replacing base64 images with file paths."""
    data = asdict(result)
    for trace in data.get("traces", []):
        obs = trace.get("observation", {})
        info = trace.get("info", {})
        # replace the observation's base64 with the saved file path
        img_path = info.pop("_img_path", None)
        if img_path:
            obs["image_url"] = img_path
        elif isinstance(obs.get("image_url"), str) and obs["image_url"].startswith("data:"):
            obs["image_url"] = "[base64_image_omitted]"
        # replace the reveal image's base64
        for key in ("_reveal_image_url", "_resolve_image_url"):
            val = info.get(key, "")
            if isinstance(val, str) and val.startswith("data:"):
                info[key] = "[base64_image_omitted]"
        # replace with the resolve file path
        resolve_path = info.pop("_resolve_img_path", None)
        if resolve_path:
            info["_resolve_image_url"] = resolve_path
    return data
