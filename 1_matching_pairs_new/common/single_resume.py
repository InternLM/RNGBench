"""Validate a partial single-player run without discarding recorded replies."""

import copy
import io
import json
from pathlib import Path

from common.materialize import materialize
from common.turn_logger import TurnLogger


class ResumeError(ValueError):
    """The saved run cannot be continued without changing its history."""


class PartialReplay:
    """Feed saved replies through the normal loop before allowing live calls.

    The original log and images remain read-only until every saved turn has
    been regenerated and checked. The loop therefore restores its own phase,
    retry count and pending feedback, including an incomplete final round.
    """

    def __init__(self, path, raw, turns):
        self.path = Path(path)
        self.raw = raw
        self.turns = turns
        self.position = 0
        self.activated = False
        self._live_client = None

    @classmethod
    def load(cls, path, config):
        path = Path(path)
        if not path.exists():
            return None
        raw = path.read_bytes()
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise ResumeError("Cannot resume a corrupt game.json; original files preserved") from exc
        if not isinstance(data, dict) or not isinstance(data.get("config"), dict):
            raise ResumeError("Cannot resume a game without a valid config")
        if data.get("result") is not None:
            raise ResumeError("Run is already finalized; use a new output directory or explicit overwrite")
        saved = data["config"]
        # Legacy logs without oracle_memory used the default False.
        saved = dict(saved, oracle_memory=saved.get("oracle_memory", False))
        for key, value in config.items():
            if key not in saved or saved[key] != value:
                raise ResumeError(f"Cannot resume with a different or missing {key}")
        turns = data.get("turns")
        if not isinstance(turns, list) or any(not isinstance(t, dict) for t in turns):
            raise ResumeError("Cannot resume without a valid turns list")
        if len(turns) > config["max_responses"]:
            raise ResumeError("Recorded replies exceed the configured budget")
        return cls(path, raw, turns)

    @property
    def replaying(self):
        return self.position < len(self.turns)

    def activate(self):
        if self.replaying:
            raise ResumeError("Run ended before all saved replies were replayed")
        if not self.activated:
            if self.path.read_bytes() != self.raw:
                raise ResumeError("game.json changed while validating the resume")
            self.activated = True

    def image_store(self, original):
        return _ReplayImageStore(self, original) if original is not None else None

    def client(self, live_factory, image_store):
        replay = self

        class Client:
            def chat(self, messages):
                if replay.replaying:
                    turn = replay.turns[replay.position]
                    expected = turn.get("messages_sent")
                    if not isinstance(expected, list):
                        raise ResumeError("Saved turn has no valid message snapshot")
                    if image_store is not None:
                        expected = materialize(expected, image_store)
                    if messages != expected:
                        raise ResumeError(f"Request differs at saved reply {replay.position + 1}")
                    response = turn.get("response")
                    if not isinstance(response, dict):
                        raise ResumeError("Saved turn has no valid response")
                    return copy.deepcopy(response)
                replay.activate()
                if replay._live_client is None:
                    replay._live_client = live_factory()
                return replay._live_client.chat(messages)

        return Client()

    def logger(self, config):
        replay = self

        class Logger(TurnLogger):
            def _flush(self):
                if replay.activated:
                    super()._flush()

            def log_call(self, **kwargs):
                turn_id = super().log_call(**kwargs)
                if replay.replaying:
                    if self._data["turns"][-1] != replay.turns[replay.position]:
                        raise ResumeError(f"State differs at saved reply {replay.position + 1}")
                    replay.position += 1
                    if not replay.replaying:
                        replay.activate()
                return turn_id

            def finalize(self, result):
                replay.activate()
                super().finalize(result)

        return Logger(self.path, config=config)


class _ReplayImageStore:
    def __init__(self, replay, original):
        self.replay = replay
        self.original = original

    def save_pil(self, img, name):
        if self.replay.activated:
            return self.original.save_pil(img, name=name)
        # Match ImageStore's JPEG encoding without rewriting any saved image.
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=self.original.quality, optimize=True)
        dest = self.original.root / f"{name}.jpg"
        if not dest.is_file() or dest.read_bytes() != buf.getvalue():
            raise ResumeError(f"Saved image is missing or differs: {name}")
        return str(dest.relative_to(self.original.base_dir))

    def load_as_data_url(self, rel_path):
        return self.original.load_as_data_url(rel_path)
