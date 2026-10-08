"""Public-entry regression tests for exact, fail-closed single-game resume.

All model replies are scripted. Temporary fixtures are disposable; no model
client or network connection is created.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

GAME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GAME_ROOT.parent
for location in (REPO_ROOT, GAME_ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from common.run_dir import prepare_run_dir
from modes import single_normal


class ScriptedClient:
    def __init__(self, replies):
        self.replies = iter(copy.deepcopy(replies))
        self.requests = []

    def chat(self, messages):
        self.requests.append(copy.deepcopy(messages))
        try:
            return next(self.replies)
        except StopIteration as exc:
            raise AssertionError("Unexpected additional model request") from exc


def file_snapshot(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*") if path.is_file()
    }


class SingleResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rngbench-single-resume-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.kwargs = {
            "model": "offline-resume-fixture",
            "label": "Offline resume fixture",
            "rows": 2,
            "cols": 4,
            "num_cards": 4,
            "seed": 0,
            "render": "image",
            "theme": "poker",
            "assets_dir": str(GAME_ROOT / "assets"),
            "cell_size": 64,
            "max_responses": 32,
            "max_retries": 2,
            "cot_enabled": True,
            "oracle_memory": False,
        }
        # Independent layout construction is only used to choose a known test
        # sequence. The tested runner still performs all actual state changes.
        deck = [face for face in range(4) for _ in range(2)]
        random.Random(0).shuffle(deck)
        pairs = [
            [chr(97 + i // 4) + chr(65 + i % 4)
             for i, value in enumerate(deck) if value == face]
            for face in range(4)
        ]
        self.pairs = pairs
        a, b, c, d = pairs
        action = lambda coord: f"Thought: scripted fixture\nAction: {coord}"
        contents = [
            "Missing action", action(a[0]), action(a[0]), action(b[0]),
            "Missing action", "Missing action", "Missing action",
            action(c[0]), action(c[0]), action(c[0]), action(c[0]),
            action(a[0]), action(a[1]),
            action(a[0]), action(b[0]), action(b[1]),
            action(c[0]), action(c[1]), action(d[0]), action(d[1]),
        ]
        self.replies = [
            {"content": content, "reasoning": f"PRIVATE_FIXTURE_REASONING_{i}"}
            for i, content in enumerate(contents, 1)
        ]
        self.baseline_dir = self.root / "baseline"
        self.baseline_result, self.baseline_client = self.run_game(
            self.baseline_dir, self.replies
        )
        self.baseline = json.loads((self.baseline_dir / "game.json").read_text())
        self.assertIsNone(self.baseline_result["error"])
        self.assertTrue(self.baseline_result["done"])
        self.assertEqual(self.baseline_result["score"], 4)
        self.assertEqual(len(self.baseline["turns"]), 20)

    def run_game(self, directory, replies, **overrides):
        client = ScriptedClient(replies)
        options = dict(self.kwargs, **overrides)
        with patch.object(single_normal, "make_client", return_value=client):
            result = single_normal.run_one_game(out_dir=directory, **options)
        return result, client

    def make_partial(self, cut, name=None, baseline=None, source_dir=None):
        baseline = baseline or self.baseline
        source_dir = source_dir or self.baseline_dir
        directory = self.root / (name or f"partial-{cut}")
        directory.mkdir()
        partial = copy.deepcopy(baseline)
        partial["turns"] = partial["turns"][:cut]
        partial["result"] = None

        # Retain only images that existed by the last logged reply. A successful
        # second flip writes its both-card image before logging the reply, even
        # though no later request yet references it.
        images = {"images/ground_truth.jpg"}
        for turn in partial["turns"]:
            for message in turn["messages_sent"]:
                if not isinstance(message.get("content"), list):
                    continue
                for part in message["content"]:
                    if part.get("type") == "image_url":
                        images.add(part["image_url"]["url"])
            outcome = turn["env_result"]
            if outcome.get("applied") and outcome.get("phase_after") == "flip_first":
                images.add(f"images/round_{turn['round']:03d}_both.jpg")
        for relative in images:
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source_dir / relative).read_bytes())
        (directory / "game.json").write_text(json.dumps(partial, ensure_ascii=False))
        return directory

    def assert_resume_equivalent(self, cut):
        directory = self.make_partial(cut)
        prior_images = {
            name: data for name, data in file_snapshot(directory).items()
            if name.endswith(".jpg")
        }
        result, client = self.run_game(directory, self.replies[cut:])
        actual = json.loads((directory / "game.json").read_text())
        self.assertEqual(result, self.baseline_result)
        self.assertEqual(actual, self.baseline)
        self.assertEqual(client.requests, self.baseline_client.requests[cut:])
        after = file_snapshot(directory)
        for name, image_bytes in prior_images.items():
            self.assertEqual(after[name], image_bytes, name)

    def assert_rejected_unchanged(self, directory, **overrides):
        before = file_snapshot(directory)
        client = ScriptedClient(self.replies)
        with patch.object(single_normal, "make_client", return_value=client):
            with self.assertRaises(Exception):
                single_normal.run_one_game(
                    out_dir=directory, **dict(self.kwargs, **overrides)
                )
        self.assertEqual(client.requests, [], "Resume rejection made a model request")
        self.assertEqual(file_snapshot(directory), before, "Resume rejection changed evidence")

    def test_resume_at_every_reply_boundary_matches_uninterrupted_game(self):
        for cut in range(len(self.replies) + 1):
            with self.subTest(cut=cut):
                self.assert_resume_equivalent(cut)

    def test_orphan_first_flip_does_not_get_replayed_as_a_new_request(self):
        # Reply 2 succeeded after one parse retry, leaving a revealed first card.
        self.assert_resume_equivalent(2)

    def test_retry_and_forfeit_boundaries_preserve_remaining_attempts(self):
        # Both phases have failed/retried requests, and cuts 7/11 have exhausted
        # all three attempts and are about to append their forfeit messages.
        for cut in (1, 3, 5, 6, 7, 8, 9, 10, 11, 14):
            with self.subTest(cut=cut):
                self.assert_resume_equivalent(cut)

    def test_response_cap_already_spent_only_finalizes(self):
        for cap in (2, 4, 7):
            with self.subTest(cap=cap):
                source = self.root / f"cap-baseline-{cap}"
                expected, _ = self.run_game(source, self.replies, max_responses=cap)
                baseline = json.loads((source / "game.json").read_text())
                directory = self.make_partial(
                    cap, name=f"cap-partial-{cap}", baseline=baseline, source_dir=source
                )
                result, client = self.run_game(directory, [], max_responses=cap)
                self.assertEqual(client.requests, [])
                self.assertEqual(result, expected)
                self.assertEqual(json.loads((directory / "game.json").read_text()), baseline)

    def test_board_already_cleared_only_finalizes(self):
        directory = self.make_partial(20)
        result, client = self.run_game(directory, [])
        self.assertEqual(client.requests, [])
        self.assertEqual(result, self.baseline_result)
        self.assertEqual(json.loads((directory / "game.json").read_text()), self.baseline)

    def test_incompatible_config_is_rejected_before_evidence_changes(self):
        variants = [
            {"seed": 1}, {"max_responses": 33}, {"max_responses": 1},
            {"theme": "noise"}, {"render": "text"}, {"cell_size": 32},
            {"model": "another-offline-model"}, {"max_retries": 1},
            {"cot_enabled": False}, {"oracle_memory": True},
            {"rows": 4, "cols": 2}, {"num_cards": 2},
        ]
        for index, options in enumerate(variants):
            with self.subTest(options=options):
                directory = self.make_partial(4, name=f"config-{index}")
                self.assert_rejected_unchanged(directory, **options)

    def test_corrupt_turn_state_or_history_is_rejected_unchanged(self):
        mutations = {
            "turn_id": lambda d: d["turns"][1].update(turn_id=50),
            "round": lambda d: d["turns"][1].update(round=3),
            "face": lambda d: d["turns"][1]["env_result"].update(revealed_face="wrong"),
            "phase": lambda d: d["turns"][1]["env_result"].update(phase_after="flip_first"),
            "applied": lambda d: d["turns"][1]["env_result"].update(applied=False),
            "retry_of": lambda d: d["turns"][2].update(retry_of=99),
            "call_type": lambda d: d["turns"][2].update(call_type="flip_first"),
            "parse": lambda d: d["turns"][1]["parse"].update(coord=self.pairs[3][0]),
            "response": lambda d: d["turns"][1]["response"].update(content="Action: " + self.pairs[3][0]),
            "history": lambda d: d["turns"][1]["messages_sent"][0].update(content="Changed system prompt"),
        }
        for name, mutate in mutations.items():
            with self.subTest(corruption=name):
                directory = self.make_partial(4, name=f"corrupt-{name}")
                path = directory / "game.json"
                doc = json.loads(path.read_text())
                original = copy.deepcopy(doc)
                mutate(doc)
                self.assertNotEqual(doc, original, "Corruption fixture made no change")
                path.write_text(json.dumps(doc, ensure_ascii=False))
                self.assert_rejected_unchanged(directory)

    def test_missing_or_changed_historical_image_is_rejected_unchanged(self):
        for change in ("missing", "changed"):
            with self.subTest(change=change):
                directory = self.make_partial(2, name=f"image-{change}")
                path = directory / "images/round_001_start.jpg"
                if change == "missing":
                    path.unlink()
                else:
                    with Image.open(path) as original:
                        modified = original.convert("RGB")
                    modified.putpixel((0, 0), (255, 0, 255))
                    modified.save(path, format="JPEG", quality=85)
                self.assert_rejected_unchanged(directory)

    def test_malformed_json_is_rejected_unchanged(self):
        directory = self.make_partial(2)
        (directory / "game.json").write_text('{"config":')
        self.assert_rejected_unchanged(directory)

    def test_resume_directory_policy_preserves_corrupt_json(self):
        directory = self.root / "corrupt-directory-policy"
        directory.mkdir()
        (directory / "game.json").write_text('{"config":')
        (directory / "sentinel.txt").write_text("must remain intact")
        before = file_snapshot(directory)
        with self.assertRaises(Exception):
            prepare_run_dir(directory, "resume")
        self.assertEqual(file_snapshot(directory), before)


if __name__ == "__main__":
    unittest.main()
