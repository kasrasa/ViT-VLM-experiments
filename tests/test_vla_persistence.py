"""Behavioral checks for saved splits, metrics, and logging."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from vit_vlm_experiments.vla import data, metrics
from vit_vlm_experiments.vla.logging_config import configure_logging


class SplitPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "split.json"
        self.config = {
            "file": str(self.path),
            "strategy": "episode_stratified_episode",
            "train_fraction": 0.5,
            "seed": 42,
        }
        self.cache = {"episode_ids": np.array([0, 0, 1, 1, 2, 2, 3, 3])}

    def test_round_trip_reuses_saved_episode_split(self):
        first = data.create_or_load_split(self.cache, self.config)
        payload = json.loads(self.path.read_text())
        self.assertEqual(payload["settings"]["seed"], 42)
        with patch.object(data, "write_split_file", side_effect=AssertionError("rewrote split")):
            second = data.create_or_load_split(self.cache, self.config)
        np.testing.assert_array_equal(first["train_indices"], second["train_indices"])

    def test_legacy_flat_split_is_reused(self):
        first = data.create_or_load_split(self.cache, self.config)
        saved = json.loads(self.path.read_text())
        legacy = {**saved["settings"], "train_episodes": saved["train_episodes"],
                  "val_episodes": saved["val_episodes"]}
        self.path.write_text(json.dumps(legacy))
        with patch.object(data, "write_split_file", side_effect=AssertionError("rewrote split")):
            second = data.create_or_load_split(self.cache, self.config)
        np.testing.assert_array_equal(first["val_episodes"], second["val_episodes"])

    def test_empty_or_invalid_split_is_regenerated_after_three_reads(self):
        real_read_text = Path.read_text
        for contents in ("", "{}", '{"train_episodes": "bad", "val_episodes": []}'):
            with self.subTest(contents=contents):
                self.path.write_text(contents)
                with patch.object(data.time, "sleep"), patch.object(
                    Path, "read_text", autospec=True, side_effect=real_read_text,
                ) as reader:
                    result = data.create_or_load_split(self.cache, self.config)
                    self.assertEqual(reader.call_count, 3)
                self.assertEqual(len(result["train_episodes"]), 2)
                self.assertEqual(len(json.loads(self.path.read_text())["val_episodes"]), 2)

    def test_persistent_read_error_stops_run(self):
        self.path.write_text("{}")
        with patch.object(data.time, "sleep"), patch.object(
            Path, "read_text", side_effect=PermissionError("denied")
        ) as reader:
            with self.assertRaises(data.SplitFileReadError):
                data.create_or_load_split(self.cache, self.config)
        self.assertEqual(reader.call_count, 3)

    def test_transient_read_error_can_reuse_split(self):
        first = data.create_or_load_split(self.cache, self.config)
        original = Path.read_text
        attempts = 0

        def intermittent_read(path, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("temporary failure")
            return original(path, **kwargs)

        with patch.object(data.time, "sleep"), patch.object(
            Path, "read_text", autospec=True, side_effect=intermittent_read
        ), patch.object(data, "write_split_file", side_effect=AssertionError("rewrote split")):
            second = data.create_or_load_split(self.cache, self.config)
        self.assertEqual(attempts, 2)
        np.testing.assert_array_equal(first["train_indices"], second["train_indices"])

    def test_write_error_retries_and_preserves_previous_file(self):
        self.path.write_text("old contents")
        with patch.object(data.time, "sleep"), patch.object(
            data, "atomic_write_text", side_effect=PermissionError("denied")
        ) as writer:
            with self.assertRaises(OSError):
                data.create_or_load_split(self.cache, {**self.config, "reuse_existing": False})
        self.assertEqual(writer.call_count, 3)
        self.assertEqual(self.path.read_text(), "old contents")

    def test_atomic_replace_failure_keeps_previous_file_and_cleans_temporary(self):
        self.path.write_text("old contents")
        with patch.object(Path, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                data.atomic_write_text(self.path, "new contents")
        self.assertEqual(self.path.read_text(), "old contents")
        self.assertEqual(sorted(self.path.parent.iterdir()), [self.path])


class MetricsAndLoggingTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "metrics.json"

    def test_metrics_include_provenance(self):
        self.assertTrue(metrics.save_metrics({"mae": 0.1}, self.path, {"dataset": "demo"}))
        self.assertEqual(json.loads(self.path.read_text()), {
            "mae": 0.1, "metadata": {"dataset": "demo"}
        })

    def test_metrics_write_failure_does_not_replace_prior_result(self):
        self.path.write_text("previous")
        with patch.object(metrics, "atomic_write_text", side_effect=OSError("disk full")):
            self.assertFalse(metrics.save_metrics({"mae": 0.1}, self.path))
        self.assertEqual(self.path.read_text(), "previous")

    def test_reconfiguring_logger_does_not_duplicate_handlers(self):
        settings = {"output_dir": self.directory.name}
        logger = configure_logging(settings)
        try:
            self.assertEqual(len(logger.handlers), 2)
            self.assertIs(logger, configure_logging(settings))
            self.assertEqual(len(logger.handlers), 2)
            logger.info("logging check")
            self.assertIn("logging check", (Path(self.directory.name) / "run.log").read_text())
        finally:
            for handler in list(logger.handlers):
                if getattr(handler, "_vla_managed", False):
                    logger.removeHandler(handler)
                    handler.close()


if __name__ == "__main__":
    unittest.main()
