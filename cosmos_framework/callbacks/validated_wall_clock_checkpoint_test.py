"""CPU qualification of best publication and deterministic validation bookkeeping."""

import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from cosmos_framework.callbacks.validated_wall_clock_checkpoint import (
    ValidatedWallClockCheckpoint,
    preserve_best_checkpoint,
)


class ValidatedCheckpointTest(unittest.TestCase):
    def test_operator_pause_requires_completed_save_and_requests_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / "pause.request"
            request.touch()
            callback = ValidatedWallClockCheckpoint(tmp, tmp, pause_request_path=str(request))
            checkpointer = SimpleNamespace(save=Mock())
            callback.trainer = SimpleNamespace(checkpointer=checkpointer, request_stop_after_checkpoint=Mock())
            with patch.object(callback, "_cp_data_window_active", return_value=False), patch.object(callback, "_verify_complete", return_value=True):
                callback.on_training_step_end(None, {}, {}, None, 23)
            checkpointer.save.assert_called_once()
            callback.trainer.request_stop_after_checkpoint.assert_called_once_with(23)

    def test_resume_reference_validation_does_not_promote_unsaved_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback = ValidatedWallClockCheckpoint(tmp, tmp, validation_samples=1)
            callback.best_score = 0.3
            callback.on_validation_start(None, None, 316)
            callback.on_validation_step_end(None, {}, {}, torch.tensor(0.2), 316)
            callback.on_validation_end(None, 316)
            self.assertEqual(callback.best_score, 0.3)

    def test_best_survives_ring_rotation_and_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "checkpoints/iter_000000001"
            (source / "model").mkdir(parents=True)
            (source / "model/weights").write_bytes(b"first weights")
            preserve_best_checkpoint(source, root / "best", 0.4, 1)
            shutil.rmtree(source)
            self.assertEqual((root / "best/current/model/weights").read_bytes(), b"first weights")
            next_source = root / "checkpoints/iter_000000002"
            (next_source / "model").mkdir(parents=True)
            (next_source / "model/weights").write_bytes(b"second weights")
            with patch("os.link", side_effect=OSError("unsupported")):
                preserve_best_checkpoint(next_source, root / "best", 0.3, 2)
            self.assertFalse((root / "best/iter_000000001").exists())
            self.assertEqual((root / "best/current/model/weights").read_bytes(), b"second weights")
            self.assertEqual(
                json.loads((root / "best/current/best-selection.json").read_text())["validation_loss"], 0.3
            )

    def test_validation_restores_rng_and_rejects_missing_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback = ValidatedWallClockCheckpoint(tmp, tmp, validation_samples=2)
            random.seed(8)
            np.random.seed(8)
            torch.manual_seed(8)
            py = random.getstate()
            np_state = np.random.get_state()
            torch_state = torch.get_rng_state().clone()
            callback.on_validation_start(None, None, iteration=0)
            random.random()
            np.random.rand()
            torch.rand(2)
            callback.on_validation_step_end(None, {}, {}, torch.tensor(0.2))
            with self.assertRaisesRegex(ValueError, "Incomplete held-out"):
                callback.on_validation_end(None)
            self.assertEqual(random.getstate(), py)
            np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
            torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)

    def test_retention_keeps_latest_five(self):
        callback = ValidatedWallClockCheckpoint("/unused", "/unused", keep_last=5)
        callback._wall_clock_iterations = list(range(1, 8))
        with patch.object(callback, "_enqueue_delete") as delete:
            callback._prune()
        self.assertEqual(callback._wall_clock_iterations, [3, 4, 5, 6, 7])
        self.assertEqual([call.args[0] for call in delete.call_args_list], [1, 2])

    def test_worse_validation_preserves_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            callback = ValidatedWallClockCheckpoint(str(root / "checkpoints"), tmp, validation_samples=1)
            callback.trainer = SimpleNamespace(checkpointer=SimpleNamespace(save_dirname=str(root / "checkpoints")))
            source = root / "checkpoints/iter_000000001"
            source.mkdir(parents=True)
            (source / "weight").write_bytes(b"best")
            callback.pending_validation = 1
            callback.on_validation_start(None, None, iteration=1)
            callback.on_validation_step_end(None, {}, {}, torch.tensor(0.2), iteration=1)
            with patch.object(callback, "_verify_complete", return_value=True):
                callback.on_validation_end(None, iteration=1)
            callback.pending_validation = 2
            callback.on_validation_start(None, None, iteration=2)
            callback.on_validation_step_end(None, {}, {}, torch.tensor(0.3), iteration=2)
            callback.on_validation_end(None, iteration=2)
            self.assertEqual((root / "best/current").resolve(), root / "best/iter_000000001")

    def test_completed_checkpoint_requests_native_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            callback = ValidatedWallClockCheckpoint(tmp, tmp)
            callback.world_size = 1
            callback.trainer = SimpleNamespace(
                checkpointer=SimpleNamespace(save_dirname=tmp), request_validation=Mock()
            )
            cursor = root / "iter_000000018/dataloader/rank_0.pkl"
            cursor.parent.mkdir(parents=True)
            cursor.write_bytes(b"cursor")
            with patch.object(ValidatedWallClockCheckpoint.__bases__[0], "on_save_checkpoint_success"):
                callback.on_save_checkpoint_success(18, 1)
            callback.trainer.request_validation.assert_called_once_with(18)
            self.assertEqual(callback.pending_validation, 18)

    def test_real_checkpointer_storage_override_and_validation_request(self):
        from cosmos_framework.checkpoint.base import AbstractCheckpointer
        from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides
        from cosmos_framework.trainer import ImaginaireTrainer
        from cosmos_framework.utils.config import CheckpointConfig, JobConfig

        class Checkpointer(AbstractCheckpointer):
            def save(self, *args, **kwargs):
                pass

            def load(self, *args, **kwargs):
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            job = JobConfig(project="unit", group="test", name="run")
            default = Checkpointer(CheckpointConfig(), job)
            self.assertEqual(default.save_dirname, job.path_local + "/checkpoints")
            explicit = Checkpointer(CheckpointConfig(local_dirname=tmp), job)
            self.assertEqual(explicit.save_dirname, tmp)
            self.assertIn(
                f"checkpoint.local_dirname={tmp}",
                build_hydra_overrides({"job": {"experiment": "fixture"}, "checkpoint": {"local_dirname": tmp}}),
            )
            with self.assertRaises(ValueError):
                Checkpointer(CheckpointConfig(local_dirname="relative"), job)
            trainer = SimpleNamespace(_validation_requested_at=None)
            ImaginaireTrainer.request_validation(trainer, 18)
            self.assertEqual(trainer._validation_requested_at, 18)
            with self.assertRaises(ValueError):
                ImaginaireTrainer.request_validation(trainer, 19)


if __name__ == "__main__":
    unittest.main()
