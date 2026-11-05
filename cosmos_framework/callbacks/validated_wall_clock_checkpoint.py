# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native timed checkpoint rotation with fixed-RNG held-out best selection."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.callbacks.wall_clock_checkpoint import WallClockCheckpoint


def preserve_best_checkpoint(source: Path, root: Path, score: float, iteration: int) -> None:
    """Publish a separate best snapshot before unlinking the preceding best.

    Hardlinks preserve checkpoint bytes after rotation without another full copy.
    A filesystem without hardlinks falls back to copying. The atomic pointer is
    changed only after all files and the score record have been committed.
    """
    if not math.isfinite(score):
        raise ValueError("Best checkpoint score must be finite")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"iter_{iteration:09d}"
    temporary = root / f".iter_{iteration:09d}.pending"
    old = (root / "current").resolve() if (root / "current").is_symlink() else None

    def link_or_copy(src, dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        return dst

    shutil.copytree(source, temporary, copy_function=link_or_copy)
    # Never overwrite a hardlinked score file from the source checkpoint.
    metadata = temporary / "best-selection.json"
    metadata.write_text(json.dumps({"iteration": iteration, "validation_loss": score}) + "\n")
    temporary.rename(destination)
    pointer = root / ".current.pending"
    pointer.symlink_to(destination.name, target_is_directory=True)
    pointer.replace(root / "current")
    if old is not None and old != destination:
        if old.parent != root.resolve():
            raise ValueError("Previous best points outside this run")
        shutil.rmtree(old)


class ValidatedWallClockCheckpoint(WallClockCheckpoint):
    """Reuse the native timer, writer and retention; validate each committed save.

    Native validation runs after callback outputs are released, selected by a
    runtime request for the just-saved iteration. Configuration remains frozen.
    This requires synchronous local-filesystem DCP.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        evidence_dir: str,
        validation_samples: int = 32,
        interval_minutes: float = 30,
        keep_last: int = 5,
        pause_request_path: str | None = None,
    ) -> None:
        super().__init__(
            interval_minutes=interval_minutes, keep_last=keep_last, check_every_n=1, assumed_write_minutes=0
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.evidence_dir = Path(evidence_dir)
        self.validation_samples = validation_samples
        self.best_score = float("inf")
        self.pending_validation = None
        self.validation_sum = 0.0
        self.validation_count = 0
        self.validation_rng = None
        self.pause_request_path = Path(pause_request_path) if pause_request_path else None
        self.paused_at = None

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        requested = [bool(self.pause_request_path and self.pause_request_path.exists()) if self.rank() == 0 else False]
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(requested, src=0)
        if not requested[0]:
            return super().on_training_step_end(model, data_batch, output_batch, loss, iteration)
        if self._cp_data_window_active():
            return
        if self.pending_validation != iteration:
            self._triggered_iterations.add(iteration)
            self.trainer.checkpointer.save(
                model, self._optimizer, self._scheduler, self._grad_scaler, iteration=iteration
            )
        if not self._verify_complete(iteration):
            raise RuntimeError("Operator pause requires a complete checkpoint")
        self.paused_at = iteration
        self.trainer.request_stop_after_checkpoint(iteration)

    def on_train_end(self, model, iteration=0):
        if self.paused_at is not None:
            self.event({"event": "paused", "iteration": iteration, "checkpoint": self._checkpoint_dirname(iteration)})

    @staticmethod
    def rank() -> int:
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def event(self, record: dict) -> None:
        if self.rank() == 0:
            with (self.evidence_dir / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def on_train_start(self, model, iteration=0):
        if self.config.checkpoint.dcp_async_mode_enabled:
            raise ValueError("Validated timed checkpoints require synchronous DCP")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        if Path(self.trainer.checkpointer.save_dirname).resolve() != self.checkpoint_dir.resolve():
            raise ValueError("Configure checkpoint.local_dirname to the admitted NFS destination")
        previous_best = self.checkpoint_dir.parent / "best/current/best-selection.json"
        if previous_best.is_file():
            self.best_score = float(json.loads(previous_best.read_text())["validation_loss"])
        super().on_train_start(model, iteration)
        self.event(
            {
                "event": "checkpoint_policy",
                "minutes": self._interval_seconds / 60,
                "keep_last": self._keep_last,
                "validation_samples": self.validation_samples,
                "best_metric": "fixed-held-out-native-flow-loss",
                "checkpoint_dir": str(self.checkpoint_dir),
            }
        )

    def on_save_checkpoint_success(self, iteration=0, elapsed_time=0):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        # The sampler/conditioning cursor is per-rank pickle state outside DCP
        # shard metadata; every rank must finish it before retention advances.
        if not all(
            (Path(self._checkpoint_dirname(iteration)) / f"dataloader/rank_{rank}.pkl").is_file()
            for rank in range(self.world_size)
        ):
            raise RuntimeError("Completed checkpoint is missing per-rank dataloader state")
        # Include the terminal save in the same five-item rotating window.
        self._triggered_iterations.add(iteration)
        super().on_save_checkpoint_success(iteration, elapsed_time)
        self.pending_validation = iteration
        self.trainer.request_validation(iteration)
        self.event(
            {
                "event": "checkpoint_complete",
                "iteration": iteration,
                "elapsed_seconds": elapsed_time,
                "path": self._checkpoint_dirname(iteration),
            }
        )

    def _verify_complete(self, iteration):
        return super()._verify_complete(iteration) and all(
            (Path(self._checkpoint_dirname(iteration)) / f"dataloader/rank_{rank}.pkl").is_file()
            for rank in range(self.world_size)
        )

    def _milestone_lands_mid_write(self, iteration):
        # Synchronous writes finish before the next optimizer update; there is
        # no overlapping write to protect by deferring an otherwise due save.
        return False

    def on_validation_start(self, model, dataloader, iteration=0):
        self.validation_sum, self.validation_count = 0.0, 0
        self.validation_python_rng = random.getstate()
        self.validation_numpy_rng = np.random.get_state()
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        self.validation_rng = torch.random.fork_rng(devices=devices)
        self.validation_rng.__enter__()
        seed = 20261105 + self.rank()
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)

    def on_validation_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise ValueError("Nonfinite held-out validation loss")
        self.validation_sum += value
        self.validation_count += 1

    def on_validation_end(self, model, iteration=0):
        if self.validation_rng is not None:
            self.validation_rng.__exit__(None, None, None)
            self.validation_rng = None
        random.setstate(self.validation_python_rng)
        np.random.set_state(self.validation_numpy_rng)
        distributed = torch.distributed.is_initialized()
        values = torch.tensor(
            [self.validation_sum, self.validation_count], dtype=torch.float64, device="cuda" if distributed else "cpu"
        )
        if distributed:
            torch.distributed.all_reduce(values)
        total, count = values.tolist()
        if count != self.validation_samples or not math.isfinite(total):
            raise ValueError(f"Incomplete held-out evaluation: {count} / {self.validation_samples}")
        score = total / count
        improved = self.pending_validation == iteration and iteration > 0 and score < self.best_score
        error = [None]
        if improved and self.rank() == 0:
            try:
                if self.pending_validation != iteration or not self._verify_complete(iteration):
                    raise ValueError("Best selection requires the matching completed checkpoint")
                preserve_best_checkpoint(
                    Path(self._checkpoint_dirname(iteration)), self.checkpoint_dir.parent / "best", score, iteration
                )
            except Exception as exc:
                error[0] = str(exc)
        if distributed:
            torch.distributed.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(f"Best checkpoint publication failed: {error[0]}")
        if improved:
            self.best_score = score
        self.event(
            {
                "event": "validation",
                "iteration": iteration,
                "loss": score,
                "samples": int(count),
                "improved": improved,
                "best_loss": None if not math.isfinite(self.best_score) else self.best_score,
            }
        )
        self.pending_validation = None
