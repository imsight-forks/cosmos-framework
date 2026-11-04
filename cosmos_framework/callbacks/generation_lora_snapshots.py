# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in immutable inference adapters after actual native optimizer updates."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch

from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.generator.lora_export import export_generation_lora


class GenerationLoraSnapshots(Callback):
    """Gather only trainable adapters; leave model, optimizer and RNG untouched.

    Every rank participates in DTensor collectives. Rank zero commits the full
    adapter and then appends its manifest identity to the progress index. This
    does not save resumable optimizer state or launch generation on trainer GPUs.
    """

    def __init__(self, output_dir: str, binding_path: str) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.binding = json.loads(Path(binding_path).read_text())
        self.last_update = 0

    @staticmethod
    def _rank() -> int:
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    @staticmethod
    def _check_rank_zero(error: str | None) -> None:
        errors = [error]
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(errors, src=0)
        if errors[0] is not None:
            raise RuntimeError(errors[0])

    def on_train_start(self, model: torch.nn.Module, iteration: int = 0) -> None:
        del model
        self.last_update = iteration
        error = None
        if self._rank() == 0:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=False)
            except Exception as exc:
                error = f"Cannot create immutable adapter snapshot root: {exc}"
        self._check_rank_zero(error)

    @torch.no_grad()
    def on_training_step_end(
        self,
        model: torch.nn.Module,
        data_batch: dict,
        output_batch: dict,
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del data_batch, output_batch, loss
        if iteration != self.last_update + 1:
            raise ValueError("Adapter snapshots require consecutive optimizer updates")
        started = time.monotonic()
        rank = self._rank()
        state: dict[str, torch.Tensor] = {}
        for name, parameter in sorted(model.named_parameters()):
            if not parameter.requires_grad:
                continue
            value = parameter.detach()
            if hasattr(value, "full_tensor"):
                value = value.full_tensor()
            if rank == 0:
                state[name] = value.cpu().contiguous()
        error = None
        if rank == 0:
            try:
                destination = self.output_dir / f"iter_{iteration:09d}"
                manifest = export_generation_lora(
                    model,
                    destination,
                    base_binding=self.binding["base_binding"],
                    training_view_sha256=self.binding["training_view_sha256"],
                    update=iteration,
                    source_identities=self.binding["source_identities"],
                    materialized_state=state,
                )
                event = {
                    "event": "adapter_snapshot",
                    "update": iteration,
                    "path": str(destination),
                    "manifest_sha256": hashlib.sha256((destination / "training_manifest.json").read_bytes()).hexdigest(),
                    "weights_sha256": manifest["files"]["adapter_model.safetensors"],
                    "trainable_parameter_count": manifest["trainable_parameter_count"],
                    "seconds": time.monotonic() - started,
                }
                with (self.output_dir / "index.jsonl").open("a") as stream:
                    stream.write(json.dumps(event, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception as exc:
                error = f"Adapter snapshot {iteration} failed: {exc}"
        self._check_rank_zero(error)
        self.last_update = iteration
