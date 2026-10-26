# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Six-update native LoRA smoke with an identical pre/post loss objective."""

import copy
import hashlib
import math
import random

import numpy as np
import torch

from cosmos_framework.callbacks.generation_lora_audit import GenerationLoraAudit


def _rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def _restore(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def _rng_identity(state):
    digest = hashlib.sha256()
    digest.update(repr((state[0], state[1][0], state[1][2:])).encode())
    digest.update(state[1][1].tobytes())
    digest.update(state[2].numpy().tobytes())
    for device, cuda_state in enumerate(state[3]):
        digest.update(str(device).encode())
        digest.update(cuda_state.numpy().tobytes())
    return digest.hexdigest()


def _mean_fixed_loss(value):
    local = float(value.detach().item())
    mean = value.detach().double().clone()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(mean)
        mean /= torch.distributed.get_world_size()
    return local, float(mean.item())


def _batch_identity(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((tuple(item.shape), item.dtype)).encode())
            digest.update(item.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(key.encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


class SourceViewTrainingSmoke(GenerationLoraAudit):
    """Observes native training; comparison forwards perform no optimizer updates."""

    def __init__(self, evidence_dir, max_updates=6):
        super().__init__(evidence_dir)
        if type(max_updates) is not int or not 1 <= max_updates <= 6:
            raise ValueError("Native training smoke permits at most six optimizer updates")
        self.max_updates = max_updates
        self.fixed_batch = None
        self.fixed_rng = None
        self.initial_loss = None
        self.modes = set()
        self.updates = 0

    def on_train_start(self, model, iteration=0):
        if (
            iteration != 0
            or self.config.trainer.max_iter > self.max_updates
            or self.config.trainer.grad_accum_iter != 1
        ):
            raise ValueError("Smoke requires a fresh capped run with one microbatch per update")
        super().on_train_start(model, iteration)

    def on_training_step_start(self, model, data_batch, iteration=0):
        if self.fixed_batch is not None:
            return
        self.fixed_batch = copy.deepcopy(data_batch)
        self.fixed_rng = _rng()
        with torch.no_grad():
            _, value = model.training_step(copy.deepcopy(self.fixed_batch), iteration=0)
        self.initial_local_loss, self.initial_loss = _mean_fixed_loss(value)
        _restore(self.fixed_rng)
        if not math.isfinite(self.initial_loss):
            raise ValueError("Nonfinite fixed initial objective")
        self._write(
            {
                "event": "fixed_objective_initial",
                "loss": self.initial_loss,
                "local_loss": self.initial_local_loss,
                "rng_sha256": _rng_identity(self.fixed_rng),
                "reduction": "distributed-mean",
                "batch_sha256": _batch_identity(self.fixed_batch),
                "iteration": 0,
            }
        )

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration=0):
        if self.updates >= self.max_updates:
            raise ValueError("Native smoke optimizer update cap exceeded")
        super().on_before_optimizer_step(model, optimizer, scheduler, grad_scaler, iteration)
        if not any(entry["norm"] is not None and entry["norm"] > 0 for entry in self.gradients.values()):
            raise ValueError("Native smoke has no nonzero adapter gradient")
        self.updates += 1

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        self.modes.update(data_batch.get("conditioning_mode", []))
        roles = data_batch.get("conditioning_stream_roles", [])
        masks = output_batch.get("condition_mask_vision")
        if len(roles) != 1 or not isinstance(masks, list):
            raise ValueError("Native smoke requires one sample with explicit packed vision roles")
        stream_roles = roles[0].split(",")
        if len(masks) != len(stream_roles) or stream_roles[-1] != "target":
            raise ValueError("Native packed stream inventory differs from selected controls")
        if any(not bool(mask.eq(1).all()) for mask in masks[:-1]) or not bool(masks[-1].eq(0).all()):
            raise ValueError("Native smoke controls must be clean and target entirely denoised")
        if not model.config.rectified_flow_training_config.exclude_fully_conditioned_items:
            raise ValueError("Native smoke must exclude clean controls from denoising loss")
        x0, xt = output_batch["x0"], output_batch["xt"]
        if len(x0) != len(stream_roles) or len(xt) != len(stream_roles):
            raise ValueError("Native latent inventory differs from selected streams")
        if any(
            not torch.equal(clean.to(device=noised.device, dtype=noised.dtype), noised)
            for clean, noised in zip(x0[:-1], xt[:-1], strict=True)
        ):
            raise ValueError("Native conditioning latents were noised")
        self._write(
            {
                "event": "packed_stream_roles",
                "iteration": iteration,
                "target_only_loss": True,
                "control_latents_unchanged": True,
                "control_comparison": "native-input-dtype",
                "masks": [
                    {"role": role, "clean_frames": int(mask.sum().item()), "frames": mask.numel()}
                    for role, mask in zip(stream_roles, masks, strict=True)
                ],
            }
        )
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)
        self._write(
            {
                "event": "conditioning",
                "iteration": iteration,
                "modes": data_batch.get("conditioning_mode", []),
                "roles": data_batch.get("conditioning_stream_roles", []),
                "exact_rates": data_batch.get("source_effective_rate", []),
                "executions": data_batch.get("conditioning_execution_fingerprint", []),
            }
        )

    def on_train_end(self, model, iteration=0):
        super().on_train_end(model, iteration)
        if self.fixed_batch is None or self.fixed_rng is None:
            raise ValueError("Native smoke did not evaluate a fixed objective")
        saved_rng = _rng()
        _restore(self.fixed_rng)
        if _rng_identity(_rng()) != _rng_identity(self.fixed_rng):
            raise ValueError("Fixed-objective RNG restoration differs")
        with torch.no_grad():
            _, value = model.training_step(copy.deepcopy(self.fixed_batch), iteration=0)
        final_local_loss, final_loss = _mean_fixed_loss(value)
        _restore(saved_rng)
        if torch.distributed.is_initialized():
            peer_modes = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(peer_modes, sorted(self.modes))
            self.modes = {mode for peer in peer_modes for mode in peer}
        succeeded = (
            math.isfinite(final_loss)
            and final_loss < self.initial_loss
            and self.modes == {"seg", "depth", "depth-seg"}
            and 1 <= self.updates <= self.max_updates
        )
        self._write(
            {
                "event": "smoke_complete",
                "success": succeeded,
                "fixed_initial_loss": self.initial_loss,
                "fixed_final_loss": final_loss,
                "local_initial_loss": self.initial_local_loss,
                "local_final_loss": final_local_loss,
                "rng_sha256": _rng_identity(self.fixed_rng),
                "comparison_iteration": 0,
                "reduction": "distributed-mean",
                "batch_sha256": _batch_identity(self.fixed_batch),
                "updates": self.updates,
                "max_updates": self.max_updates,
                "modes": sorted(self.modes),
            }
        )
        if not succeeded:
            raise ValueError(
                "Native six-update smoke failed mode coverage or fixed-objective loss decrease; retain evidence without extending the budget"
            )
