# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in per-rank evidence for native generation-attention LoRA qualification."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.generator.lora import LoraInjectedLinear
from cosmos_framework.utils.misc import get_local_tensor_if_DTensor

_ADAPTER_NAME = re.compile(
    r"^net\.language_model\.model\.layers\.\d+\.(?:_checkpoint_wrapped_module\.)?self_attn\."
    r"(?:q_proj_moe_gen|k_proj_moe_gen|v_proj_moe_gen|o_proj_moe_gen)\.lora_[AB]\.weight$"
)


def _fingerprint(parameter: torch.Tensor) -> str:
    value = get_local_tensor_if_DTensor(parameter.detach()).contiguous().cpu()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _auxiliary_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    # Native tokenizer interfaces are plain objects: their .model.model modules
    # are deliberately absent from ImaginaireModel.named_parameters(). Include
    # the actual loaded Wan/audio weights in the frozen-state proof explicitly.
    registered = {id(parameter) for parameter in model.parameters()}
    result = {}
    for attribute in ("tokenizer_vision_gen", "tokenizer_sound_gen"):
        interface = getattr(model, attribute, None)
        network = getattr(getattr(interface, "model", None), "model", None)
        if not isinstance(network, torch.nn.Module):
            continue
        for name, parameter in network.named_parameters():
            if parameter.requires_grad:
                raise ValueError(f"Auxiliary tokenizer parameter is trainable: {attribute}.{name}")
            if id(parameter) not in registered:
                result[f"{attribute}.model.model.{name}"] = parameter
    return result


class GenerationLoraAudit(Callback):
    """Assert optimizer scope and retain complete local frozen-weight fingerprints.

    This inspects the upstream training loop; it never computes a substitute
    objective. DTensor fingerprints describe each rank's local shard. Compare
    every rank for a distributed qualification, retaining topology alongside it.
    Use a fresh evidence directory per process attempt, including resume.
    """

    def __init__(self, evidence_dir: str) -> None:
        super().__init__()
        self.evidence_dir = Path(evidence_dir)
        self.initial_frozen: dict[str, str] = {}
        self.initial_adapters: dict[str, str] = {}
        self.initial_auxiliary: dict[str, str] = {}
        self.gradients: dict[str, dict[str, float | None]] = {}
        self.started = 0.0
        self.path: Path | None = None

    def _write(self, event: dict[str, Any]) -> None:
        assert self.path is not None
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")

    def on_train_start(self, model: torch.nn.Module, iteration: int = 0) -> None:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.evidence_dir / f"rank-{rank}.jsonl"
        with self.path.open("x", encoding="utf-8"):
            pass
        self.started = time.monotonic()
        trainable = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if not _ADAPTER_NAME.fullmatch(name):
                    raise ValueError(f"Unexpected trainable parameter: {name}")
                self.initial_adapters[name] = _fingerprint(parameter)
                trainable.append(
                    {
                        "name": name,
                        "shape": list(parameter.shape),
                        "local_shape": list(get_local_tensor_if_DTensor(parameter).shape),
                        "dtype": str(parameter.dtype),
                        "count": parameter.numel(),
                    }
                )
            else:
                self.initial_frozen[name] = _fingerprint(parameter)
        if not trainable:
            raise ValueError("No generation-attention adapters are trainable")
        self.initial_auxiliary = {
            name: _fingerprint(parameter) for name, parameter in _auxiliary_parameters(model).items()
        }
        modules = {
            name: {"rank": module._lora_rank, "alpha": module._lora_alpha, "scale": module._lora_scale}
            for name, module in model.named_modules()
            if isinstance(module, LoraInjectedLinear)
        }
        self._write(
            {
                "event": "start",
                "iteration": iteration,
                "rank": rank,
                "trainable": trainable,
                "frozen": self.initial_frozen,
                "adapters": self.initial_adapters,
                "auxiliary_frozen": self.initial_auxiliary,
                "trainable_count": sum(entry["count"] for entry in trainable),
                "adapter_modules": modules,
            }
        )

    def on_before_optimizer_step(
        self,
        model: torch.nn.Module,
        optimizer: Any,
        scheduler: Any,
        grad_scaler: Any,
        iteration: int = 0,
    ) -> None:
        del scheduler, grad_scaler
        names = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
        optimizers = getattr(optimizer, "optimizers", (optimizer,))
        groups = [group for child in optimizers for group in child.param_groups]
        grouped = [parameter for group in groups for parameter in group["params"]]
        if len(grouped) != len(names) or {id(parameter) for parameter in grouped} != set(names):
            raise ValueError("Optimizer groups do not exactly match generation adapters")
        gradients = {}
        for parameter in grouped:
            grad = parameter.grad
            norm = None if grad is None else float(get_local_tensor_if_DTensor(grad.detach()).float().norm().item())
            if norm is not None and not math.isfinite(norm):
                raise ValueError(f"Nonfinite adapter gradient: {names[id(parameter)]}")
            gradients[names[id(parameter)]] = {"norm": norm}
        self.gradients = gradients
        self._write(
            {
                "event": "before_update",
                "iteration": iteration,
                "gradients": gradients,
                "optimizer_groups": [
                    {
                        "lr": group["lr"],
                        "weight_decay": group.get("weight_decay", 0),
                        "parameters": [names[id(parameter)] for parameter in group["params"]],
                    }
                    for group in groups
                ],
            }
        )

    def on_after_backward(self, model: torch.nn.Module, iteration: int = 0) -> None:
        # Inspect before native clipping/force-finite callbacks can replace an
        # invalid gradient with zero. Missing gradients remain explicit evidence.
        gradients = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            grad = parameter.grad
            norm = None if grad is None else float(get_local_tensor_if_DTensor(grad.detach()).float().norm().item())
            if norm is not None and not math.isfinite(norm):
                raise ValueError(f"Nonfinite pre-clipping adapter gradient: {name}")
            gradients[name] = {"norm": norm}
        self._write({"event": "after_backward", "iteration": iteration, "gradients": gradients})

    def on_training_step_end(
        self,
        model: torch.nn.Module,
        data_batch: dict,
        output_batch: dict,
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, output_batch
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise ValueError("Nonfinite native training loss")
        memory = {}
        if torch.cuda.is_available():
            memory = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
            }
        cursors = {
            key: data_batch[key].tolist()
            for key in ["sample_worker_id", "sample_epoch", "sample_index"]
            if key in data_batch
        }
        self._write(
            {
                "event": "update",
                "iteration": iteration,
                "loss": value,
                "elapsed_seconds": time.monotonic() - self.started,
                "memory": memory,
                "cursors": cursors,
            }
        )

    def on_train_end(self, model: torch.nn.Module, iteration: int = 0) -> None:
        frozen = {}
        adapters = {}
        for name, parameter in model.named_parameters():
            (adapters if parameter.requires_grad else frozen)[name] = _fingerprint(parameter)
        changed_frozen = sorted(name for name in self.initial_frozen if frozen.get(name) != self.initial_frozen[name])
        changed_adapters = sorted(
            name for name in self.initial_adapters if adapters.get(name) != self.initial_adapters[name]
        )
        auxiliary = {name: _fingerprint(parameter) for name, parameter in _auxiliary_parameters(model).items()}
        changed_auxiliary = sorted(
            name for name in self.initial_auxiliary if auxiliary.get(name) != self.initial_auxiliary[name]
        )
        self._write(
            {
                "event": "end",
                "iteration": iteration,
                "frozen": frozen,
                "adapters": adapters,
                "changed_frozen": changed_frozen,
                "changed_adapters": changed_adapters,
                "auxiliary_frozen": auxiliary,
                "changed_auxiliary": changed_auxiliary,
                "elapsed_seconds": time.monotonic() - self.started,
            }
        )
        if (
            changed_frozen
            or set(frozen) != set(self.initial_frozen)
            or changed_auxiliary
            or set(auxiliary) != set(self.initial_auxiliary)
        ):
            raise ValueError("Frozen parameters changed during native LoRA training")
        if not changed_adapters:
            raise ValueError("No adapter weights changed during native training")
