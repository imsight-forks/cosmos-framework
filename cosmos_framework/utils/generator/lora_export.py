# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Portable generation-attention LoRA bridge to Diffusers Cosmos3.

This module belongs to the maintained framework fork, independent of the
originating project's distribution. Resumable DCP checkpoints remain owned by
the native trainer; these artifacts contain only inference adapter weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from cosmos_framework.utils.generator.lora import LoraInjectedLinear

GENERATION_PROJECTIONS = {
    "q_proj_moe_gen": "add_q_proj",
    "k_proj_moe_gen": "add_k_proj",
    "v_proj_moe_gen": "add_v_proj",
    "o_proj_moe_gen": "to_add_out",
}
_NATIVE_MODULE = re.compile(
    r"^(?:(?:model\.)?net\.)?(?:language_model\.)?model\.layers\.(\d+)"
    r"\.(?:_checkpoint_wrapped_module\.)?self_attn\.(q_proj_moe_gen|k_proj_moe_gen|v_proj_moe_gen|o_proj_moe_gen)$"
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value: str, name: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _binding(value: Mapping[str, str]) -> dict[str, str]:
    result = dict(value)
    if set(result) != {"repository", "revision", "sha256"}:
        raise ValueError("base_binding requires repository, revision, and sha256")
    if not all(isinstance(item, str) and item for item in result.values()):
        raise ValueError("base_binding values must be nonempty strings")
    _digest(result["sha256"], "base_binding.sha256")
    return result


def export_generation_lora(
    network: torch.nn.Module,
    destination: str | Path,
    *,
    base_binding: Mapping[str, str],
    training_view_sha256: str,
    update: int,
    source_identities: Mapping[str, str],
) -> dict[str, Any]:
    """Publish an immutable, complete attention-only adapter directory.

    Call on an unsharded/materialized model (or a gathered CPU replica), never
    rank-local DTensors. The base digest binds the caller's canonical checkpoint
    manifest; callers must verify that manifest's payloads before loading base.
    """
    binding = _binding(base_binding)
    _digest(training_view_sha256, "training_view_sha256")
    if isinstance(update, bool) or not isinstance(update, int) or update < 1:
        raise ValueError("update must be a positive optimizer update count")
    if not source_identities or not all(
        isinstance(key, str) and key and isinstance(value, str) and value for key, value in source_identities.items()
    ):
        raise ValueError("source_identities must record source and dependency identities")

    weights: dict[str, torch.Tensor] = {}
    mappings = []
    adapter_parameters: set[int] = set()
    scales: set[tuple[int, float]] = set()
    layers: dict[int, set[str]] = {}
    for name, module in network.named_modules():
        if not isinstance(module, LoraInjectedLinear):
            continue
        match = _NATIVE_MODULE.fullmatch(name)
        if match is None:
            raise ValueError(f"Adapter is outside generation-attention allowlist: {name}")
        layer, projection = match.groups()
        target = f"layers.{layer}.self_attn.{GENERATION_PROJECTIONS[projection]}"
        rank, alpha = module._lora_rank, module._lora_alpha
        if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"Invalid adapter rank/alpha at {name}")
        scales.add((rank, alpha))
        layers.setdefault(int(layer), set()).add(projection)
        for side, expected in [("A", (rank, module.in_features)), ("B", (module.out_features, rank))]:
            parameter = getattr(module, f"lora_{side}").weight
            if parameter.is_meta or hasattr(parameter, "to_local"):
                raise ValueError(f"Gather adapter before export: {name}.lora_{side}")
            if tuple(parameter.shape) != expected or not torch.isfinite(parameter).all():
                raise ValueError(f"Invalid adapter shape/values: {name}.lora_{side}")
            key = f"{target}.lora_{side}.weight"
            if key in weights:
                raise ValueError(f"Duplicate adapter mapping: {key}")
            weights[key] = parameter.detach().cpu().contiguous().clone()
            adapter_parameters.add(id(parameter))
        mappings.append(
            {
                "native": name,
                "diffusers": target,
                "base_shape": list(module.weight.shape),
                "rank": rank,
                "alpha": alpha,
                "scale": alpha / rank,
            }
        )
    if not weights or len(scales) != 1:
        raise ValueError("Export requires a nonempty adapter with uniform rank and alpha")
    all_generation_modules = {
        name
        for name, module in network.named_modules()
        if _NATIVE_MODULE.fullmatch(name) and isinstance(module, torch.nn.Linear)
    }
    if all_generation_modules != {entry["native"] for entry in mappings}:
        raise ValueError("Every generation attention projection must have an adapter")
    if set(layers) != set(range(len(layers))) or any(
        selected != set(GENERATION_PROJECTIONS) for selected in layers.values()
    ):
        raise ValueError("Every contiguous generation layer must contain Q/K/V/output adapters")
    for name, parameter in network.named_parameters():
        if parameter.requires_grad and id(parameter) not in adapter_parameters:
            raise ValueError(f"Trainable parameter is outside adapter allowlist: {name}")

    rank, alpha = next(iter(scales))
    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": sorted(item["diffusers"] for item in mappings),
        "lora_dropout": 0.0,
        "bias": "none",
        "inference_mode": True,
        "use_rslora": False,
        "use_dora": False,
    }
    manifest = {
        "schema": "cosmos3-generation-attention-lora-v1",
        "base_binding": binding,
        "training_view_sha256": training_view_sha256,
        "update": update,
        "source_identities": dict(source_identities),
        "target_mapping": mappings,
        "trainable_parameter_count": sum(tensor.numel() for tensor in weights.values()),
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in weights.items()},
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        save_file(weights, staging / "adapter_model.safetensors")
        (staging / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        manifest["files"] = {
            name: _sha256(staging / name) for name in ["adapter_model.safetensors", "adapter_config.json"]
        }
        (staging / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # A fresh directory is committed together; existing completed artifacts
        # are never reused as a mutable training/export destination.
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def load_generation_lora(
    transformer: torch.nn.Module,
    directory: str | Path,
    *,
    expected_base_binding: Mapping[str, str],
    adapter_name: str = "style",
) -> dict[str, Any]:
    """Validate and load into Diffusers' actual PEFT adapter API.

    The caller loads and verifies the bound base first. Adapter disabling uses
    the model's ``disable_adapters()`` API and leaves that base unchanged.
    """
    from peft import LoraConfig, set_peft_model_state_dict

    directory = Path(directory)
    manifest = json.loads((directory / "training_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "cosmos3-generation-attention-lora-v1":
        raise ValueError("Unsupported adapter schema")
    if _binding(expected_base_binding) != manifest["base_binding"]:
        raise ValueError("Adapter base checkpoint binding does not match")
    if set(manifest["files"]) != {"adapter_model.safetensors", "adapter_config.json"}:
        raise ValueError("Invalid adapter file inventory")
    for name, digest in manifest["files"].items():
        if _sha256(directory / name) != digest:
            raise ValueError(f"Adapter artifact checksum mismatch: {name}")
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    fixed_config = {
        "peft_type": "LORA",
        "lora_dropout": 0.0,
        "bias": "none",
        "inference_mode": True,
        "use_rslora": False,
        "use_dora": False,
    }
    if set(config) != set(fixed_config) | {"r", "lora_alpha", "target_modules"} or any(
        config[key] != value for key, value in fixed_config.items()
    ):
        raise ValueError("Unsupported adapter configuration")
    weights = load_file(directory / "adapter_model.safetensors", device="cpu")
    if set(weights) != set(manifest["tensors"]):
        raise ValueError("Adapter tensor inventory does not match manifest")
    targets = {entry["diffusers"]: entry for entry in manifest["target_mapping"]}
    if not targets or set(config["target_modules"]) != set(targets):
        raise ValueError("Adapter target inventory does not match config")
    target_pattern = r"layers\.\d+\.self_attn\.(add_q_proj|add_k_proj|add_v_proj|to_add_out)"
    model_targets = {
        name
        for name, module in transformer.named_modules()
        if re.fullmatch(target_pattern, name) and isinstance(module, torch.nn.Linear)
    }
    if model_targets != set(targets):
        raise ValueError("Adapter does not cover every generation attention projection")
    expected_keys = {f"{target}.lora_{side}.weight" for target in targets for side in ["A", "B"]}
    if set(weights) != expected_keys:
        raise ValueError("Adapter tensors do not match complete target pairs")
    for target, entry in targets.items():
        if not re.fullmatch(r"layers\.\d+\.self_attn\.(add_q_proj|add_k_proj|add_v_proj|to_add_out)", target):
            raise ValueError(f"Target is outside generation-attention allowlist: {target}")
        module = transformer.get_submodule(target)
        if not isinstance(module, torch.nn.Linear) or list(module.weight.shape) != entry["base_shape"]:
            raise ValueError(f"Base target shape/type mismatch: {target}")
        if entry["rank"] != config["r"] or entry["alpha"] != config["lora_alpha"]:
            raise ValueError(f"Adapter scaling mismatch: {target}")
        for side, shape in [("A", (config["r"], module.in_features)), ("B", (module.out_features, config["r"]))]:
            key = f"{target}.lora_{side}.weight"
            if tuple(weights[key].shape) != shape or not torch.isfinite(weights[key]).all():
                raise ValueError(f"Invalid adapter tensor: {key}")
            if manifest["tensors"][key] != {"shape": list(shape), "dtype": str(weights[key].dtype)}:
                raise ValueError(f"Adapter tensor metadata mismatch: {key}")
    if getattr(transformer, "peft_config", {}).get(adapter_name) is not None:
        raise ValueError(f"Adapter already exists: {adapter_name}")
    transformer.add_adapter(LoraConfig(**config), adapter_name=adapter_name)
    result = set_peft_model_state_dict(transformer, weights, adapter_name=adapter_name)
    missing_adapters = [name for name in result.missing_keys if f".{adapter_name}." in name and "lora_" in name]
    if missing_adapters or result.unexpected_keys:
        raise ValueError(f"PEFT adapter loading failed: {missing_adapters}, {result.unexpected_keys}")
    return manifest
