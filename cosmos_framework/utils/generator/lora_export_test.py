# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU checks of the native adapter / actual Diffusers PEFT boundary."""

import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from cosmos_framework.utils.generator.lora import (
    init_lora_weights_post_materialization,
    inject_lora_pre_fsdp,
    lora_disabled,
)
from cosmos_framework.utils.generator.lora_export import (
    GENERATION_PROJECTIONS,
    export_generation_lora,
    load_generation_lora,
)

BINDING = {"repository": "fixture/Cosmos3-Nano", "revision": "test", "sha256": "1" * 64}


def tiny_transformer():
    return Cosmos3OmniTransformer(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=64,
        latent_channel=4,
        latent_patch_size=2,
        patch_latent_dim=16,
        rope_axes_dim=[4, 2, 2],
    ).eval()


def denoiser_inputs():
    torch.manual_seed(912)
    return dict(
        input_ids=torch.tensor([3, 4]),
        text_indexes=torch.tensor([0, 1]),
        position_ids=torch.arange(10).repeat(3, 1),
        und_len=2,
        sequence_length=10,
        vision_tokens=[torch.randn(1, 4, 2, 4, 4)],
        vision_token_shapes=[(2, 2, 2)],
        vision_sequence_indexes=torch.arange(2, 10),
        vision_mse_loss_indexes=torch.arange(2, 10),
        vision_timesteps=torch.full((8,), 500.0),
        vision_noisy_frame_indexes=[torch.tensor([0, 1])],
    )


def native_projection_model(transformer):
    """Native names/injector with the exact Diffusers projection base tensors."""
    native = torch.nn.Module()
    native.model = torch.nn.Module()
    native.model.layers = torch.nn.ModuleList()
    for layer in transformer.layers:
        native_layer = torch.nn.Module()
        native_layer.self_attn = torch.nn.Module()
        for source, target in GENERATION_PROJECTIONS.items():
            setattr(native_layer.self_attn, source, copy.deepcopy(getattr(layer.self_attn, target)))
        native.model.layers.append(native_layer)
    inject_lora_pre_fsdp(
        native,
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules=",".join(GENERATION_PROJECTIONS),
    )
    # Preserve the copied base tensors: only injector-created adapters are meta.
    for module in native.modules():
        if hasattr(module, "lora_A"):
            module.lora_A.to_empty(device="cpu")
            module.lora_B.to_empty(device="cpu")
    init_lora_weights_post_materialization(native)
    return native


def export(native, path):
    return export_generation_lora(
        native,
        path,
        base_binding=BINDING,
        training_view_sha256="2" * 64,
        update=20,
        source_identities={"framework_revision": "fixture", "dependency_lock_sha256": "3" * 64},
    )


class LoraExportTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_exports_actual_activation_checkpoint_wrapped_layers(self):
        native = native_projection_model(tiny_transformer())
        for index, layer in enumerate(native.model.layers):
            native.model.layers[index] = checkpoint_wrapper(layer)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = export(native, Path(tmp) / "wrapped")
            self.assertEqual(len(manifest["target_mapping"]), 8)
            self.assertTrue(
                all("._checkpoint_wrapped_module.self_attn." in entry["native"] for entry in manifest["target_mapping"])
            )
            transformer = tiny_transformer()
            load_generation_lora(transformer, Path(tmp) / "wrapped", expected_base_binding=BINDING)

    def test_actual_diffusers_forward_disable_and_fresh_process_reload(self):
        base = tiny_transformer()
        native = native_projection_model(base)
        inputs = denoiser_inputs()
        with torch.no_grad():
            base_output = base(**inputs).sample[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export(native, root / "zero")
            zero = copy.deepcopy(base)
            load_generation_lora(zero, root / "zero", expected_base_binding=BINDING)
            with torch.no_grad():
                torch.testing.assert_close(zero(**inputs).sample[0], base_output, rtol=0, atol=0)

            # A real native update exercises orientation and alpha/r, including
            # the smaller grouped-query K/V output width.
            optimizer = torch.optim.AdamW([p for p in native.parameters() if p.requires_grad], lr=1e-2)
            x = torch.randn(7, 32)
            for _ in range(3):
                optimizer.zero_grad()
                loss = sum(
                    getattr(layer.self_attn, source)(x).square().mean()
                    for layer in native.model.layers
                    for source in GENERATION_PROJECTIONS
                )
                loss.backward()
                optimizer.step()
            manifest = export(native, root / "trained")
            self.assertEqual(len(manifest["target_mapping"]), 8)
            candidate = copy.deepcopy(base)
            load_generation_lora(candidate, root / "trained", expected_base_binding=BINDING)
            native_projection = native.model.layers[0].self_attn.q_proj_moe_gen
            adapted_projection_output = native_projection(x)
            with lora_disabled(native):
                torch.testing.assert_close(
                    native_projection(x), torch.nn.functional.linear(x, native_projection.weight), rtol=0, atol=0
                )
                with lora_disabled(native):
                    self.assertFalse(native_projection._lora_enabled)
                self.assertFalse(native_projection._lora_enabled)
            torch.testing.assert_close(native_projection(x), adapted_projection_output, rtol=0, atol=0)
            with self.assertRaisesRegex(RuntimeError, "probe failure"):
                with lora_disabled(native):
                    raise RuntimeError("probe failure")
            self.assertTrue(native_projection._lora_enabled)
            for native_layer, diffusers_layer in zip(native.model.layers, candidate.layers):
                for source, target in GENERATION_PROJECTIONS.items():
                    torch.testing.assert_close(
                        getattr(native_layer.self_attn, source)(x),
                        getattr(diffusers_layer.self_attn, target)(x),
                        rtol=1e-6,
                        atol=1e-7,
                    )
            with torch.no_grad():
                expected = candidate(**inputs).sample[0]
                self.assertGreater((expected - base_output).abs().max().item(), 0)
                candidate.disable_adapters()
                torch.testing.assert_close(candidate(**inputs).sample[0], base_output, rtol=0, atol=0)
            torch.save({"base": base.state_dict(), "inputs": inputs}, root / "fixture.pt")
            child = """
import sys, torch
from pathlib import Path
from cosmos_framework.utils.generator.lora_export_test import tiny_transformer, BINDING
from cosmos_framework.utils.generator.lora_export import load_generation_lora
root=Path(sys.argv[1])
fixture=torch.load(root/'fixture.pt', weights_only=True)
model=tiny_transformer()
model.load_state_dict(fixture['base'], strict=True)
load_generation_lora(model, root/'trained', expected_base_binding=BINDING)
with torch.no_grad():
    torch.save(model(**fixture['inputs']).sample[0], root/'reloaded.pt')
"""
            subprocess.run([sys.executable, "-c", child, str(root)], check=True, timeout=120)
            actual = torch.load(root / "reloaded.pt", weights_only=True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_binding_shape_corruption_and_existing_destination(self):
        base = tiny_transformer()
        native = native_projection_model(base)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter"
            export(native, path)
            with self.assertRaises(FileExistsError):
                export(native, path)
            with self.assertRaisesRegex(ValueError, "binding"):
                load_generation_lora(base, path, expected_base_binding={**BINDING, "revision": "wrong"})
            wrong = tiny_transformer()
            wrong.layers[0].self_attn.add_q_proj = torch.nn.Linear(16, 32)
            with self.assertRaisesRegex(ValueError, "shape/type"):
                load_generation_lora(wrong, path, expected_base_binding=BINDING)
            with (path / "adapter_model.safetensors").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_generation_lora(base, path, expected_base_binding=BINDING)

    def test_rejects_unexpected_trainable_weights_and_partial_pairs(self):
        native = native_projection_model(tiny_transformer())
        native.unexpected = torch.nn.Parameter(torch.ones(1))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "allowlist"):
                export(native, Path(tmp) / "invalid")
            native.unexpected.requires_grad_(False)
            del native.model.layers[1].self_attn.o_proj_moe_gen
            with self.assertRaisesRegex(ValueError, "Q/K/V/output"):
                export(native, Path(tmp) / "partial")


if __name__ == "__main__":
    unittest.main()
