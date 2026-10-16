# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Failure-boundary checks for native training evidence collection."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from cosmos_framework.callbacks.generation_lora_audit import GenerationLoraAudit
from cosmos_framework.utils.generator.lora_export_test import native_projection_model, tiny_transformer
from cosmos_framework.utils.generator.optimizer import OptimizersContainer


class GenerationLoraAuditTest(unittest.TestCase):
    def model(self):
        root = torch.nn.Module()
        root.net = torch.nn.Module()
        root.net.language_model = native_projection_model(tiny_transformer())
        return root

    def test_accepts_actual_activation_checkpoint_wrapper_only(self):
        model = self.model()
        layers = model.net.language_model.model.layers
        for index, layer in enumerate(layers):
            layers[index] = checkpoint_wrapper(layer)
        with tempfile.TemporaryDirectory() as tmp:
            audit = GenerationLoraAudit(tmp)
            audit.on_train_start(model)
            self.assertEqual(len(audit.initial_adapters), 16)
            self.assertTrue(all("._checkpoint_wrapped_module.self_attn." in name for name in audit.initial_adapters))
            layers[0]._checkpoint_wrapped_module.accidental = torch.nn.Parameter(torch.ones(1))
            with self.assertRaisesRegex(ValueError, "Unexpected trainable"):
                GenerationLoraAudit(str(Path(tmp) / "reject")).on_train_start(model)

    def test_covers_unregistered_native_tokenizer_weights(self):
        model = self.model()
        auxiliary = torch.nn.Linear(2, 2).requires_grad_(False)
        model.tokenizer_vision_gen = SimpleNamespace(model=SimpleNamespace(model=auxiliary))
        self.assertFalse(any(parameter is auxiliary.weight for parameter in model.parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            audit = GenerationLoraAudit(tmp)
            audit.on_train_start(model)
            self.assertEqual(len(audit.initial_auxiliary), 2)
            with torch.no_grad():
                auxiliary.weight.add_(1)
            with self.assertRaisesRegex(ValueError, "Frozen parameters changed"):
                audit.on_train_end(model)
            auxiliary.requires_grad_(True)
            with self.assertRaisesRegex(ValueError, "Auxiliary tokenizer parameter is trainable"):
                GenerationLoraAudit(str(Path(tmp) / "reject")).on_train_start(model)

    def test_records_real_update_and_rejects_frozen_mutation(self):
        torch.manual_seed(91)
        model = self.model()
        with tempfile.TemporaryDirectory() as tmp:
            audit = GenerationLoraAudit(tmp)
            audit.on_train_start(model)
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=0.01)
            loss = sum(
                layer.self_attn.q_proj_moe_gen(torch.randn(5, 32)).square().mean()
                for layer in model.net.language_model.model.layers
            )
            loss.backward()
            audit.on_after_backward(model)
            audit.on_before_optimizer_step(model, optimizer, None, None)
            optimizer.step()
            with patch("torch.cuda.is_available", return_value=False):
                audit.on_training_step_end(model, {"sample_index": torch.tensor([0])}, {}, loss, 1)
            audit.on_train_end(model, 1)
            events = [json.loads(line) for line in (Path(tmp) / "rank-0.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["changed_frozen"], [])
            self.assertTrue(events[-1]["changed_adapters"])
            first_gradients = events[1]["gradients"]
            self.assertEqual(
                first_gradients["net.language_model.model.layers.0.self_attn.q_proj_moe_gen.lora_A.weight"]["norm"],
                0.0,
            )
            self.assertGreater(
                first_gradients["net.language_model.model.layers.0.self_attn.q_proj_moe_gen.lora_B.weight"]["norm"],
                0.0,
            )
            with torch.no_grad():
                model.net.language_model.model.layers[0].self_attn.q_proj_moe_gen.weight.add_(1)
            with self.assertRaisesRegex(ValueError, "Frozen parameters changed"):
                audit.on_train_end(model, 1)
            gradient = model.net.language_model.model.layers[0].self_attn.q_proj_moe_gen.lora_B.weight.grad
            gradient.fill_(float("nan"))
            with self.assertRaisesRegex(ValueError, "Nonfinite pre-clipping"):
                audit.on_after_backward(model)

    def test_actual_optimizer_container_update_preserves_groups_and_enforces_scope(self):
        torch.manual_seed(92)
        model = self.model()
        optimizer = OptimizersContainer(
            model,
            optimizer_type="adamw",
            fused=True,
            lr=0.01,
            weight_decay=0.0,
            keys_to_select=["lora_"],
            lr_multipliers={"q_proj_moe_gen": 2.0},
        )
        self.assertFalse(hasattr(optimizer, "param_groups"))
        self.assertEqual(len(optimizer.optimizers), 1)
        groups = optimizer.optimizers[0].param_groups
        self.assertEqual(len(groups), 2)
        with tempfile.TemporaryDirectory() as tmp:
            audit = GenerationLoraAudit(tmp)
            audit.on_train_start(model)
            optimizer.zero_grad()
            loss = sum(
                projection(torch.randn(5, projection.in_features)).square().mean()
                for layer in model.net.language_model.model.layers
                for projection in layer.self_attn.children()
            )
            loss.backward()
            audit.on_after_backward(model)
            parameters = [parameter for group in groups for parameter in group["params"]]
            original_groups = [
                (id(group), group["lr"], group["weight_decay"], tuple(map(id, group["params"]))) for group in groups
            ]
            weights = [parameter.detach().clone() for parameter in parameters]
            gradients = [parameter.grad.clone() for parameter in parameters]
            audit.on_before_optimizer_step(model, optimizer, None, None)
            self.assertEqual(
                original_groups,
                [(id(group), group["lr"], group["weight_decay"], tuple(map(id, group["params"]))) for group in groups],
            )
            self.assertEqual(len(optimizer.optimizers[0].state), 0)
            self.assertTrue(
                all(torch.equal(parameter, value) for parameter, value in zip(parameters, weights, strict=True))
            )
            self.assertTrue(
                all(torch.equal(parameter.grad, value) for parameter, value in zip(parameters, gradients, strict=True))
            )
            # Every child must be inspected: a duplicated child duplicates all
            # parameter membership and must fail before an optimizer update.
            optimizer.optimizers.append(optimizer.optimizers[0])
            with self.assertRaisesRegex(ValueError, "Optimizer groups"):
                audit.on_before_optimizer_step(model, optimizer, None, None)
            optimizer.optimizers.pop()
            missing = groups[0]["params"].pop()
            with self.assertRaisesRegex(ValueError, "Optimizer groups"):
                audit.on_before_optimizer_step(model, optimizer, None, None)
            groups[0]["params"].append(missing)
            frozen = model.net.language_model.model.layers[0].self_attn.q_proj_moe_gen.weight
            groups[0]["params"].append(frozen)
            with self.assertRaisesRegex(ValueError, "Optimizer groups"):
                audit.on_before_optimizer_step(model, optimizer, None, None)
            groups[0]["params"].pop()
            optimizer.step()
            with patch("torch.cuda.is_available", return_value=False):
                audit.on_training_step_end(model, {"sample_index": torch.tensor([0])}, {}, loss, 1)
            audit.on_train_end(model, 1)
            events = [json.loads(line) for line in (Path(tmp) / "rank-0.jsonl").read_text().splitlines()]
            reported = next(event["optimizer_groups"] for event in events if event["event"] == "before_update")
            self.assertEqual([group["lr"] for group in reported], [group["lr"] for group in groups])
            self.assertEqual([group["weight_decay"] for group in reported], [group["weight_decay"] for group in groups])
            self.assertEqual({name for group in reported for name in group["parameters"]}, set(audit.initial_adapters))
            self.assertTrue(events[-1]["changed_adapters"])
            self.assertEqual(events[-1]["changed_frozen"], [])
            self.assertEqual(events[-1]["changed_auxiliary"], [])

    def test_rejects_extra_trainables_optimizer_scope_and_reused_attempt(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as tmp:
            model.accidental = torch.nn.Parameter(torch.ones(1))
            with self.assertRaisesRegex(ValueError, "Unexpected trainable"):
                GenerationLoraAudit(str(Path(tmp) / "unexpected")).on_train_start(model)
            model.accidental.requires_grad_(False)
            audit = GenerationLoraAudit(str(Path(tmp) / "valid"))
            audit.on_train_start(model)
            with self.assertRaises(FileExistsError):
                GenerationLoraAudit(str(Path(tmp) / "valid")).on_train_start(model)
            optimizer = torch.optim.AdamW(model.parameters())
            with self.assertRaisesRegex(ValueError, "Optimizer groups"):
                audit.on_before_optimizer_step(model, optimizer, None, None)


if __name__ == "__main__":
    unittest.main()
