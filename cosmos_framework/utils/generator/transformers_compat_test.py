# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU regression checks for the Transformers 4.57 / 5.1 Nano compatibility boundary."""

import copy
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM
from cosmos_framework.utils.generator.monkey_patch import patch_qwen3_vl_forward, patch_qwen3_vl_vision_varlen_attention


def _tiny_config() -> Qwen3VLConfig:
    return Qwen3VLConfig(
        text_config={
            "vocab_size": 256,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "rope_scaling": {"rope_type": "default", "mrope_section": [8, 12, 12]},
            "attention_dropout": 0.0,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 128,
            "num_position_embeddings": 256,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
        vision_end_token_id=253,
        tie_word_embeddings=False,
    )


class TransformersCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2701)

    def _models(self) -> tuple[Qwen3VLForConditionalGeneration, Qwen3VLForConditionalGeneration]:
        reference = Qwen3VLForConditionalGeneration(_tiny_config()).eval()
        candidate = copy.deepcopy(reference)
        patch_qwen3_vl_forward(candidate.model)
        return reference, candidate

    def test_native_nano_tied_embeddings_and_safe_export(self) -> None:
        config = _tiny_config().to_dict()
        config["tie_word_embeddings"] = True
        config["text_config"]["tie_word_embeddings"] = True
        model = Qwen3VLTextForCausalLM(Qwen3VLMoTConfig(config, include_visual=False))
        model.tie_weights()
        self.assertIs(model.get_input_embeddings().weight, model.get_output_embeddings().weight)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)

    def test_hf_training_wrapper_meta_initialization(self) -> None:
        from cosmos_framework.model.generator.hf_model import HFModel

        with tempfile.TemporaryDirectory() as directory:
            _tiny_config().save_pretrained(directory)
            model = HFModel(directory, dtype=torch.float32, attn_implementation="sdpa", trust_remote_code=False)
        parameters = list(model.parameters())
        self.assertTrue(parameters)
        self.assertTrue(all(parameter.is_meta for parameter in parameters))

    def test_unqualified_transformers_version_is_rejected(self) -> None:
        with patch("cosmos_framework.utils.generator.monkey_patch.transformers.__version__", "5.2.0"):
            for patcher in [patch_qwen3_vl_forward, patch_qwen3_vl_vision_varlen_attention]:
                with self.assertRaisesRegex(ValueError, "version 5.2.0 is not supported"):
                    patcher(torch.nn.Identity())

    def test_text_only_forward_and_zero_visual_gradients(self) -> None:
        reference, candidate = self._models()
        candidate.model.visual.config.return_dict = False
        tokens = torch.tensor([[5, 6, 7, 8]])
        with torch.no_grad():
            expected = reference(input_ids=tokens, use_cache=False).logits
        calls = []
        hook = candidate.model.visual.register_forward_hook(lambda *_: calls.append(True))
        actual = candidate(input_ids=tokens, use_cache=False).logits
        hook.remove()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(len(calls), 1)
        actual.square().mean().backward()
        for parameter in candidate.model.visual.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

    def test_image_video_and_mixed_forward_backward(self) -> None:
        for modes in [("image",), ("video",), ("image", "video")]:
            with self.subTest(modes=modes):
                reference, candidate = self._models()
                tokens = [5]
                kwargs = {}
                for mode in modes:
                    tokens.extend([252, 250 if mode == "image" else 251, 253])
                    pixel_key = "pixel_values" if mode == "image" else "pixel_values_videos"
                    kwargs[pixel_key] = torch.randn(4, 24)
                    kwargs[mode + "_grid_thw"] = torch.tensor([[1, 2, 2]])
                kwargs["input_ids"] = torch.tensor([tokens + [6]])
                expected = reference(**kwargs, use_cache=False).logits
                expected.square().mean().backward()
                calls = []
                hook = candidate.model.visual.register_forward_hook(lambda *_: calls.append(True))
                actual = candidate(**kwargs, use_cache=False).logits
                hook.remove()
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                self.assertEqual(len(calls), 1)
                actual.square().mean().backward()
                for (name, parameter), (other_name, other) in zip(
                    candidate.named_parameters(), reference.named_parameters(), strict=True
                ):
                    self.assertEqual(name, other_name)
                    if other.grad is not None:
                        torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-4, atol=1e-6)

    def test_cached_decode_without_explicit_cache_position(self) -> None:
        reference, candidate = self._models()
        tokens = torch.tensor([[5, 6, 7, 8]])
        with torch.no_grad():
            expected_prefill = reference(input_ids=tokens, use_cache=True)
            actual_prefill = candidate(input_ids=tokens, use_cache=True)
            expected = reference(input_ids=torch.tensor([[5, 6, 7, 8, 9]]), use_cache=False).logits[:, -1:]
            actual = candidate(
                input_ids=torch.tensor([[9]]), past_key_values=actual_prefill.past_key_values, use_cache=True
            ).logits
        torch.testing.assert_close(actual_prefill.logits, expected_prefill.logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_vision_varlen_patch_preserves_outputs_and_gradients(self) -> None:
        reference, candidate = self._models()
        observed_boundaries = []

        def math_varlen(module, query, key, value, attention_mask, scaling, **kwargs):
            # Isolate the HF API wiring from the CUDA backend. Use independent per-image
            # math SDPA as the operator; the production attention kernels have GPU probes.
            self.assertIsNone(attention_mask)
            boundaries = kwargs["cu_seq_lens_q"].tolist()
            self.assertEqual(boundaries, kwargs["cu_seq_lens_k"].tolist())
            self.assertEqual(kwargs["max_length_q"], 8)
            observed_boundaries.append(boundaries)
            chunks = [
                torch.nn.functional.scaled_dot_product_attention(
                    query[:, :, start:end], key[:, :, start:end], value[:, :, start:end], scale=scaling
                )
                for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
            ]
            return torch.cat(chunks, dim=2).transpose(1, 2), None

        with patch("cosmos_framework.utils.generator.hf_attention_cosmos.hf_attention_cosmos", math_varlen):
            self.assertEqual(patch_qwen3_vl_vision_varlen_attention(candidate), 2)
        pixels = torch.randn(12, 24)
        grid = torch.tensor([[1, 2, 2], [1, 4, 2]])
        expected = reference.model.visual(pixels, grid_thw=grid)
        actual = candidate.model.visual(pixels, grid_thw=grid)
        self.assertIs(type(actual), type(expected))

        def feature_tensors(output):
            if isinstance(output, tuple):
                return [output[0], *output[1]]
            return [output.pooler_output, *output.deepstack_features]

        expected_features, actual_features = feature_tensors(expected), feature_tensors(actual)
        for left, right in zip(actual_features, expected_features, strict=True):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        sum(tensor.square().mean() for tensor in actual_features).backward()
        sum(tensor.square().mean() for tensor in expected_features).backward()
        for left, right in zip(candidate.model.visual.parameters(), reference.model.visual.parameters(), strict=True):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6)
        self.assertEqual(observed_boundaries, [[0, 4, 12], [0, 4, 12]])


if __name__ == "__main__":
    unittest.main()
