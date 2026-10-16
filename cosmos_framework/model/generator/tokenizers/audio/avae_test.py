# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The frozen audio tokenizer must include weight-norm replacement parameters."""

import unittest
from unittest.mock import patch

import torch
from torch.nn.utils import remove_weight_norm, weight_norm

from cosmos_framework.model.generator.tokenizers.audio.avae import AVAEModel


class WeightNormalizedAudio(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = weight_norm(torch.nn.Linear(3, 2, bias=False))

    def remove_weight_norm(self):
        remove_weight_norm(self.projection)


class AVAEFreezeTest(unittest.TestCase):
    def test_freezes_materialized_parameters_without_changing_output(self):
        torch.manual_seed(47)
        audio = WeightNormalizedAudio()
        inputs = torch.randn(4, 3)
        with torch.no_grad():
            expected = audio.projection(inputs)
            expected_weight = audio.projection.weight.detach().clone()
        with patch(
            "cosmos_framework.model.generator.tokenizers.audio.avae._load_avae_model",
            return_value=(audio, {}),
        ):
            wrapper = AVAEModel(dtype=torch.float32, device="cpu")
        self.assertFalse(wrapper.model.training)
        self.assertFalse(any(parameter.requires_grad for parameter in wrapper.model.parameters()))
        self.assertFalse(hasattr(audio.projection, "weight_g"))
        torch.testing.assert_close(audio.projection.weight, expected_weight, rtol=0, atol=0)
        torch.testing.assert_close(audio.projection(inputs), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
