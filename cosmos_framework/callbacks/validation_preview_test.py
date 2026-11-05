# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The preview derivative preserves frozen validation inputs and label colors."""

import unittest

import torch
from scripts.explore.cosmos3_lora_training.native_preview import generate_rgb, preview_batch, rgb_frames


class ValidationPreviewTest(unittest.TestCase):
    def test_derivative_is_isolated_and_segmentation_uses_nearest(self):
        depth = torch.linspace(-1, 1, 3 * 21 * 8 * 8).reshape(3, 21, 8, 8)
        seg = torch.ones_like(depth)
        seg[:, :, :, :4] = -1
        batch = {
            "video": [[depth, seg, depth.clone()]],
            "image_size": [torch.tensor([8.0, 8.0, 8.0, 8.0])] * 3,
            "padding_mask": [torch.zeros(1, 1, 8, 8)],
            "num_frames": [torch.tensor([21])],
            "frame_end": [torch.tensor([20])],
            "conditioning_stream_roles": ["depth,seg,target"],
        }
        result = preview_batch(batch, 17, 5, 5)
        self.assertEqual(result["video"][0][0].shape, (3, 17, 5, 5))
        self.assertEqual(set(result["video"][0][1].unique().tolist()), {-1.0, 1.0})
        self.assertEqual(batch["video"][0][0].shape, (3, 21, 8, 8))
        self.assertEqual(batch["num_frames"][0].item(), 21)
        self.assertEqual(result["num_frames"][0].item(), 17)
        self.assertEqual(result["padding_mask"][0].shape, (1, 1, 5, 5))
        self.assertEqual(rgb_frames(result["video"][0][2]).shape, (17, 5, 5, 3))

    def test_rejects_incomplete_control_view(self):
        with self.assertRaisesRegex(ValueError, "requires segmentation"):
            preview_batch({"video": [[torch.zeros(3, 17, 8, 8)]], "image_size": []}, 17, 4, 4)

    def test_native_input_mutation_cannot_change_paired_controls(self):
        class MutatingModel:
            def __init__(self):
                self.seen = []

            def generate_samples_from_batch(self, batch):
                self.seen.append(len(batch["video"][0]))
                target = batch["video"][0].pop()
                batch["video"] = []
                return {"vision": [target]}

            def decode(self, value):
                return value

        model = MutatingModel()
        batch = {"video": [[torch.zeros(3, 17, 4, 4) for _ in range(3)]]}
        base = generate_rgb(model, batch)
        adapted = generate_rgb(model, batch)
        self.assertEqual(model.seen, [3, 3])
        self.assertEqual(len(batch["video"][0]), 3)
        self.assertTrue((base == adapted).all())
