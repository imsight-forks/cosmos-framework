# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Paired segmentation clock, palette, cursor-boundary and diagnostic regressions."""

import shutil
import subprocess
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.local_datasets import transfer_sft_dataset as module
from cosmos_framework.data.generator.local_datasets.helper import ffmpeg_decode_video
from cosmos_framework.data.generator.local_datasets.sft_dataset import SFTDataset
from cosmos_framework.data.generator.sequence_packing import SequencePlan


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required for palette decode")
def test_nearest_decoder_preserves_lossless_mask_palette(tmp_path):
    pixels = np.zeros((3, 6, 8, 3), dtype=np.uint8)
    pixels[:, :, :4] = (255, 0, 0)
    pixels[:, :, 4:] = (0, 255, 0)
    path = tmp_path / "palette.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "8x6",
            "-r",
            "7",
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            str(path),
        ],
        input=pixels.tobytes(),
        check=True,
    )
    decoded = np.stack(list(ffmpeg_decode_video(str(path), scale_hw=(13, 17), scale_flags="neighbor")))
    assert decoded.shape == (3, 13, 17, 3)
    assert np.unique(decoded.reshape(-1, 3), axis=0).tolist() == [[0, 255, 0], [255, 0, 0]]


def test_seg_control_uses_nearest_neighbor_and_shared_frame_selection(monkeypatch):
    source = object.__new__(module.TransferSFTDataset)
    source.control_type = "seg"
    source.s3_client = Mock()
    monkeypatch.setattr(module, "download_from_s3", lambda *_: b"fake video")
    decoded = np.stack([np.full((6, 8, 3), index, dtype=np.uint8) for index in range(9)])
    decoder = Mock(return_value=iter(decoded))
    monkeypatch.setattr(module, "ffmpeg_decode_video", decoder)
    actual = source._load_precomputed_control(
        "seg.mkv",
        start_frame=2,
        end_frame=6,
        temporal_interval=2,
        resize_hw=(6, 8),
        crop_y=1,
        crop_x=2,
        target_h=4,
        target_w=4,
    )
    assert actual.shape == (3, 4, 4, 3)
    assert actual[:, 0, 0, 0].tolist() == [2, 4, 6]
    assert decoder.call_args.kwargs["scale_flags"] == "neighbor"


def test_paired_output_keeps_rgb_target_and_shared_control_positions(monkeypatch):
    source = object.__new__(module.TransferSFTDataset)
    source.control_type = "seg"
    source.output_sizes = {"16,9": (4, 4)}
    rgb = torch.full((3, 5, 4, 4), 255, dtype=torch.uint8)
    size = torch.tensor([4, 4, 4, 4])
    parent = dict(
        video=rgb,
        frame_start=3,
        num_multiplier=2,
        image_size=size,
        sequence_plan=SequencePlan(has_text=True, has_vision=True),
    )
    monkeypatch.setattr(SFTDataset, "process_one_sample", lambda *_: dict(parent))
    loader = Mock(return_value=np.zeros((5, 4, 4, 3), dtype=np.uint8))
    source._load_precomputed_control = loader
    actual = source.process_one_sample(
        dict(
            uuid="paired",
            width=4,
            height=4,
            aspect_ratio="16,9",
            control_path="seg.mkv",
        )
    )
    control, target = actual["video"]
    assert torch.equal(control, torch.full_like(control, -1))
    assert torch.equal(target, torch.ones_like(target))
    assert actual["dataset_name"] == "video_transfer"
    assert actual["sequence_plan"].share_vision_temporal_positions
    assert actual["sequence_plan"].condition_frame_indexes_vision == []
    assert loader.call_args.kwargs["start_frame"] == 3
    assert loader.call_args.kwargs["end_frame"] == 11
    assert loader.call_args.kwargs["temporal_interval"] == 2


def test_49_frame_diagnostic_is_not_dropped_and_caption_budget_is_retained(monkeypatch):
    monkeypatch.setattr(module.boto3, "client", lambda *_, **__: Mock())
    metadata_loader = Mock(return_value=[dict(uuid="tiny", t2w_windows=[dict(start_frame=0, end_frame=48)])])
    constructor = Mock()
    monkeypatch.setattr(module, "_load_transfer_metadata", metadata_loader)
    monkeypatch.setattr(module, "TransferSFTDataset", constructor)
    module.get_transfer_sft_dataset(
        "train.jsonl",
        num_video_frames=49,
        control_type="seg",
        max_caption_tokens=2048,
        conditioning_fps=-1,
        sample_by_window=True,
    )
    assert metadata_loader.call_args.kwargs["min_frames"] == 49
    assert constructor.call_args.kwargs["max_caption_tokens"] == 2048
    assert constructor.call_args.kwargs["conditioning_fps"] == -1
    assert len(constructor.call_args.kwargs["metadata"]) == 1


def test_map_adapter_fails_on_missing_frozen_sample(monkeypatch):
    monkeypatch.setattr(module.boto3, "client", lambda *_, **__: Mock())
    source = Mock(metadata=[dict(uuid="bad")], s3_credentials={})
    source.process_one_sample.return_value = None
    dataset = module.TransferSFTMapDataset(source)
    with pytest.raises(RuntimeError, match="Frozen paired training sample"):
        dataset[0]


def test_map_adapter_rejects_uncheckpointed_worker_rng(monkeypatch):
    monkeypatch.setattr(module.boto3, "client", lambda *_, **__: Mock())
    source = Mock(metadata=[dict(uuid="sample")], s3_credentials={})
    dataset = module.TransferSFTMapDataset(source)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: Mock())
    with pytest.raises(RuntimeError, match="num_workers=0"):
        dataset[0]
