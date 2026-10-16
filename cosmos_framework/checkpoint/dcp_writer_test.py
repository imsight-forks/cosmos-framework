# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU regressions for the opt-in local DCP writer and its error reporting."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.filesystem as filesystem
from pydantic import ValidationError
from torch.distributed.checkpoint.api import CheckpointException

from cosmos_framework.checkpoint import dcp as native_dcp
from cosmos_framework.configs.toml_config import sft_config as toml_config
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides
from cosmos_framework.utils import log
from cosmos_framework.utils.config import CheckpointConfig


def _checkpointer(copy_ahead=None):
    checkpointer = object.__new__(native_dcp.DistributedCheckpointer)
    checkpointer.config_checkpoint = CheckpointConfig(dcp_per_thread_copy_ahead=copy_ahead)
    checkpointer.save_to_object_store = False
    return checkpointer


def test_default_keeps_the_original_writer_constructor(monkeypatch, tmp_path):
    constructor = Mock()
    monkeypatch.setattr(native_dcp, "FileSystemWriter", constructor)
    destination = str(tmp_path / "checkpoint")
    assert _checkpointer().get_storage_writer(destination) is constructor.return_value
    constructor.assert_called_once_with(path=destination)
    assert CheckpointConfig().dcp_per_thread_copy_ahead is None
    assert toml_config.CheckpointConfig().dcp_per_thread_copy_ahead is None


@pytest.mark.parametrize("copy_ahead", [0, 1, 10_000_000])
def test_explicit_copy_ahead_preserves_other_public_writer_defaults(copy_ahead, tmp_path):
    writer = _checkpointer(copy_ahead).get_storage_writer(str(tmp_path / "checkpoint"))
    assert isinstance(writer, native_dcp._TracebackFileSystemWriter)
    assert writer.per_thread_copy_ahead == copy_ahead
    assert writer.thread_count == 1
    assert writer.sync_files is True
    assert writer.single_file_per_rank is True


@pytest.mark.parametrize("invalid", [-1, 0.5, "0", True, False])
def test_native_and_toml_configs_reject_invalid_copy_ahead(invalid):
    with pytest.raises(ValueError, match="dcp_per_thread_copy_ahead"):
        CheckpointConfig(dcp_per_thread_copy_ahead=invalid)
    with pytest.raises(ValidationError):
        toml_config.CheckpointConfig(dcp_per_thread_copy_ahead=invalid)


@pytest.mark.parametrize("invalid", [-1, True, "0"])
def test_native_assignment_rejects_invalid_overrides(invalid):
    config = CheckpointConfig()
    with pytest.raises(ValueError, match="dcp_per_thread_copy_ahead"):
        config.dcp_per_thread_copy_ahead = invalid


def test_paired_toml_selects_zero_and_forwards_the_checkpoint_field():
    recipe = (
        Path(__file__).resolve().parents[2] / "examples/toml/sft_config/segmentation_sft_nano.toml"
    )
    with recipe.open("rb") as stream:
        raw = toml_config.tomllib.load(stream)
    validated = toml_config.SFTExperimentConfig.model_validate(raw)
    assert validated.checkpoint.dcp_per_thread_copy_ahead == 0
    assert "checkpoint.dcp_per_thread_copy_ahead=0" in build_hydra_overrides(raw)
    omitted = {"job": {"experiment": "paired"}}
    assert not any(
        "dcp_per_thread_copy_ahead" in override for override in build_hydra_overrides(omitted)
    )


def test_zero_selects_serial_loader_and_real_cpu_dcp_values_roundtrip(monkeypatch, tmp_path):
    assert not torch.cuda.is_initialized()
    selected = []
    original_serial = filesystem._SerialCpuLoader
    original_queue_writer = filesystem._write_files_from_queue

    class ObservedSerialLoader(original_serial):
        def __init__(self, *args, **kwargs):
            selected.append("serial")
            super().__init__(*args, **kwargs)

    def unexpected_overlap(*args, **kwargs):
        raise AssertionError("Copy-ahead zero must not construct the overlapping loader")

    def queue_writer_with_available_cuda(*args, **kwargs):
        # Restrict the availability spy to filesystem dispatch. No CUDA API is
        # needed: the selected serial loader receives actual CPU tensors.
        with patch.object(torch.cuda, "is_available", return_value=True):
            return original_queue_writer(*args, **kwargs)

    monkeypatch.setattr(filesystem, "_SerialCpuLoader", ObservedSerialLoader)
    monkeypatch.setattr(filesystem, "_OverlappingCpuLoader", unexpected_overlap)
    monkeypatch.setattr(filesystem, "_write_files_from_queue", queue_writer_with_available_cuda)
    state = {
        "fp32": torch.arange(24, dtype=torch.float32).reshape(6, 4).t(),
        "bf16": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "iteration": 20,
    }
    destination = tmp_path / "checkpoint"
    writer = _checkpointer(0).get_storage_writer(str(destination))
    dcp.save(state, storage_writer=writer, no_dist=True)
    restored = {
        "fp32": torch.empty_like(state["fp32"]),
        "bf16": torch.empty_like(state["bf16"]),
        "iteration": 0,
    }
    dcp.load(restored, checkpoint_id=destination, no_dist=True)
    assert selected == ["serial"]
    assert (destination / ".metadata").is_file()
    assert torch.equal(restored["fp32"], state["fp32"])
    assert torch.equal(restored["bf16"], state["bf16"])
    assert restored["iteration"] == 20
    assert not torch.cuda.is_initialized()


def test_success_returns_the_identical_public_writer_future(monkeypatch, tmp_path):
    future = torch.futures.Future()
    future.set_result([])
    monkeypatch.setattr(filesystem.FileSystemWriter, "write_data", lambda *args: future)
    writer = _checkpointer(0).get_storage_writer(str(tmp_path / "checkpoint"))
    assert writer.write_data(Mock(), Mock()) is future
    assert future.wait() == []


@pytest.mark.parametrize("phase", ["write_data", "future_wait"])
def test_original_write_and_future_failures_are_logged_as_text_and_reraised(
    phase, monkeypatch, tmp_path
):
    original_error = RuntimeError("distinctive original checkpoint failure")

    def original_write_failure(*args):
        raise original_error

    class FailedFuture:
        def wait(self):
            raise original_error

    implementation = (
        original_write_failure if phase == "write_data" else lambda *args: FailedFuture()
    )
    monkeypatch.setattr(filesystem.FileSystemWriter, "write_data", implementation)
    records = []
    sink = log.logger.add(
        lambda message: records.append(message), format="{message}", level="ERROR"
    )
    destination = tmp_path / "checkpoint"
    writer = _checkpointer(0).get_storage_writer(str(destination))
    try:
        with pytest.raises(RuntimeError) as raised:
            writer.write_data(Mock(), Mock())
    finally:
        log.logger.remove(sink)
    assert raised.value is original_error
    assert len(records) == 1
    message = records[0]
    assert str(destination) in str(message)
    assert "Traceback (most recent call last)" in str(message)
    assert "distinctive original checkpoint failure" in str(message)
    assert (
        "original_write_failure" in str(message)
        if phase == "write_data"
        else "wait" in str(message)
    )
    assert message.record["exception"] is None
    assert message.record["extra"]["rank0_only"] is False


def test_actual_failed_cpu_dcp_write_does_not_publish_metadata(monkeypatch, tmp_path):
    assert not torch.cuda.is_initialized()

    def failed_tensor_write(*args, **kwargs):
        raise RuntimeError("actual CPU DCP serialization failure")

    monkeypatch.setattr(filesystem, "_write_item", failed_tensor_write)
    destination = tmp_path / "checkpoint"
    writer = _checkpointer(0).get_storage_writer(str(destination))
    with pytest.raises(CheckpointException):
        dcp.save({"weight": torch.arange(4)}, storage_writer=writer, no_dist=True)
    assert not (destination / ".metadata").exists()
    assert not (destination / ".metadata.tmp").exists()
    assert not torch.cuda.is_initialized()
