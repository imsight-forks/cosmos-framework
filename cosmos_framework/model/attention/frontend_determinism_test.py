# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU dispatch regressions; FA2 entrypoints are spies, never CUDA kernels."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from cosmos_framework.model.attention import frontend


@pytest.fixture
def dispatch(monkeypatch):
    """Use the unchanged FA2 wrapper with tensor checks and kernel calls spied."""
    calls = {"chooser": [], "kernel": []}
    interface = ModuleType("flash_attn.flash_attn_interface")

    def dense(**kwargs):
        calls["kernel"].append(kwargs)
        query = kwargs["q"]
        lse = torch.zeros(query.shape[0], query.shape[2], query.shape[1])
        return query, lse, None

    def varlen(**kwargs):
        calls["kernel"].append(kwargs)
        query = kwargs["q"]
        lse = torch.zeros(query.shape[1], query.shape[0])
        return query, lse, None

    interface.flash_attn_func = dense
    interface.flash_attn_varlen_func = varlen
    monkeypatch.setitem(sys.modules, "flash_attn.flash_attn_interface", interface)
    # The installed CPU import chooses a stub. Load the actual unchanged wrapper
    # file under a test name so both wrapper branches can be checked without CUDA.
    wrapper_path = Path(frontend.__file__).parent / "flash2/functions.py"
    if not wrapper_path.exists():
        import cosmos_framework.model.attention.flash2 as flash2

        wrapper_path = Path(flash2.__file__).with_name("functions.py")
    spec = importlib.util.spec_from_file_location("cpu_fa2_dispatch_wrapper", wrapper_path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    monkeypatch.setattr(wrapper, "assert_universal_tensor_checks", lambda *args: None)
    monkeypatch.setattr(wrapper, "flash2_attention_check", lambda **kwargs: True)
    monkeypatch.setattr(wrapper, "is_torch_compiling", lambda: False)
    monkeypatch.setattr(frontend, "universal_tensor_checks", lambda **kwargs: True)
    monkeypatch.setattr(frontend, "attention_tensor_checks", lambda **kwargs: True)
    monkeypatch.setattr(frontend, "attention_param_checks", lambda **kwargs: None)
    monkeypatch.setattr(frontend, "is_torch_compiling", lambda: False)

    def varlen_checks(**kwargs):
        return tuple(
            kwargs[name] for name in ("cumulative_seqlen_Q", "cumulative_seqlen_KV", "max_seqlen_Q", "max_seqlen_KV")
        )

    def choose(**kwargs):
        calls["chooser"].append(kwargs)
        return "flash2"

    monkeypatch.setattr(frontend, "varlen_tensor_checks", varlen_checks)
    monkeypatch.setattr(frontend, "choose_backend", choose)
    monkeypatch.setitem(frontend.BACKEND_MAP, "flash2", wrapper.flash2_attention)
    return calls


def invoke(*, deterministic=False, varlen=False, backend_kwargs=None, requires_grad=True):
    query = torch.arange(32, dtype=torch.float32).reshape(1, 4, 2, 4)
    query.requires_grad_(requires_grad)
    kwargs = {}
    if varlen:
        offsets = torch.tensor([0, 4], dtype=torch.int32)
        kwargs.update(cumulative_seqlen_Q=offsets, cumulative_seqlen_KV=offsets, max_seqlen_Q=4, max_seqlen_KV=4)
    return frontend.attention(
        query, query, query, backend="flash2", deterministic=deterministic, backend_kwargs=backend_kwargs, **kwargs
    )


@pytest.mark.parametrize("varlen", [False, True], ids=["dense", "varlen-wrapper-only"])
@pytest.mark.parametrize(
    "environment, explicit, expected",
    [
        (None, False, False),
        ("", False, False),
        ("0", False, False),
        ("true", False, False),
        ("1", False, True),
        (None, True, True),
        ("0", True, True),
        ("1", True, True),
    ],
)
def test_environment_opt_in_reaches_backend_choice_and_fa2_entrypoint(
    monkeypatch, dispatch, environment, explicit, expected, varlen
):
    if environment is None:
        monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC", raising=False)
    else:
        monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", environment)
    rng = torch.get_rng_state().clone()
    mode = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    result = invoke(deterministic=explicit, varlen=varlen)
    assert result.shape == (1, 4, 2, 4)
    assert len(dispatch["chooser"]) == len(dispatch["kernel"]) == 1
    assert dispatch["chooser"][0]["deterministic"] is expected
    assert dispatch["kernel"][0]["deterministic"] is expected
    assert torch.are_deterministic_algorithms_enabled() is mode
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only
    assert torch.equal(torch.get_rng_state(), rng)
    assert not torch.cuda.is_initialized()


def test_backend_false_cannot_disable_environment_opt_in(monkeypatch, dispatch):
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    backend_kwargs = {"deterministic": False}
    invoke(backend_kwargs=backend_kwargs)
    assert dispatch["chooser"][0]["deterministic"] is True
    assert dispatch["kernel"][0]["deterministic"] is True
    assert backend_kwargs == {"deterministic": False}


def test_omitted_caller_flag_honors_launch_opt_in(monkeypatch, dispatch):
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    query = torch.zeros(1, 4, 2, 4, requires_grad=True)
    frontend.attention(query, query, query)
    assert dispatch["chooser"][0]["deterministic"] is True
    assert dispatch["kernel"][0]["deterministic"] is True


def test_existing_backend_specific_true_remains_supported(monkeypatch, dispatch):
    monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC", raising=False)
    backend_kwargs = {"deterministic": True}
    invoke(backend_kwargs=backend_kwargs)
    assert dispatch["chooser"][0]["deterministic"] is False
    assert dispatch["kernel"][0]["deterministic"] is True
    assert backend_kwargs == {"deterministic": True}


def test_environment_is_resolved_each_call(monkeypatch, dispatch):
    for value in ("0", "1", "0"):
        monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", value)
        invoke()
    assert [row["deterministic"] for row in dispatch["chooser"]] == [False, True, False]
    assert [row["deterministic"] for row in dispatch["kernel"]] == [False, True, False]


def test_opt_in_is_kept_without_gradients(monkeypatch, dispatch):
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    invoke(requires_grad=False)
    assert dispatch["chooser"][0]["requires_grad"] is False
    assert dispatch["chooser"][0]["deterministic"] is True
    assert dispatch["kernel"][0]["deterministic"] is True


def test_opt_in_does_not_silently_fallback_if_backend_is_incompatible(monkeypatch, dispatch):
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    observed = []

    def reject(**kwargs):
        observed.append(kwargs["deterministic"])
        return None

    monkeypatch.setattr(frontend, "choose_backend", reject)
    with pytest.raises(ValueError, match="incompatible"):
        invoke()
    assert observed == [True]
    assert dispatch["kernel"] == []
