# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit experiment selection avoids importing unrelated runtime integrations."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from cosmos_framework.configs.toml_config.sft_config import (
    SFTExperimentConfig,
    load_experiment_from_toml,
)
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides


def test_experiment_module_is_metadata_not_a_hydra_job_override():
    raw = {"job": {"task": "vfm", "experiment": "paired", "experiment_module": "project.recipes"}}
    config = SFTExperimentConfig.model_validate(raw)
    assert config.job.experiment_module == "project.recipes"
    overrides = build_hydra_overrides(raw)
    assert "experiment=paired" in overrides
    assert not any("experiment_module" in item for item in overrides)
    with pytest.raises(ValidationError):
        SFTExperimentConfig.model_validate({"job": {**raw["job"], "experiment_modul": "typo"}})


def test_loader_passes_only_selected_module_to_existing_config_interface(tmp_path, monkeypatch):
    from cosmos_framework.utils import config as config_module

    loader = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(config_module, "load_config", loader)
    recipe = tmp_path / "paired.toml"
    recipe.write_text('[job]\nexperiment="paired"\nexperiment_module="project.recipes"\n', encoding="utf-8")
    load_experiment_from_toml(recipe)
    assert loader.call_args.kwargs == {"experiment_module": "project.recipes"}
    assert not any("experiment_module=" in item for item in loader.call_args.args[1])


def test_vfm_registers_only_the_explicit_module(monkeypatch):
    from cosmos_framework.configs.base import config as base_config

    imported = []
    monkeypatch.setattr(base_config.importlib, "import_module", imported.append)
    base_config._register_experiments("project.paired_recipe")
    assert imported == ["project.paired_recipe"]


def test_omitted_module_preserves_shipped_catalog(monkeypatch):
    from cosmos_framework.configs.base import config as base_config

    imported = []
    monkeypatch.setattr(base_config.importlib, "import_module", imported.append)
    base_config._register_experiments()
    assert len(imported) == 8
    assert "cosmos_framework.configs.base.experiment.sft.vision_sft_nano" in imported
    assert any("action_policy_droid_nano" in item for item in imported)
