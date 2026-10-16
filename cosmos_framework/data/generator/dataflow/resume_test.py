# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpoint->restart resume parity for CosmosDataLoader(MapDistributor) using
CosmosDataLoaderStateCallback. Single process, num_workers=0."""

from __future__ import annotations

import torch

from cosmos_framework.callbacks.cosmos_dataloader_state import CosmosDataLoaderStateCallback
from cosmos_framework.data.generator.dataflow import (
    CosmosDataLoader,
    IdentityProcessor,
    MapDistributor,
)


class _IdDS(torch.utils.data.Dataset):
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {"id": torch.tensor(idx)}


def _build(seed=0):
    return CosmosDataLoader(
        distributor=MapDistributor(_IdDS(20), shuffle=False, seed=seed),
        processor=IdentityProcessor(),
        batch_size=1,
        num_workers=0,
        seed=seed,
    )


def test_resume_continues_without_dup_or_skip():
    cb = CosmosDataLoaderStateCallback()
    loader = _build()
    it = iter(loader)
    seen_ids = []
    for _ in range(5):
        b = next(it)
        cb._update_state_from_batch(b)
        seen_ids.append(b["id"].item())
    assert seen_ids == [0, 1, 2, 3, 4]

    state = cb.state_dict()
    assert state[0]["index"] == 4
    cb2 = CosmosDataLoaderStateCallback()
    cb2.load_state_dict(state)

    loader2 = _build()
    it2 = iter(loader2)  # one iterator: env-var fast-forward happens once, then continues
    resumed = [next(it2)["id"].item() for _ in range(3)]
    assert resumed == [5, 6, 7]


def test_restart_iterator_preserves_trainer_rng_and_next_sample(monkeypatch):
    """Cursor restoration must preserve the next diffusion-noise draw too."""
    torch.manual_seed(132)
    callback = CosmosDataLoaderStateCallback()
    iterator = iter(_build(seed=42))
    for _ in range(5):
        batch = next(iterator)
        callback._update_state_from_batch(batch)
        torch.randn(11)  # The trainer consumes noise between loader draws.
    cursor = callback.state_dict()
    rng = torch.get_rng_state()
    expected_id = next(iterator)["id"].item()
    expected_noise = torch.randn(11)

    restored_callback = CosmosDataLoaderStateCallback()
    # Scope the callback's environment writes to this test.
    monkeypatch.setenv("COSMOS_DL_STATE_WORKER_0_EPOCH", "0")
    monkeypatch.setenv("COSMOS_DL_STATE_WORKER_0_INDEX", "-1")
    restored_callback.load_state_dict(cursor)
    torch.set_rng_state(rng)
    restarted = iter(_build(seed=42))
    assert next(restarted)["id"].item() == expected_id
    assert torch.equal(torch.randn(11), expected_noise)
