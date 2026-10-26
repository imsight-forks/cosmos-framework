# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Checkpoint the counter-based conditioning policy with the native data cursor."""

from cosmos_framework.callbacks.cosmos_dataloader_state import CosmosDataLoaderStateCallback


class SourceViewModeStateCallback(CosmosDataLoaderStateCallback):
    def __init__(self, execution_fingerprint: str, **kwargs):
        super().__init__(**kwargs)
        self.execution_fingerprint = execution_fingerprint
        self.next_exposure = 0

    def _update_state_from_batch(self, data_batch):
        super()._update_state_from_batch(data_batch)
        fingerprints = data_batch.get("conditioning_execution_fingerprint", [])
        if any(value != self.execution_fingerprint for value in fingerprints):
            raise ValueError("Conditioning execution identity differs from checkpoint policy")
        exposures = data_batch.get("conditioning_exposure", [])
        for value in exposures:
            counter = int(value.item()) if hasattr(value, "item") else int(value)
            self.next_exposure = max(self.next_exposure, counter + 1)

    def state_dict(self):
        return {
            "format": "source-view-mode-cursor-v1",
            "execution_fingerprint": self.execution_fingerprint,
            "next_exposure": self.next_exposure,
            "workers": super().state_dict(),
        }

    def load_state_dict(self, state):
        if (
            state.get("format") != "source-view-mode-cursor-v1"
            or state.get("execution_fingerprint") != self.execution_fingerprint
        ):
            raise ValueError("Incompatible source-view conditioning checkpoint policy")
        counter = state.get("next_exposure")
        if type(counter) is not int or counter < 0:
            raise ValueError("Invalid conditioning exposure counter")
        self.next_exposure = counter
        super().load_state_dict(state["workers"])
