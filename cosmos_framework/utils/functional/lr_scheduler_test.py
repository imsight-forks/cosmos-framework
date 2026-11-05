# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Continuation keeps the terminal schedule value after its original budget."""

import unittest

from cosmos_framework.utils.functional.lr_scheduler import LambdaLinearScheduler, LambdaWarmUpCosineScheduler


class SchedulerContinuationTest(unittest.TestCase):
    def test_completed_schedule_holds_floor(self):
        for scheduler_type in (LambdaLinearScheduler, LambdaWarmUpCosineScheduler):
            for floor in (0.1, 1.0):
                schedule = scheduler_type(
                    warm_up_steps=[0], f_min=[floor], f_max=[1.0], f_start=[1.0], cycle_lengths=[300]
                )
                for step in (300, 301, 600, 10**12):
                    with self.subTest(scheduler=scheduler_type, floor=floor, step=step):
                        self.assertAlmostEqual(schedule(step), floor)
