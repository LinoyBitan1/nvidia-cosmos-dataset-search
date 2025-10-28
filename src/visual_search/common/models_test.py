# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any, Dict, List, Tuple

from src.visual_search.common.models import is_field_included


def test_is_field_included():
    test_cases: List[Tuple[Dict[str, Any], bool]] = [
        ({"field": "session_id", "operator": "in", "value": ["a", "b"]}, True),
        ({"field": "patch_id", "operator": "==", "value": "1"}, False),
        # allowed: session_id == ? and other = ?
        (
            {
                "operator": "AND",
                "conditions": [
                    {"field": "session_id", "operator": "in", "value": ["a", "b"]},
                    {"field": "patch_id", "operator": "==", "value": "1"},
                ],
            },
            True,
        ),
        # not allowed: other_1 == ? and other_2 = ?
        (
            {
                "operator": "AND",
                "conditions": [
                    {"field": "camera", "operator": "in", "value": ["a", "b"]},
                    {"field": "patch_id", "operator": "==", "value": "1"},
                ],
            },
            False,
        ),
        # not allowed: session_id == ? or other = ?
        (
            {
                "operator": "OR",
                "conditions": [
                    {"field": "session_id", "operator": "in", "value": ["a", "b"]},
                    {
                        "field": "camera_id",
                        "operator": "in",
                        "value": ["camera_front_wide_120fov"],
                    },
                ],
            },
            False,
        ),
        # allowed: session_id == ? or (session_id == ? and other = ?)
        (
            {
                "operator": "OR",
                "conditions": [
                    {"field": "session_id", "operator": "==", "value": "a"},
                    {
                        "operator": "AND",
                        "conditions": [
                            {"field": "session_id", "operator": "==", "value": "b"},
                            {
                                "field": "camera_id",
                                "operator": "in",
                                "value": ["camera_front_wide_120fov"],
                            },
                        ],
                    },
                ],
            },
            True,
        ),
    ]
    for test_filter, result in test_cases:
        assert is_field_included(test_filter, "session_id") == result
