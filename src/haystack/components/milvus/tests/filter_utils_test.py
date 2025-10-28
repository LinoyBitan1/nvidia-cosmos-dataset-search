# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for filter utilities of milvus document store in Haystack."""

from src.haystack.components.milvus.filter_utils import (
    AndOperation,
    EqOperation,
    GteOperation,
    GtOperation,
    InOperation,
    LteOperation,
    LtOperation,
    NeOperation,
    NinOperation,
    NotOperation,
    OrOperation,
)


def test_filters() -> None:
    assert (
        InOperation("dummy_field", ["1.0"]).convert_to_milvus()
        == '(dummy_field in ["1.0"])'
    )
    assert (
        NinOperation("dummy_field", ["1.0"]).convert_to_milvus()
        == '(dummy_field not in ["1.0"])'
    )
    assert (
        AndOperation(
            [EqOperation("dummy_field1", "1.0"), EqOperation("dummy_field2", "2.0")]
        ).convert_to_milvus()
        == '((dummy_field1 == "1.0") and (dummy_field2 == "2.0"))'
    )
    assert (
        AndOperation(
            [
                GteOperation("dummy_field1", "1.0"),
                LteOperation("dummy_field2", "2.0"),
                LtOperation("dummy_field3", "3.0"),
            ]
        ).convert_to_milvus()
        == '((dummy_field1 >= "1.0") and (dummy_field2 <= "2.0") and (dummy_field3 < "3.0"))'
    )
    assert (
        NotOperation(
            [GtOperation("dummy_field1", "1.0"), NeOperation("dummy_field2", "2.0")]
        ).convert_to_milvus()
        == '((dummy_field1 <= "1.0") or (dummy_field2 == "2.0"))'
    )
    assert (
        OrOperation(
            [GtOperation("dummy_field1", "1.0"), NeOperation("dummy_field2", "2.0")]
        ).convert_to_milvus()
        == '((dummy_field1 > "1.0") or (dummy_field2 != "2.0"))'
    )
