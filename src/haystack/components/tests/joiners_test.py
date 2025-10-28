# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for Haystack joiners."""

import pytest

from src.haystack.components.joiners import Concatenate, Flatten


@pytest.mark.parametrize(
    "type_alias", ["typing.List[float]", "float", "haystack.Document"]
)
def test_serialization_concatenate(type_alias: str) -> None:
    component = Concatenate(type_alias=type_alias)
    serialized_dict = component.to_dict()
    new_component = Concatenate.from_dict(serialized_dict)
    assert new_component.to_dict() == serialized_dict


@pytest.mark.parametrize("type_alias", ["typing.List[float]", "typing.Sequence[float]"])
def test_serialization_flatten(type_alias: str) -> None:
    component = Flatten(type_alias=type_alias)
    serialized_dict = component.to_dict()
    new_component = Flatten.from_dict(serialized_dict)
    assert new_component.to_dict() == serialized_dict
