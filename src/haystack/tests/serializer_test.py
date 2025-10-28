# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for generic haystack serializer."""

from typing import Dict

import pytest
from haystack import component

from src.haystack.serializer import INIT_PARAMETERS_KEY, TYPE_KEY, SerializerMixin


def test_serializer_mixin() -> None:
    """Tests serializer mixin for arbitrary haystack component classes."""

    @component
    class Correct(SerializerMixin):
        def __init__(self, a: float, b: int, c: str) -> None:
            self.a = a
            self.b = b
            self.c = c

        @component.output_types(output=int)
        def run(self, data: int) -> Dict[str, int]:
            return {"output": data}

    a = Correct(1.0, 2, "a")
    assert a.a == 1.0
    assert a.b == 2
    assert a.c == "a"

    a_dict = a.to_dict()
    assert a_dict[TYPE_KEY] == "serializer_test.Correct"
    assert a_dict[INIT_PARAMETERS_KEY] == {"a": 1.0, "b": 2, "c": "a"}

    new_a = Correct.from_dict(a_dict)
    assert new_a.a == 1.0
    assert new_a.b == 2
    assert new_a.c == "a"


def test_serializer_mixin_empty() -> None:
    """Test case when there is no init specified."""

    @component
    class Empty(SerializerMixin):
        @component.output_types(output=int)
        def run(self, data: int) -> Dict[str, int]:
            return {"output": data}

    a = Empty()
    a_dict = a.to_dict()
    assert a_dict[TYPE_KEY] == "serializer_test.Empty"
    assert a_dict[INIT_PARAMETERS_KEY] == {}

    new_a = Empty.from_dict(a_dict)
    assert new_a


def test_serializer_multiple_inheritance() -> None:
    """Test case when the component has multiple parents."""

    class SomeOtherMixin:
        def __init__(self, foo: int, bar: str = "localhost:8000") -> None:
            self.foo = foo
            self.bar = bar

    @component
    class Composite(SerializerMixin, SomeOtherMixin):
        @component.output_types(output=int)
        def run(self, data: int) -> Dict[str, int]:
            return {"output": data}

    a = Composite(foo=2, bar="localhost:9000")
    assert a.foo == 2
    assert a.bar == "localhost:9000"

    a_dict = a.to_dict()
    assert a_dict[TYPE_KEY] == "serializer_test.Composite"
    assert a_dict[INIT_PARAMETERS_KEY] == {"foo": 2, "bar": "localhost:9000"}

    a_new = Composite.from_dict(a_dict)
    assert a_new.foo == 2
    assert a_new.bar == "localhost:9000"


def test_raise_exception() -> None:
    """Test that intelligible error is raised with incomplete init method."""

    @component
    class Incorrect(SerializerMixin):
        def __init__(self, a: float, b: int) -> None:
            # missing assigning `b` to `self`
            self.a = a

        @component.output_types(output=int)
        def run(self, data: int) -> Dict[str, int]:
            return {"output": data}

    new_a = Incorrect.from_dict(
        {TYPE_KEY: "serializer_test.Incorrect", INIT_PARAMETERS_KEY: {"a": 1.0, "b": 2}}
    )
    assert new_a.a == 1.0
    assert getattr(new_a, "b", None) is None

    a = Incorrect(1.0, 2)
    assert a.a == 1.0
    assert getattr(a, "b", None) is None

    with pytest.raises(
        AttributeError,
        match="`b` attribute is missing from `self` in `Incorrect`. Please assign to `self` for serializer to work.",
    ):
        a.to_dict()
