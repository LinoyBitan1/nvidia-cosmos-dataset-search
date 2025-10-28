# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Haystack components for manipulating list inputs."""

import itertools
from typing import Any, Dict, List

from haystack import component
from haystack.core.component.types import Variadic
from haystack.utils import deserialize_type

from src.haystack.serializer import SerializerMixin


@component
class Concatenate(SerializerMixin):
    """Concatenates multiple variatic inputs."""

    def __init__(self, type_alias: str) -> None:
        self.type_alias = type_alias
        type_ = deserialize_type(type_alias)
        component.set_input_types(self, input=Variadic[List[type_]])  # type: ignore
        component.set_output_types(self, output=List[type_])  # type: ignore

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        concatenated = []
        index: List[int] = []
        for i, inp in enumerate(kwargs["input"]):
            concatenated.extend(inp)
            index.extend(itertools.repeat(i, len(inp)))
        return {"output": concatenated, "index": index}


@component
class Flatten(SerializerMixin):
    """Flatten lists."""

    def __init__(self, type_alias: str) -> None:
        self.type_alias = type_alias
        type_ = deserialize_type(type_alias)
        component.set_input_types(self, input=List[List[type_]])  # type: ignore
        component.set_output_types(self, output=List[type_])  # type: ignore

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        flattened = []
        index: List[int] = []
        for i, inp in enumerate(kwargs["input"]):
            flattened.extend(inp)
            index.extend(itertools.repeat(i, len(inp)))
        return {"output": flattened, "index": index}
