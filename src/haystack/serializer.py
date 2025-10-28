# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Serializer utility for haystack components."""


import inspect
from typing import Any, Dict, Final

from haystack.core.serialization import default_from_dict, default_to_dict

TYPE_KEY: Final = "type"
INIT_PARAMETERS_KEY: Final = "init_parameters"


class SerializerMixin:
    """Serializer mixin that infers serialization parameters from `__init__` signature."""

    def to_dict(self) -> Dict[str, Any]:
        """Default `to_dict` method."""
        init_method = self.__class__.__init__
        signature = inspect.signature(init_method)
        param_names = [
            param.name
            for param in signature.parameters.values()
            if param.name not in ("self", "args", "kwargs")
        ]
        param_dict = dict()
        for param in param_names:
            if not hasattr(self, param):
                raise AttributeError(
                    f"`{param}` attribute is missing from `self` in `{self.__class__.__name__}`. "
                    "Please assign to `self` for serializer to work."
                )
            param_dict[param] = getattr(self, param)
        return default_to_dict(self, **param_dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """Default `from_dict` method."""
        return default_from_dict(cls, data)
