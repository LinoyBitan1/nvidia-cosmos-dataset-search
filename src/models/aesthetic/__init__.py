# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Module __init__ for src.models.aesthetic."""

from src.models.aesthetic.dataset import make_loader  # noqa: F401
from src.models.aesthetic.model import AestheticPredictor  # noqa: F401
from src.models.aesthetic.predict import load_model  # noqa: F401
