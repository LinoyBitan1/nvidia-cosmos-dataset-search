# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from typing import ClassVar, Dict, List

from pydantic import BaseModel


class Settings(BaseModel):
    debug: bool = False
    cors_allowed_domains: List[str] = [
        "http://localhost:8080",
        "https://localhost:8080",
        "http://localhost:3000",
        "https://ui.dev.vius.cv.nvidia.com",
        "https://ui.staging.vius.cv.nvidia.com",
        "https://ui.prod.vius.cv.nvidia.com",
        "https://ui.vius.cv.nvidia.com",
        "*",  # Waabi # TODO - this is a security hole. For cvds_blueprint, we need to modify our ingress to inject localhost:8080 as the client header.
    ]
    openapi_tags: ClassVar[List[Dict[str, str]]] = [
        {"name": "Collections", "description": "Operations related to collections."},
        {
            "name": "Document Indexing",
            "description": "Operations related to documents.",
        },
        {"name": "Health", "description": "Operations related to health."},
        {
            "name": "Retrieval",
            "description": "Operations related to document retrieval.",
        },
    ]
    if os.getenv("EXPOSE_BACKFILL_ENDPOINT"):
        openapi_tags.append(
            {
                "name": "Backfill",
                "description": "Operations related to backfilling sessions.",
            }
        )


settings = Settings()
