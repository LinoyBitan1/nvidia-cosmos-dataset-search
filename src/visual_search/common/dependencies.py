# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from anyio import CapacityLimiter
from anyio.lowlevel import RunVar
from fastapi import FastAPI

from src.visual_search.common.models import StoredIngestProcess  # noqa: F401
from src.visual_search.common.pipelines import load_pipelines


@asynccontextmanager
async def app_dependencies(app: FastAPI) -> AsyncIterator[None]:
    # The elastic_transport client throws and logs errors about not
    # being able to connect, this is expected during the load phase
    # so we temporarily silence this logger.
    l1 = logging.getLogger("elastic_transport.transport").getEffectiveLevel()
    l2 = logging.getLogger("elastic_transport.node_pool").getEffectiveLevel()

    logging.getLogger("elastic_transport.transport").setLevel(logging.CRITICAL)
    logging.getLogger("elastic_transport.node_pool").setLevel(logging.CRITICAL)

    # Load the pipelines
    await load_pipelines()

    logging.getLogger("elastic_transport.transport").setLevel(l1)
    logging.getLogger("elastic_transport.node_pool").setLevel(l2)

    # This line changes the default number of threads in the AnyIO thread pool.
    # Increasing it allows for better concurrency in this heavy IO service while
    # using syncronous endpoint handlers
    RunVar("_default_thread_limiter").set(CapacityLimiter(100))  # type: ignore
    yield
