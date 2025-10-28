# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import time
from typing import Annotated

from fastapi import Depends, HTTPException

from ...logger import logger
from ..db_models import create_collection, get_collections
from ..models import Collection, make_path_with_id_restrictions
from ..pipelines import EnabledPipeline, enabled_pipelines, get_all_pipelines


# GetCollection is a FastAPI dependency that will perform Collection lookup
async def connected_collection_handler(
    *,
    collection_id: str = make_path_with_id_restrictions(
        "Identifier for the collection."
    ),
) -> Collection:
    t0 = time.time()
    logger.debug(f"connected_collection_handler    start    {collection_id}")
    stored_collection = get_collections(collection_id)
    t1 = time.time()
    logger.debug(f"connected_collection_handler    t1    {collection_id}    {t1 - t0}s")
    if not stored_collection:
        raise HTTPException(
            status_code=404, detail=f"Collection {collection_id} does not exist"
        )

    t2 = time.time()
    logger.debug(
        f"connected_collection_handler    end    {collection_id}    {t2 - t1}s"
    )
    return stored_collection


GetCollection = Depends(connected_collection_handler)


# GetCollectionModel is a FastAPI dependency that will perform Collection lookup
# This endpoint returns a read only copy of the collection
async def detached_collection_handler(
    *,
    collection_id: str = make_path_with_id_restrictions(
        "Identifier for the collection."
    ),
) -> Collection:
    t0 = time.time()
    logger.debug(f"detached_collection_handler    start    {collection_id}")
    stored_collection = get_collections(collection_id)
    t1 = time.time()
    logger.debug(f"detached_collection_handler    t1    {collection_id}    {t1 - t0}s")
    if not stored_collection:
        raise HTTPException(
            status_code=404, detail=f"Collection {collection_id} does not exist"
        )

    t2 = time.time()
    logger.debug(f"detached_collection_handler    end    {collection_id}    {t2 - t1}s")
    copied_collection = Collection(**stored_collection.dict())
    return copied_collection


GetCollectionModel = Depends(detached_collection_handler)


# This makes it possible to delete collections attached to pipelines that are no long enabled.
def maybe_pipeline_handler(
    collection: Annotated[Collection, GetCollectionModel]
) -> EnabledPipeline | None:
    try:
        return enabled_pipelines[collection.pipeline]
    except KeyError:
        return None


def pipeline_handler(
    collection: Annotated[Collection, GetCollectionModel]
) -> EnabledPipeline:
    logger.debug("pipeline_handler    start")
    t0 = time.time()
    available_pipeline_keys = list(enabled_pipelines.keys())
    all_pipelines = get_all_pipelines()
    t1 = time.time()
    logger.debug(f"pipeline_handler    end    {t1 - t0}s")
    try:
        pipeline = all_pipelines[collection.pipeline]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Pipeline {collection.pipeline} does not exist. "
                "Available Pipelines: "
                f"{[key for key in available_pipeline_keys]}"
            ),
        )

    if not pipeline.enabled:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Pipeline {collection.pipeline} is disabled "
                "and cannot be used for any associated collection operations. "
                "Available Pipelines: "
                f"{[key for key in available_pipeline_keys]}"
            ),
        )

    enabled_pipeline = enabled_pipelines[collection.pipeline]

    return enabled_pipeline


GetPipeline = Depends(pipeline_handler)
MaybeGetPipeline = Depends(maybe_pipeline_handler)
