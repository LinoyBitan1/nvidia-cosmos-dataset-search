# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from src.visual_search.common.models import PipelineMode, PipelinesResponse
from src.visual_search.common.pipelines import draw_pipeline as draw_haystack_pipeline
from src.visual_search.common.pipelines import get_all_pipelines

router = APIRouter()


@router.get(
    "/pipelines",
    response_model=PipelinesResponse,
    responses={
        200: {"description": "Successful pipeline listing."},
    },
    tags=["Pipelines"],
    summary="List pipelines",
    description="""
        Returns a list of pipelines supported by the service.
        Take note of the `enabled` and `missing` attributes in
        the response. `enabled` will indicate whether or not a
        pipeline is usable. If `enabled` is `false`, that means
        that the service was not configured with the correct
        set of ENV vars. The `missing` array will list off the
        ENV vars that need to be passed into the service to
        enable the pipeline.
    """,
    operation_id="get_pipelines",
)
def get_pipelines() -> PipelinesResponse:
    """Retrieve all collections."""
    pipelines = get_all_pipelines()
    return PipelinesResponse(pipelines=list(pipelines.values()))


@router.get(
    "/pipelines/draw/{name}",
    responses={
        200: {"description": "Draw pipeline, either in index or query mode."},
        404: {"description": "Pipeline does not exist or is disabled."},
    },
    tags=["Pipelines"],
    summary="Draw pipeline",
    description="""Draws pipeline processing graph in index or query mode.""",
    operation_id="draw_pipeline",
)
def draw_pipeline(name: str, mode: PipelineMode = PipelineMode.INDEX) -> Response:
    """Draw pipeline."""
    try:
        image_bytes = draw_haystack_pipeline(name, mode.value)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(f"Requested pipeline {name} does not exist or is disabled!"),
        )
    return Response(image_bytes, media_type="image/png")
