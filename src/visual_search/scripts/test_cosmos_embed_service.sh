#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# First, encode the video file
video_b64=$(base64 /home/svenkatanara/projects/WIP/CVML-840/cvds/src/visual_search/scripts/sample_data/video1.mp4)

curl -X 'POST' \
  'http://0.0.0.0:9000/v1/embeddings' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  --data @- <<EOF
{
  "input": [
    "data:video/mp4;base64,${video_b64}"
  ],
  "encoding_format": "float",
  "model": "nvidia/cosmos-embed1",
  "request_type": "query"
}
EOF