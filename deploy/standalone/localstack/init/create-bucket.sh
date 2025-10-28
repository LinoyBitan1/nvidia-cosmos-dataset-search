#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set -euo pipefail

# Ensure the S3 bucket used by Milvus exists in Localstack and has permissive CORS for local dev

BUCKET_NAME="${BUCKET_NAME:-cosmos-test-bucket}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://localhost:4566}"

aws_cmd() {
  if command -v awslocal >/dev/null 2>&1; then
    awslocal "$@"
  else
    aws --endpoint-url "${ENDPOINT_URL}" "$@"
  fi
}

echo "[localstack:init] ensuring bucket '${BUCKET_NAME}' in region '${REGION}'"

if aws_cmd s3api head-bucket --bucket "${BUCKET_NAME}" 2>/dev/null; then
  echo "[localstack:init] bucket '${BUCKET_NAME}' already exists"
else
  if [ "${REGION}" = "us-east-1" ]; then
    aws_cmd s3api create-bucket --bucket "${BUCKET_NAME}"
  else
    aws_cmd s3api create-bucket --bucket "${BUCKET_NAME}" --create-bucket-configuration LocationConstraint="${REGION}"
  fi
  echo "[localstack:init] created bucket '${BUCKET_NAME}'"
fi

# Apply a permissive CORS policy for local development
aws_cmd s3api put-bucket-cors \
  --bucket "${BUCKET_NAME}" \
  --cors-configuration '{
    "CORSRules": [{
      "AllowedOrigins": ["http://localhost:8080", "*"],
      "AllowedMethods": ["GET", "PUT", "POST", "DELETE", "HEAD"],
      "AllowedHeaders": ["*"],
      "ExposeHeaders": ["ETag"],
      "MaxAgeSeconds": 3000
    }]
  }' || true

echo "[localstack:init] bucket '${BUCKET_NAME}' is ready"


