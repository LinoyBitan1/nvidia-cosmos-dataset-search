# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# 
# Python Base Dockerfile

FROM nvcr.io/nvidia/base/ubuntu:22.04_20240212


ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies and security updates
# Note: Explicitly upgrading libtiff5 to fix CVE-2025-9900 (write-what-where vulnerability)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    git-lfs \
    ffmpeg \
    libmagic1 \
    && apt-get upgrade -y libtiff5 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN pip install uv

# Create user first
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# Create /app directory owned by appuser  
RUN mkdir -p /app && chown appuser:appuser /app

# Set working directory
WORKDIR /app

# Switch to appuser
USER appuser

# Now everything runs as appuser in appuser-owned directory

COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./

# Install dependencies WITHOUT client extras (no Ray, Fire, Rich, python-magic)
# Backend services don't need the CDS client CLI dependencies
RUN echo "Installing Python dependencies (excluding client extras)..." && \
    uv sync --frozen --no-cache --index-strategy unsafe-best-match

# Activate the uv virtual environment for all Python commands
ENV PATH="/app/.venv/bin:$PATH"

RUN python --version && python -c "import sys; print(sys.executable)"