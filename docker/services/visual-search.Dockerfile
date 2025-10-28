# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Visual Search FastAPI service using python-base
FROM python-base:latest

# Switch to appuser before copying
USER appuser

# Copy application code
COPY --chown=appuser:appuser src/triton /app/src/triton
COPY --chown=appuser:appuser src/models /app/src/models
COPY --chown=appuser:appuser src/haystack /app/src/haystack
COPY --chown=appuser:appuser src/wrappers /app/src/wrappers
COPY --chown=appuser:appuser src/visual_search /app/src/visual_search
COPY --chown=appuser:appuser ./LICENSE /app/LICENSE
COPY --chown=appuser:appuser ./OSS_SOURCES.txt /app/OSS_SOURCES.txt

# Expose port (default 8000, configurable via GUNICORN_PORT)
EXPOSE 8000

# Health check - use GUNICORN_PORT environment variable or default to 8000
# Increased timeout to 30s to handle slow Milvus collection metadata operations
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${GUNICORN_PORT:-8000}/health || exit 1

# Set Python path and run the application using gunicorn wrapper
ENV PYTHONPATH="/app"
CMD ["python", "/app/src/wrappers/gunicorn_wrapper.py", "src/visual_search/main.py", "app"]
