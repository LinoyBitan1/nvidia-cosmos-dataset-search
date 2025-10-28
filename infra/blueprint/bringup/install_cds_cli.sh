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

# Install CDS CLI from packaged source

echo "Installing CDS CLI from source..."

# Use BASH_SOURCE to get correct path when sourced (not executed)
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PACKAGED_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "Script directory: $SCRIPT_DIR"
echo "Package root: $PACKAGED_ROOT"

cd "$PACKAGED_ROOT" || { echo "ERROR: Cannot cd to $PACKAGED_ROOT"; exit 1; }

echo "Working directory: $(pwd)"
echo "Checking for pyproject.toml..."
ls -la pyproject.toml || (echo "ERROR: pyproject.toml not found!" && exit 1)

# Install uv if not already installed
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv (Python packaging tool)..."
    python3 -m pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv --python python3.10
fi

# Activate virtual environment
source .venv/bin/activate

# Install CDS CLI with client dependencies
echo "Installing CDS CLI with client dependencies..."
uv pip install -e ".[client]" --index-strategy unsafe-best-match

# Verify installation
if command -v cds >/dev/null 2>&1; then
    echo "CDS CLI installed successfully!"
    cds --version || echo "CDS version check skipped"
else
    echo "CDS CLI installation failed"
    exit 1
fi

### Create the config directory and migrate old vius config if exists
mkdir -p ~/.config/cds
if [ -f ~/.config/vius/config ]; then
    echo "Migrating old vius config to cds config..."
    cp ~/.config/vius/config ~/.config/cds/config
fi
