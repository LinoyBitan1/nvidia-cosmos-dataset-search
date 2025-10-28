# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Generic Python entrypoint for triton server call."""

import subprocess
import sys


def main() -> None:
    cmd = ["tritonserver"] + sys.argv[1:]
    print("Running command", cmd)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main() 