# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Uvicorn Python wrapper."""

import sys

import uvicorn

if __name__ == "__main__":
    print("Wrapper received arguments:", sys.argv)
    file = sys.argv.pop(1)
    function = sys.argv.pop(1)
    stem, extension = file.split(".")
    assert extension == "py", f"File should be a Python (.py) file! Got {extension}."
    module_import_path = stem.replace("/", ".")
    app_name = module_import_path + ":" + function
    print(f"Starting uvicorn app `{app_name}`")
    sys.argv.insert(1, app_name)
    uvicorn.main() 