# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os

from cryptography.fernet import Fernet


def generate_secret_key():
    # Convert to base64 for easier storage/transmission
    encryption_key = Fernet.generate_key().decode()
    return encryption_key


def set_env_variable(key_name="SECRET_ENCRYPTION_KEY"):
    secret_key = generate_secret_key()

    # Set environment variable for current process
    os.environ[key_name] = secret_key

    # Print instructions for permanent setup
    print(f"export {key_name}='{secret_key}'")
    return secret_key


if __name__ == "__main__":
    set_env_variable()
