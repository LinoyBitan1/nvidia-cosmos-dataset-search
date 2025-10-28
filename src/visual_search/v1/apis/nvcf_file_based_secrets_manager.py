# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import os
from typing import Dict

from src.visual_search.common.exceptions import SecretsNotFoundError

class NVCFFileBasedSecretsManager:
    def __init__(self, nvfc_secrets_path: str = None):
        # Use SECRETS_FILE_PATH as the default NVCF Location
        self.nvfc_secrets_path = os.environ.get("NGC_SECRETS_FILE_PATH", "/var/secrets/secrets.json")

    def acquire_key(self) -> Dict[str, str] | None:
        # Check environment variables first
        access_key = os.environ.get("CVDS_S3_ACCESS_KEY")
        secret_key = os.environ.get("CVDS_S3_SECRET_KEY")
        aws_region = os.environ.get("AWS_REGION")
        if access_key and secret_key:
            return {"aws_access_key_id": access_key, "aws_secret_access_key": secret_key, "aws_region": aws_region}

        # Read from the NVCF Location
        try:
            with open(self.nvfc_secrets_path) as f:
                secrets = json.load(f)
                access_key = secrets.get("CVDS_S3_ACCESS_KEY")
                secret_key = secrets.get("CVDS_S3_SECRET_KEY")
                aws_region = secrets.get("AWS_REGION")
                if access_key and secret_key:
                    return {"aws_access_key_id": access_key, "aws_secret_access_key": secret_key, "aws_region": aws_region}
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        return None

    def get_secrets(self) -> Dict[str, str]:
        secret_value = self.acquire_key()
        if secret_value is not None:
            return secret_value
        else:
            raise SecretsNotFoundError("NVCF Secrets not found!")
