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
import unittest
from unittest.mock import mock_open, patch

from src.visual_search.v1.apis.nvcf_file_based_secrets_manager import (
    NVCFFileBasedSecretsManager,
    SecretsNotFoundError,
)


class TestNVCFFileBasedSecretsManager(unittest.TestCase):

    def test_initialization(self):
        manager = NVCFFileBasedSecretsManager()
        self.assertEqual(manager.nvfc_secrets_path, "/var/secrets/secrets.json")

    def test_acquire_key_from_env(self):
        with patch.dict(
            os.environ,
            {
                "CVDS_S3_ACCESS_KEY": "test_access_key",
                "CVDS_S3_SECRET_KEY": "test_secret_key",
            },
        ):
            manager = NVCFFileBasedSecretsManager()
            secrets = manager.acquire_key()
            self.assertEqual(secrets["aws_access_key_id"], "test_access_key")
            self.assertEqual(secrets["aws_secret_access_key"], "test_secret_key")

    def test_acquire_key_from_nvfc_location(self):
        mock_data = json.dumps(
            {
                "CVDS_S3_ACCESS_KEY": "test_access_key",
                "CVDS_S3_SECRET_KEY": "test_secret_key",
            }
        )
        with patch("builtins.open", mock_open(read_data=mock_data)):
            manager = NVCFFileBasedSecretsManager()
            secrets = manager.acquire_key()
            self.assertEqual(secrets["aws_access_key_id"], "test_access_key")
            self.assertEqual(secrets["aws_secret_access_key"], "test_secret_key")

    def test_key_not_found(self):
        with patch("builtins.open", mock_open(read_data="{}")):
            manager = NVCFFileBasedSecretsManager()
            self.assertIsNone(manager.acquire_key())

    def test_missing_file(self):
        with patch("builtins.open", side_effect=FileNotFoundError()):
            manager = NVCFFileBasedSecretsManager()
            self.assertIsNone(manager.acquire_key())

    def test_invalid_json(self):
        with patch("builtins.open", mock_open(read_data="invalid_json")):
            manager = NVCFFileBasedSecretsManager()
            self.assertIsNone(manager.acquire_key())

    def test_secrets_not_found_error(self):
        manager = NVCFFileBasedSecretsManager()
        # Mock the acquire_key method to return None, simulating a missing secret
        with patch.object(manager, 'acquire_key', return_value=None):
            with self.assertRaises(SecretsNotFoundError):
                manager.get_secrets()


if __name__ == "__main__":
    unittest.main() 