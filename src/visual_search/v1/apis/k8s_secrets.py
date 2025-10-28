# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import base64
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from src.visual_search.common.exceptions import SecretsNotFoundError


def get_k8s_secret(storage_secrets: str, namespace: str = "default") -> dict:
    """
    Retrieve a secret from Kubernetes and decode its data.

    :param storage_secrets: The name of the Kubernetes secret.
    :param namespace: The Kubernetes namespace where the secret is located.
    :return: A dictionary containing the decoded secret data.
    :raises SecretsNotFoundError: If the secret is not found.
    """
    # Load Kubernetes configuration
    config.load_incluster_config()

    # Create a Kubernetes API client
    v1 = client.CoreV1Api()

    try:
        # Retrieve the secret from Kubernetes
        secret = v1.read_namespaced_secret(name=storage_secrets, namespace=namespace)
    except ApiException as e:
        if e.status == 404:
            raise SecretsNotFoundError(f"Secret '{storage_secrets}' not found in Kubernetes.")
        else:
            raise

    # Decode the secret data
    aws_access_key_id = base64.b64decode(secret.data.get("aws_access_key_id", "")).decode('utf-8')
    aws_secret_access_key = base64.b64decode(secret.data.get("aws_secret_access_key", "")).decode('utf-8')
    aws_region = base64.b64decode(secret.data.get("aws_region", "")).decode('utf-8')
    endpoint_url = base64.b64decode(secret.data.get("endpoint_url", "")).decode('utf-8')
    
    # Decode session token if present (for temporary credentials)
    aws_session_token = None
    if secret.data.get("aws_session_token"):
        aws_session_token = base64.b64decode(secret.data.get("aws_session_token")).decode('utf-8')
        if not aws_session_token.strip():
            aws_session_token = None

    result = {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "aws_region": aws_region,
        "endpoint_url": endpoint_url,
    }

    # Only add session token if it exists and is non-empty
    # This supports both permanent credentials and temporary credentials
    if aws_session_token:
        result["aws_session_token"] = aws_session_token
    
    return result 