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

# Define variables
INGRESS_NAME="simple-ingress"

# Function to get the hostname from the Ingress
get_ingress_hostname() {
    hostname=$(kubectl get ingress $INGRESS_NAME -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    echo $hostname
}

# Main loop to wait for the hostname
while true; do
    echo "Checking for hostname in Ingress $INGRESS_NAME..."
    hostname=$(get_ingress_hostname)

    if [[ -n "$hostname" ]]; then
        echo "Hostname found: $hostname"
        break
    else
        echo "Hostname not yet available. Waiting for 30 seconds..."
        sleep 30
    fi
done

echo "Ingress is ready with hostname: $hostname"
