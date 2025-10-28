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

# Function to check addon status
check_addon_status() {
    echo "Checking the status of the aws-ebs-csi-driver addon..."
    addon_info=$(eksctl get addon --name aws-ebs-csi-driver --cluster $CLUSTER_NAME --region $AWS_REGION --output json)
    status=$(echo "$addon_info" | awk -F'"' '/Status/{print $4}')
    echo "Current status: $status"
}

# Main loop to wait for addon to become ACTIVE
while true; do
    check_addon_status

    if [[ "$status" == "ACTIVE" ]]; then
        echo "The aws-ebs-csi-driver addon is now ACTIVE."
        break
    elif [[ "$status" == "CREATING" ]]; then
        echo "The addon is still being created. Waiting for 30 seconds before checking again..."
        sleep 30
    else
        echo "There was an error or the addon is in an unexpected state: $status"
        exit 1
    fi
done
