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

# Namespace to check pods in
namespace="default"

# Function to check pod statuses
check_pods() {
    all_pods_running=0  # Assume all pods are running initially

    # Get all pods in the specified namespace
    pods=$(kubectl get pods -n "$namespace" -o jsonpath='{.items[*].metadata.name}')

    for pod in $pods; do
        # Get the status of the pod
        status=$(kubectl get pod "$pod" -n "$namespace" -o jsonpath='{.status.phase}')

        # Check if the pod is not in Running state
        if [ "$status" != "Running" ]; then
            all_pods_running=1  # Set to 1 if any pod is not running
            echo "Pod $pod is in $status state.\n"
        else
            echo -n "."
        fi
    done

        echo -e "\n"
    return $all_pods_running  # Return the status
}

# Main loop to wait for all pods to be in the Running state
while true; do
    check_pods
    result=$?  # Capture the return value of check_pods

    if [ $result -eq 0 ]; then
        echo "All pods are in the Running state. Process complete."
        break
    else
        echo "Waiting for all pods to be in the Running state..."
        sleep 10 # Wait for 10 seconds before checking again
    fi
done
