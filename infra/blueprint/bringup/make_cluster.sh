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

set -e

# Ensure AWS_REGION is set
if [ -z "$AWS_REGION" ]; then
  echo "Error: AWS_REGION is not set"
  exit 1
fi

if [ -z "$CLUSTER_NAME" ]; then
  echo "Error: CLUSTER_NAME is not set"
  exit 1
fi

if [ "$1" == '-y' ]; then
  echo "Creating EKS cluster $CLUSTER_NAME in $AWS_REGION"
else
  echo "Creating EKS cluster $CLUSTER_NAME in $AWS_REGION, continue?"
  read -p "Press enter to continue"
fi

envsubst < cvs.yaml > cvs-rendered.yaml

# Check if cluster already exists or if there are orphaned CloudFormation stacks
echo "Checking for existing cluster or orphaned CloudFormation stacks..."
if eksctl get cluster --name $CLUSTER_NAME --region $AWS_REGION >/dev/null 2>&1; then
  echo "Cluster $CLUSTER_NAME already exists, skipping creation"
else
  # Check for orphaned CloudFormation stacks
  if aws cloudformation describe-stacks --region $AWS_REGION --stack-name "eksctl-$CLUSTER_NAME-cluster" >/dev/null 2>&1; then
    echo "Found orphaned CloudFormation stack for cluster $CLUSTER_NAME, cleaning up..."
    eksctl delete cluster --name $CLUSTER_NAME --region $AWS_REGION --wait || true
    echo "Cleanup completed, proceeding with cluster creation..."
  fi
  
  eksctl create cluster -f cvs-rendered.yaml
fi

# Function to check cluster status
check_cluster() {
  eksctl get cluster --name $CLUSTER_NAME --region $AWS_REGION | grep "ACTIVE"
}

# Poll until cluster is ready
echo "Waiting for cluster to become active..."
while ! check_cluster; do
  sleep 30
  echo "Still waiting..."
done

# Update kubeconfig to point to the new cluster
aws eks update-kubeconfig --name $CLUSTER_NAME --region $AWS_REGION
