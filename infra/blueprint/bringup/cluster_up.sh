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

## Cosmos Video Search (CVS)

# Resolve the directory of this script:
# Push into the script directory so that all relative paths work:
# Ensure we pop back to the original directory on exit or error:
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pushd "${SCRIPT_DIR}" > /dev/null || exit
trap 'popd > /dev/null' EXIT

source ./configuration.sh

### Make your cluster
./make_cluster.sh -y

### Install a high performance default storage class
kubectl apply -f high-performance-storageclass.yaml
eksctl utils associate-iam-oidc-provider \
    --region $AWS_REGION \
    --cluster $CLUSTER_NAME \
    --approve
eksctl create iamserviceaccount \
    --name ebs-csi-controller-sa \
    --namespace kube-system \
    --cluster $CLUSTER_NAME \
    --region $AWS_REGION \
    --attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy \
    --approve \
    --override-existing-serviceaccounts
EBS_ROLE_ARN="$(eksctl get iamserviceaccount --cluster $CLUSTER_NAME --name ebs-csi-controller-sa --namespace kube-system --output json | awk -F'"' '/roleARN/{print $4}')"
eksctl create addon \
    --name aws-ebs-csi-driver \
    --cluster $CLUSTER_NAME \
    --region $AWS_REGION \
    --service-account-role-arn $EBS_ROLE_ARN \
    --force
### Check status of EBS CSI controller status
./check_ebs_csi_driver.sh
eksctl utils migrate-to-pod-identity --cluster $CLUSTER_NAME --approve

### Deploy NVIDIA-daemonset
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
# Patch the daemonset to install on our gpu nodes:
kubectl patch ds nvidia-device-plugin-daemonset -n kube-system --patch '{
  "spec": {
    "template": {
      "spec": {
        "tolerations": [
          {
            "key": "exclusive",
            "operator": "Equal",
            "value": "cvs-gpu",
            "effect": "NoSchedule"
          }
        ]
      }
    }
  }
}'
