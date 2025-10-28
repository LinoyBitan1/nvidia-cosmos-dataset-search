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

echo "=== COSMOS-EMBED DEBUGGING SCRIPT ==="
echo "Timestamp: $(date)"
echo

echo "=== POD STATUS ==="
kubectl get pods -l app.kubernetes.io/name=nvidia-nim-cosmos-embed -o wide
echo

echo "=== POD DETAILS ==="
POD_NAME=$(kubectl get pods -l app.kubernetes.io/name=nvidia-nim-cosmos-embed -o jsonpath='{.items[0].metadata.name}')
if [ -n "$POD_NAME" ]; then
    echo "Describing pod: $POD_NAME"
    kubectl describe pod $POD_NAME
    echo
    echo "=== POD EVENTS ==="
    kubectl get events --field-selector involvedObject.name=$POD_NAME --sort-by='.lastTimestamp'
else
    echo "No cosmos-embed pod found"
fi
echo

echo "=== NODE RESOURCES ==="
kubectl get nodes -o wide
echo
kubectl describe nodes | grep -A 10 -B 5 "nvidia.com/gpu"
echo

echo "=== GPU NODE LABELS ==="
kubectl get nodes --show-labels | grep -E "(cvs-gpu|nvidia)"
echo

echo "=== STORAGE CLASSES ==="
kubectl get storageclass
echo

echo "=== PVC STATUS ==="
kubectl get pvc
echo

echo "=== HELM RELEASE STATUS ==="
helm list
echo
helm status cosmos-embed
echo

echo "=== SERVICE STATUS ==="
kubectl get service -l app.kubernetes.io/name=nvidia-nim-cosmos-embed
echo

echo "=== RESOURCE QUOTAS ==="
kubectl describe resourcequotas 2>/dev/null || echo "No resource quotas found"
echo

echo "=== END DEBUG SCRIPT ==="