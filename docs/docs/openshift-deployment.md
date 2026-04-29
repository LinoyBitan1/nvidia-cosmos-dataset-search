# Red Hat OpenShift AI Deployment Guide

This guide provides complete step-by-step instructions for deploying CDS (Cosmos Dataset Search) on Red Hat OpenShift AI (RHOAI).

## Overview

This deployment uses Helm charts with an OpenShift-specific values overlay. Deployment runs directly from your local machine using `oc` and `helm` CLI tools.

**What gets deployed**:

- Milvus vector database with built-in MinIO storage
- Cosmos-embed NIM for video embeddings (GPU)
- Visual Search API service
- React-based web UI
- Nginx reverse proxy (single Route with path-based routing)
- OpenShift SCCs, RBAC, Secrets, and Route

**Total time**: ~30-45 minutes

## Prerequisites

- **OpenShift** 4.14+ with `oc` CLI configured
- **Helm** 3.x installed
- **NGC API Key** — for pulling container images from `nvcr.io` and NIM access. Get one at https://org.ngc.nvidia.com/setup/api-key
- **GPU nodes** with NVIDIA GPU Operator configured
- **AWS CLI** — for data ingestion with MinIO/S3. Install from https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
- **python3** and **make** — for CDS CLI and dataset preparation

## 1. Login and create namespace

Choose a namespace and release name for your deployment. They can be different, but using the same name is simplest.

```bash
oc login --token=$OPENSHIFT_TOKEN --server=$OPENSHIFT_CLUSTER_URL

export CUSTOM_NAMESPACE=cosmos-cds        # your namespace
export RELEASE_NAME=cosmos-cds            # your Helm release name

oc new-project $CUSTOM_NAMESPACE
```

## 2. Configure GPU tolerations

Identify your GPU node taints:

```bash
oc get nodes -l nvidia.com/gpu.present=true -o name | \
  xargs -I{} oc describe {} | grep -A1 Taints
```

Edit `infra/blueprint/bringup/values-openshift.yaml` and uncomment/adjust the tolerations for `cosmos-embed` and `milvus.indexNode` to match your cluster taints:

```yaml
cosmos-embed:
  # Tolerations for GPU-tainted nodes (uncomment and adjust for your cluster).
  tolerations:
    - key: "nvidia.com/gpu"
      operator: Exists
      effect: NoSchedule

milvus:
  indexNode:
    # Tolerations for GPU-tainted nodes (uncomment and adjust for your cluster).
    tolerations:
      - key: "nvidia.com/gpu"
        operator: Exists
        effect: NoSchedule
```

## 3. Export your API key

```bash
export NGC_API_KEY="<your NGC API key>"
```

## 4. Deploy the application

The chart creates all required secrets (NGC image pull secret, API keys, encryption key) automatically from the values you pass. No manual `oc create secret` commands are needed.

```bash
helm dependency build infra/blueprint/bringup

helm upgrade --install $RELEASE_NAME infra/blueprint/bringup \
  -n $CUSTOM_NAMESPACE \
  -f infra/blueprint/bringup/values-openshift.yaml \
  --set "visual-search.openshift.secrets.ngcApiKey=$NGC_API_KEY" \
  --timeout 45m
```

> **Already have secrets?** If you prefer to create secrets manually before install, set `visual-search.openshift.secrets.create=false` and create `ngc-api`, `ngc-secret`, `nvcr-io`, and `secret-encryption-key` secrets yourself.

## 5. Verify the deployment

```bash
# All pods should be Running
oc get pods -n $CUSTOM_NAMESPACE

# Get and export the Route URL (used by CLI and ingestion scripts)
export CUSTOM_API_ENDPOINT=$(oc get route visual-search -n $CUSTOM_NAMESPACE -o jsonpath='{.spec.host}')
echo "https://$CUSTOM_API_ENDPOINT/cosmos-dataset-search"

# API health check
curl -k https://$CUSTOM_API_ENDPOINT/api/health

# Stream visual-search logs
oc logs -f deployment/visual-search -n $CUSTOM_NAMESPACE
```

Cosmos-embed takes the longest (~10-15 min) as it downloads a ~20GB model on first run.

**Expected pods** (pod names are prefixed with your release name):

| Pod | Purpose |
|-----|---------|
| `<release>-cosmos-embed` | Video embedding NIM (GPU) |
| `visual-search` | Search API service |
| `visual-search-react-ui` | Web UI |
| `nginx-proxy` | Reverse proxy (Route entry point) |
| `<release>-milvus-*` | Vector database (9 pods) |
| `<release>-minio` | S3-compatible object storage |
| `<release>-etcd` | Milvus metadata store |
| `<release>-kafka` | Milvus message queue |
| `<release>-zookeeper` | Kafka coordination |

## 6. Install CDS CLI

> **Note**: `CUSTOM_API_ENDPOINT` must be exported (set in Step 5) before running this script.

```bash
cd infra/blueprint/bringup
bash client_up.sh
```

**Duration**: 2-3 minutes

**What happens**:
- Installs CDS CLI from packaged source
- Creates Python virtual environment at the project root (`.venv/`)
- Configures CLI with the OpenShift Route endpoint
- Validates deployment by listing pipelines

**Verify CLI works**:

```bash
cd /path/to/cosmos-dataset-search
source .venv/bin/activate
cds pipelines list
cds collections list
```

## Data Ingestion

### Setup: Source environment variables

CDS works with any S3-compatible storage. Configure the `CUSTOM_*` variables for your storage provider:

**Option A: Built-in MinIO** (deployed with the chart):

```bash
cd /path/to/cosmos-dataset-search
source .venv/bin/activate

MINIO_ROUTE=$(oc get route minio -n $CUSTOM_NAMESPACE -o jsonpath='{.spec.host}')

export CUSTOM_AWS_ACCESS_KEY_ID=minioadmin
export CUSTOM_AWS_SECRET_ACCESS_KEY=minioadmin
export CUSTOM_AWS_REGION=us-east-1
export CUSTOM_S3_BUCKET_NAME=cosmos-videos
export CUSTOM_S3_FOLDER=msrvtt-videos
export CUSTOM_ENDPOINT_URL="https://$MINIO_ROUTE"
export CUSTOM_API_ENDPOINT=$(oc get route visual-search -n $CUSTOM_NAMESPACE -o jsonpath='{.spec.host}')
```

**Option B: AWS S3** (or any external S3):

```bash
cd /path/to/cosmos-dataset-search
source .venv/bin/activate

export CUSTOM_AWS_ACCESS_KEY_ID=<your-access-key>
export CUSTOM_AWS_SECRET_ACCESS_KEY=<your-secret-key>
export CUSTOM_AWS_REGION=<your-region>
export CUSTOM_S3_BUCKET_NAME=<your-bucket>
export CUSTOM_S3_FOLDER=<your-folder>
# CUSTOM_ENDPOINT_URL is not needed for AWS S3
export CUSTOM_API_ENDPOINT=$(oc get route visual-search -n $CUSTOM_NAMESPACE -o jsonpath='{.spec.host}')
```

> **Note**: Set `CUSTOM_ENDPOINT_URL` only for non-AWS S3 storage (MinIO, ODF/Noobaa, etc.). AWS S3 uses region-based endpoints automatically.

### Step 1: Prepare dataset

Download and prepare the MSR-VTT dataset. The sample config downloads 100 videos.

```bash
uv pip install datasets
make prepare-dataset CONFIG=scripts/msrvtt_test_100.yaml
```

**Duration**: 5-10 minutes

**What happens**:
- Downloads MSRVTT_Videos.zip from HuggingFace
- Extracts and copies 100 videos to `~/datasets/msrvtt/videos/`

> **Note**: The `prepare-dataset` script only supports HuggingFace datasets that provide videos as a downloadable ZIP file. For datasets without ZIPs, you'll need to download videos separately or provide them locally.

### Step 2: Upload videos to S3

```bash
export PROFILE_NAME="${CUSTOM_S3_BUCKET_NAME}-profile"

aws configure set aws_access_key_id "${CUSTOM_AWS_ACCESS_KEY_ID}" --profile "${PROFILE_NAME}"
aws configure set aws_secret_access_key "${CUSTOM_AWS_SECRET_ACCESS_KEY}" --profile "${PROFILE_NAME}"
aws configure set region "${CUSTOM_AWS_REGION}" --profile "${PROFILE_NAME}"
# Set endpoint URL for MinIO (skip this line for AWS S3)
aws configure set endpoint_url "${CUSTOM_ENDPOINT_URL}" --profile "${PROFILE_NAME}"

# Check if bucket exists; create it if not
aws s3 ls s3://$CUSTOM_S3_BUCKET_NAME --profile $PROFILE_NAME || \
  aws --profile $PROFILE_NAME s3 mb s3://$CUSTOM_S3_BUCKET_NAME

# Upload all videos from local dataset directory to MinIO bucket
aws --profile $PROFILE_NAME s3 cp ~/datasets/msrvtt/videos/ \
  s3://$CUSTOM_S3_BUCKET_NAME/$CUSTOM_S3_FOLDER/ --recursive

# Verify upload
echo "Videos uploaded. Checking count:"
aws --profile $PROFILE_NAME s3 ls s3://$CUSTOM_S3_BUCKET_NAME/$CUSTOM_S3_FOLDER/ | wc -l
```

**Duration**: 1-2 minutes

### Step 3: Ingest videos

The `ingest_custom_videos.sh` script **requires CUSTOM_*** variables to be set. These variables specify which S3 bucket has the videos to ingest.

```bash
cd infra/blueprint
bash ingest_custom_videos.sh
```

**What the script does with CUSTOM_* variables:**
- Creates AWS profile using CUSTOM_AWS_ACCESS_KEY_ID and CUSTOM_AWS_SECRET_ACCESS_KEY
- Gets deployment URL from CUSTOM_API_ENDPOINT (set in the setup step)
- Creates Kubernetes secret with S3 credentials
- Creates collection with storage configuration
- Ingests videos from s3://CUSTOM_S3_BUCKET_NAME/CUSTOM_S3_FOLDER/

**Duration**: 2-5 minutes for 5 videos (default limit)

**Expected output:**

```
Processed files: 5/5 ━━━━ 100%
Status code 200: 5/5 100%
Processed 5 files successfully
```

### Step 4: Configure CORS (external S3 only)

If using the built-in MinIO, skip this step — no CORS needed (same-origin).

If using external AWS S3, **you must configure CORS on your S3 buckets**. Without it, the web UI cannot load video assets from S3.

> **Note**: Requires `CUSTOM_API_ENDPOINT` and `CUSTOM_S3_BUCKET_NAME` to be set (from the Setup step above).

```bash
cd infra/blueprint/bringup
bash verify_and_configure_cors.sh -y
```

**What the script does:**
- Uses `CUSTOM_API_ENDPOINT` (set in the setup step) as the allowed CORS origin
- Checks IAM permissions for GetBucketCors and PutBucketCors
- Applies CORS policy to all buckets relevant to your deployment

**Example browser error if CORS is missing**:
```
Cross-Origin Request Blocked: The Same Origin Policy disallows reading the remote resource...
(Reason: CORS header 'Access-Control-Allow-Origin' missing)
```

> **Security tip:** The script uses your Route URL as the allowed origin. Do **not** use `*` for production. For manual CORS configuration, see [Manually Configure CORS for S3 Videos](#manually-configure-cors-for-s3-videos).

### Step 5: Verify ingestion

```bash
cds collections list

ROUTE_HOST=$(oc get route visual-search -n $CUSTOM_NAMESPACE -o jsonpath='{.spec.host}')
echo "https://$ROUTE_HOST/cosmos-dataset-search"
```

Open the Web UI and search for videos (e.g., "a person playing basketball").

### Ingest your own videos

1. Upload videos:

```bash
aws s3 cp /path/to/your/videos/ s3://$CUSTOM_S3_BUCKET_NAME/$CUSTOM_S3_FOLDER/ \
  --recursive --profile $PROFILE_NAME
```

2. Update `CUSTOM_*` variables if using a different bucket or credentials, then run:

```bash
cd infra/blueprint
bash ingest_custom_videos.sh
```

## Advanced Options

### Using AWS S3 Instead of MinIO

By default, the OpenShift overlay deploys a built-in MinIO pod for Milvus object storage. If you prefer to use AWS S3, override the storage settings at install time:

```bash
helm upgrade --install $RELEASE_NAME infra/blueprint/bringup \
  -n $CUSTOM_NAMESPACE \
  -f infra/blueprint/bringup/values-openshift.yaml \
  --set "visual-search.openshift.secrets.ngcApiKey=$NGC_API_KEY" \
  --set "milvus.minio.enabled=false" \
  --set "milvus.externalS3.enabled=true" \
  --set "milvus.externalS3.host=s3.$AWS_REGION.amazonaws.com" \
  --set "milvus.externalS3.bucketName=$S3_BUCKET_NAME" \
  --set "milvus.externalS3.region=$AWS_REGION" \
  --set "milvus.externalS3.useIAM=false" \
  --set "milvus.externalS3.accessKey=$AWS_ACCESS_KEY_ID" \
  --set "milvus.externalS3.secretKey=$AWS_SECRET_ACCESS_KEY" \
  --timeout 45m
```

> **Note**: Unlike EKS which uses IAM Roles for Service Accounts (IRSA) for keyless S3 access, OpenShift requires explicit AWS credentials via `accessKey` and `secretKey`.

### Managing Secrets

> **Important**: Storage secrets must be created in the `default` namespace due to a hardcoded namespace in the visual-search application ([k8s_secrets.py](../../src/visual_search/v1/apis/k8s_secrets.py)).

```bash
# Create
oc create secret generic my-s3-creds -n default \
  --from-literal=aws_access_key_id=$CUSTOM_AWS_ACCESS_KEY_ID \
  --from-literal=aws_secret_access_key=$CUSTOM_AWS_SECRET_ACCESS_KEY \
  --from-literal=aws_region=$CUSTOM_AWS_REGION

# Use in collection
cds collections create --pipeline cosmos_video_search_milvus \
  --name "My Collection" \
  --config-yaml <(echo "
tags:
  storage-template: 's3://$CUSTOM_S3_BUCKET_NAME/$CUSTOM_S3_FOLDER/{{filename}}'
  storage-secrets: 'my-s3-creds'
")

# List / Delete
oc get secrets -n default
oc delete secret my-s3-creds -n default
```

### Manually Configure CORS for S3 Videos

This section applies only when using external AWS S3. If using the built-in MinIO, CORS is not required.

#### Why CORS is Required

When your browser tries to load videos from S3, it makes cross-origin requests. S3 blocks these requests by default unless you explicitly configure CORS rules.

**Error you'll see without CORS**:
```
Cross-Origin Request Blocked: The Same Origin Policy disallows reading the remote resource...
(Reason: CORS header 'Access-Control-Allow-Origin' missing). Status code: 200.
```

#### Required IAM Permissions

Your AWS credentials must have these permissions on the S3 bucket:

```json
{
  "Effect": "Allow",
  "Action": [
    "s3:PutBucketCORS",
    "s3:GetBucketCORS"
  ],
  "Resource": "arn:aws:s3:::<your-bucket-name>"
}
```

If you see an "AccessDenied" error, contact your AWS administrator to grant these permissions.

#### Configure CORS

```bash
ROUTE_HOST=$(oc get route visual-search -n $CUSTOM_NAMESPACE -o jsonpath='{.spec.host}')

echo "Configuring CORS for bucket: $CUSTOM_S3_BUCKET_NAME"
echo "Allowing origin: https://$ROUTE_HOST"

aws s3api put-bucket-cors \
  --bucket $CUSTOM_S3_BUCKET_NAME \
  --region $CUSTOM_AWS_REGION \
  --profile $PROFILE_NAME \
  --cors-configuration "{
    \"CORSRules\": [{
      \"AllowedOrigins\": [\"https://${ROUTE_HOST}\"],
      \"AllowedMethods\": [\"GET\", \"HEAD\"],
      \"AllowedHeaders\": [\"*\"],
      \"ExposeHeaders\": [\"ETag\", \"Content-Length\", \"Content-Type\"],
      \"MaxAgeSeconds\": 3600
    }]
  }"

# Verify
aws s3api get-bucket-cors --bucket $CUSTOM_S3_BUCKET_NAME --region $CUSTOM_AWS_REGION --profile $PROFILE_NAME
```

**Alternative: Wildcard CORS (less secure, allows any origin)**:

```bash
aws s3api put-bucket-cors \
  --bucket $CUSTOM_S3_BUCKET_NAME \
  --region $CUSTOM_AWS_REGION \
  --profile $PROFILE_NAME \
  --cors-configuration '{
    "CORSRules": [{
      "AllowedOrigins": ["*"],
      "AllowedMethods": ["GET", "HEAD"],
      "AllowedHeaders": ["*"],
      "ExposeHeaders": ["ETag", "Content-Length", "Content-Type"],
      "MaxAgeSeconds": 3600
    }]
  }'
```

#### Test CORS Configuration

After configuring CORS, test that videos load in the web UI:

1. Open the web UI in your browser
2. Perform a search
3. Click on a video result
4. The video should load and play without errors

If you still see CORS errors, verify:
- CORS configuration is applied: `aws s3api get-bucket-cors --bucket <bucket-name> --region <region>`
- Your Route hostname matches the allowed origin in CORS rules
- You've configured CORS on all buckets that store videos

## Troubleshooting

| Problem | Check |
|---------|-------|
| Pods not starting | `oc describe pod <pod-name> -n $CUSTOM_NAMESPACE` |
| Image pull errors | Verify NGC_API_KEY: `oc get secret nvcr-io -n $CUSTOM_NAMESPACE` |
| Cosmos-embed pending | GPU nodes available? `oc get nodes -l nvidia.com/gpu.present=true` |
| Cosmos-embed creating >10 min | Normal — downloading ~20GB model |
| Route not accessible | `oc get pods -l app=nginx-proxy -n $CUSTOM_NAMESPACE` |
| PVC pending | Storage class exists? `oc get sc` |
| SCC issues | `oc get rolebinding -n $CUSTOM_NAMESPACE \| grep anyuid` |

**Useful commands:**

```bash
oc logs deployment/visual-search -n $CUSTOM_NAMESPACE --tail=100
oc logs deployment/$RELEASE_NAME-cosmos-embed -n $CUSTOM_NAMESPACE --tail=100
oc logs deployment/nginx-proxy -n $CUSTOM_NAMESPACE --tail=100
oc get svc,route -n $CUSTOM_NAMESPACE
oc get pv,pvc -n $CUSTOM_NAMESPACE
```

## OpenShift Overlay Strategy

1. An `openshift.enabled` flag is added to the base chart's `values.yaml` (default: `false`). When disabled, no OpenShift resources are rendered and the chart behaves identically to upstream.
2. All OpenShift-specific values and resources are placed in dedicated files: `values-openshift.yaml` (overlay) and `openshift.yaml` (conditional template for Routes, anyuid SCC RoleBinding, and Secrets).
3. Secrets (NGC image-pull secret, API credentials, encryption key) are created declaratively by the chart when provided via `--set` flags. This keeps the deployment to a single `helm install` command with no manual `oc create secret` steps.
4. Existing scripts (`verify_and_configure_cors.sh`, `ingest_custom_videos.sh`, `client_up.sh`) use `CUSTOM_API_ENDPOINT` to skip EKS-specific logic and use the OpenShift Route hostname instead.
5. The goal is to touch the original repository as little as possible, providing OpenShift deployment support with minimal changes.

## Known Limitations

1. **Storage secrets must be in the `default` namespace.** The visual-search application reads S3 credentials from the `default` namespace. The chart creates a RoleBinding granting the app's service accounts access to secrets in `default`. If you deploy to a different namespace, secrets must still be created in `default`.

2. **Resource sizing.** The `values-openshift.yaml` resource values match the production configuration. On smaller clusters, pods like `queryNode` (12 CPU / 96Gi) and `dataNode` (6 CPU / 24Gi) may stay Pending due to insufficient capacity. Reduce resource requests in `values-openshift.yaml` to fit your cluster.


## Uninstall

```bash
# Delete the project — removes all namespaced resources (pods, PVCs, services, etc.)
oc delete project $CUSTOM_NAMESPACE

# Clean up resources created in the default namespace
oc delete role ${CUSTOM_NAMESPACE}-secret-access-role -n default --ignore-not-found
oc delete rolebinding ${CUSTOM_NAMESPACE}-secret-access-binding -n default --ignore-not-found
oc delete secret ${CUSTOM_S3_BUCKET_NAME}-secrets-videos -n default --ignore-not-found
```
