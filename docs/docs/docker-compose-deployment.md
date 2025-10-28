# Docker Compose Deployment Guide

This guide walks through deploying CDS using Docker Compose for local development and testing.

## Prerequisites

Before proceeding, ensure you have completed the [Docker Compose Prerequisites](docker-compose-prerequisites.md), including:
- Docker and NVIDIA Container Toolkit installed
- NGC authentication configured
- Environment variables set or persisted
- LocalStack hostname mapping added to `/etc/hosts`

## Overview

Docker Compose deployment provides a complete CDS stack running on a single node. This deployment method is ideal for:
- Local development and testing
- Evaluating CDS capabilities
- Small-scale deployments
- Prototyping and experimentation

The deployment includes all required services: Visual Search API, Cosmos-embed NIM, Milvus vector database, LocalStack S3 storage, and the React web UI.

**Time Estimate**: First deployment takes 15-30 minutes (includes model downloads). Subsequent deployments take 2-5 minutes.

## Deployment Steps

### Step 1: Clone the Repository

Clone the CDS repository and navigate to the project directory:

```bash
# Clone the repository
git clone https://github.com/NVIDIA-Omniverse-blueprints/cosmos-dataset-search
cd cosmos-dataset-search

# Verify you're in the correct directory
ls -la
# You should see Makefile, pyproject.toml, and other project files
```

### Step 2: Review Milvus Configuration (Optional)

CDS includes multiple Milvus configuration files:

- **Default (Recommended)**: `deploy/standalone/milvus_localstack.yaml` - Optimized for CVDS with LocalStack
- **Official Reference**: `deploy/standalone/milvus_official.yaml` - Standard official Milvus configuration [github.com/milvus-io/milvus/blob/master/configs/milvus.yaml](https://github.com/milvus-io/milvus/blob/master/configs/milvus.yaml)

**Key optimizations in `milvus_localstack.yaml` (default):**
- GPU memory configuration for shared GPU with cosmos-embed
- Storage V2 enabled with full configuration for better reliability
- LocalStack S3 endpoints pre-configured
- Automatic startup ordering for reliable milvus startup(cosmos-embed → milvus)

**To switch to official Milvus config:**

Edit `deploy/standalone/docker-compose.build.yml`:

```yaml
milvus:
  volumes:
    # Option 1: Use LocalStack-optimized config (default, recommended)
    - ./milvus_localstack.yaml:/milvus/configs/milvus.yaml
    
    # Option 2: Use official Milvus config (requires manual LocalStack setup)
    # - ./milvus_official.yaml:/milvus/configs/milvus.yaml
```

**If using `milvus_official.yaml`, you MUST manually configure**:
1. S3/LocalStack settings (`minio.address: localstack`, `minio.port: 4566`)
2. GPU memory allocation (`gpu.initMemSize`, `gpu.maxMemSize`)
3. Storage scheme (`common.storage.scheme: s3`)

**To use a custom config:**
1. Create a new copy and Update S3 settings to point to LocalStack (see existing `milvus_localstack.yaml` or `milvus_official.yaml`)
2. Set GPU memory allocation based on your hardware

For GPU memory configuration details, see [GPU Memory Management Guide](../guides/gpu-memory-management.md).

### Step 3: Configure Environment Variables

CDS uses a `.env` file to manage all required environment variables. Start by copying the provided template:

```bash
# Copy the environment template
cp deploy/standalone/.env.example deploy/standalone/.env
```

Edit the `.env` file and update the following required variables:

```bash
# Edit the .env file with your preferred editor
nano deploy/standalone/.env
# or
vi deploy/standalone/.env
```

**Required Variables to Update:**

1. **`NVIDIA_API_KEY`** - Your NGC API key (obtained from [NGC](https://org.ngc.nvidia.com/setup/api-key))
   ```bash
   NVIDIA_API_KEY=<your-NGC-API-key>
   ```

2. **`DATA_DIR`** - Path to your data directory (e.g., `$HOME/cds-data`)
   ```bash
   DATA_DIR=/path/to/your/data
   ```
   
   **Important**: This directory must exist before proceeding. Create it if it doesn't exist:
   ```bash
   mkdir -p $HOME/cds-data
   ```
   
   This folder is used to enable faster LocalStack uploads by providing direct file system access to the containerized S3 service.

**Default Variables (typically no changes needed for local deployment):**
- `AWS_ACCESS_KEY_ID=test` (for LocalStack)
- `AWS_SECRET_ACCESS_KEY=test` (for LocalStack)
- `AWS_ENDPOINT_URL=http://localstack:4566`
- `COSMOS_EMBED_NIM_URI=http://cosmos-embed-nim:8000`
- `GUNICORN_PORT=8888`

**Optional Variables for Remote Access:**

3. **Web UI Configuration** - Only required if accessing the UI from a different host (not localhost)
   
   **When to set these**: If your web browser is running on a different machine than the CDS deployment (e.g., deploying on a remote server and accessing from your laptop), you need to update these three variables with your deployment host's IP address or hostname.
   
   **Variables to update:**
   
   - **`CDS_API_URL`** - API endpoint URL for the web UI
     ```bash
     CDS_API_URL=http://<deployment-host-ip>:8888/v1
     ```
   
   - **`CDS_CDN_URL`** - CDN URL for serving media assets (LocalStack S3)
     ```bash
     CDS_CDN_URL=http://<deployment-host-ip>:4566/cosmos-test-bucket
     ```
   
   - **`CDS_UI_URL`** - Web UI base URL
     ```bash
     CDS_UI_URL=http://<deployment-host-ip>:8080/
     ```
   
   **Example for remote access** (deploying on a server with IP 192.168.1.100):
   ```bash
   CDS_API_URL=http://192.168.1.100:8888/v1
   CDS_CDN_URL=http://192.168.1.100:4566/cosmos-test-bucket
   CDS_UI_URL=http://192.168.1.100:8080/
   ```

For a complete list of configuration options, see the comments in the `.env` file.

### Step 4: Validate Environment Configuration

Before proceeding, validate your environment configuration:

```bash
# Run the validation script
bash deploy/standalone/scripts/validate_env.sh
```

This script:
- Verifies all required variables are set
- Checks that API keys are not placeholder values
- Validates port numbers, URIs, and other settings
- Provides helpful error messages if issues are found

**Expected output**: Should complete with "Environment validation completed successfully."

If validation fails, review the error messages and update your `.env` file accordingly, then run the validation script again.

### Step 5: Install Dependencies

Install Python dependencies and set up the development environment:

```bash
make install
```

This command:
- Creates a Python virtual environment using UV
- Installs all Python dependencies with GPU support
- Generates protobuf definitions for service communication
- Sets up the complete development environment

**Expected output**: Should complete without errors. Look for messages about dependency installation and protobuf generation.

### Step 6: Build Docker Images

Build all required Docker images:

```bash
make build-docker
```

This command builds:
- Python base image with CUDA support
- Visual Search service image
- Supporting service images

**Time Estimate**: 10-20 minutes on first build. Subsequent builds use Docker cache and are much faster (1-2 minutes).

### Step 7: Launch Services

Start the complete CDS stack:

```bash
make test-integration-up
```

This command:
- Starts Milvus vector database with etcd
- Launches Cosmos-embed NIM service (GPU-accelerated)
- Starts Visual Search API service
- Launches LocalStack S3-compatible storage
- Starts React Web UI
- Waits for all services to become healthy

**Important Notes**:
- **First run**: Cosmos-embed NIM will download model weights (~20GB). This takes 10-15 minutes depending on your internet connection.
- **GPU check**: The NIM service requires GPU access. If it fails to start, verify your GPU setup with `nvidia-smi`.
- **Monitor progress**: Watch the logs to track model download and service startup.

The expected results from a successful deployment should match

```bash
Starting services...
cd deploy/standalone && docker compose -f docker-compose.build.yml up -d
[+] Running 1/1
 ✔ validate-env Pulled                                                                                                                     1.5s
[+] Running 7/7
 ✔ Container milvus-etcd                Healthy                                                                                            1.2s
 ✔ Container localstack                 Healthy                                                                                            1.2s
 ✔ Container milvus                     Healthy                                                                                            4.9s
 ✔ Container standalone-validate-env-1  Exited                                                                                             0.7s
 ✔ Container cosmos-embed               Healthy                                                                                           64.2s
 ✔ Container visual-search              Started                                                                                            0.2s
 ✔ Container cosmos-vds-web-ui          Started                                                                                            0.2s

----------------------------------------
Waiting for services to be ready...
This may take a few minutes for GPU services to initialize...
python scripts/wait_for_services.py
INFO:__main__:Waiting for services: milvus, cosmos-embed, visual-search, react-ui
INFO:__main__:Checking milvus at http://localhost:9091/healthz
INFO:__main__:milvus is ready
INFO:__main__:Checking cosmos-embed at http://localhost:9000/v1/health/ready
INFO:__main__:cosmos-embed is ready
INFO:__main__:Checking visual-search at http://localhost:8888/health
INFO:__main__:visual-search is ready
INFO:__main__:Checking react-ui at http://localhost:8080/
INFO:__main__:react-ui is ready
INFO:__main__:All services are ready!
INFO:__main__:Service check completed in 18.24s
INFO:__main__:All services are ready for testing!

Check the UI at http://localhost:8080/cosmos-dataset-search
```
You can monitor the startup process:

```bash
# View logs in real-time
make test-integration-logs

# Or view specific service logs
docker compose -f deploy/standalone/docker-compose.build.yml logs -f cosmos-embed-nim
```

### Verify Deployment (Optional)

Once all services are running, you can optionally verify each component service manually:

```bash
# Check service health
curl http://localhost:8888/health

# Expected response: "OK"

# Check Cosmos-embed NIM
curl http://localhost:9000/v1/health/ready

# Expected response: {"status":"ready"}

# List available embedding pipelines
curl http://localhost:8888/v1/pipelines

# Expected response: JSON list of available pipelines

# Check UI is responding
curl http://localhost:8080/cosmos-dataset-search
```

All health checks should return successful responses. If any service fails, see the [Troubleshooting section](#troubleshooting).

### Install and Configure CDS CLI

Install the CDS command-line interface for data ingestion and management.

**Note**: This step is required for the commands in the [Testing the Deployment](#testing-the-deployment) section below.

```bash
# Install CDS CLI
make install-cds-cli

# Activate the virtual environment
source .venv/bin/activate

# Verify CLI installation
cds --help

# Configure CLI to connect to local deployment
cds config set
# When prompted, enter: http://localhost:8888

# Verify configuration
cds pipelines list
```


## Accessing the Services

### Web UI

Access the interactive web interface at:

```
http://localhost:8080/cosmos-dataset-search
```
The web browser must be running on the localhost for access to succeed. 

The UI provides:
- Text-to-video and video-to-video search interface
- Collection browsing and management
- Real-time search results with video previews
- Data curation tools

**Note**: You'll need to create a collection and ingest data before performing searches. See [Testing the Deployment](#testing-the-deployment) below.

### API Documentation

Interactive API documentation (Swagger UI) is available at:

```
http://localhost:8888/v1/docs
```

This provides complete API reference with interactive request testing. For detailed API usage, see the [API Reference](api_reference.md).

## Testing the Deployment

### Quick Verification Test

Run the built-in integration tests to verify all services are functioning correctly:

```bash
make test-integration-run
```

This runs:
- Minimal integration test validating service communication
- Cosmos video end-to-end test verifying embedding generation

**Expected result**: All tests should pass. This confirms the deployment is working correctly.

### Ingest Sample Data

Test data ingestion with a small sample dataset:

```bash
# Ingest 100 videos from MSR-VTT dataset for testing
make ingest-msrvtt-small
```

**Note**: This command ingests a small subset of videos for quick testing. The full ingestion process:
1. Downloads videos from the dataset
2. Uploads to LocalStack S3 storage
3. Processes videos through the embedding pipeline
4. Indexes embeddings in Milvus

**Time Estimate**: 5-10 minutes for 100 videos, depending on your system.

**Alternative**: For a more comprehensive test with the full dataset:
```bash
make ingest-msrvtt
```

### Verify Data Ingestion

After ingestion completes, verify the data:

```bash
# List collections (should show the newly created collection)
curl http://localhost:8888/v1/collections

# Or use the CLI
cds collections list

# Perform a test search
cds search --collection-ids <collection-id> \
  --text-query "person walking" \
  --top-k 5
```

### Test via Web UI

Open the web UI (`http://localhost:8080/cosmos-dataset-search`) and:
1. Select the ingested collection
2. Enter a text query (e.g., "person walking", "cat playing")
3. View the search results with video previews
4. Try video-to-video search by uploading a query video

## Managing the Deployment

### Important Note on Data Persistence

**All data ingested into CDS is ephemeral and will be lost when services are stopped and restarted.** This includes all collections, ingested videos, and embeddings. You will need to re-ingest your data after restarting the services.

### Stopping and Restarting Services

Stop all running services:

```bash
make test-integration-down
```

Restart services:

```bash
make test-integration-up
```

**Note**: After restarting, all previous data will be gone. You must re-ingest your datasets.

### Clean Restart (Rebuild Images)

To stop services and also remove Docker images (forcing a rebuild on next startup):

```bash
# Stop services and remove images
make test-integration-clean

# Restart and rebuild
make test-integration-up
```

Use `make test-integration-clean` when you want to clear cached images and ensure a fresh build on the next startup.

### Advanced Management

For detailed service management, including viewing logs, restarting individual services, and advanced configuration options, see the [CDS User Guide](user-guide.md).

## Service Endpoints Reference

| Service | Endpoint | Purpose |
|---------|----------|---------|
| Visual Search API | http://localhost:8888 | REST API for search operations |
| API Documentation | http://localhost:8888/v1/docs | Interactive API docs (Swagger) |
| Cosmos-embed NIM | http://localhost:9000 | Embedding service |
| Milvus Database | localhost:19530 | Vector database (internal) |
| LocalStack S3 | http://localhost:4566 | S3-compatible storage (internal) |
| Web UI | http://localhost:8080/cosmos-dataset-search | Interactive web interface |

## Troubleshooting

If you encounter issues during deployment or operation, see the [Docker Compose Troubleshooting Guide](troubleshooting-docker-compose.md) for detailed solutions to common problems.

## Next Steps

After successfully deploying CDS, proceed to:

1. **[CDS User Guide](user-guide.md)** - Learn how to interact with CDS
   
   The user guide covers three interaction methods:
   - [UI User Guide](ui-user-guide.md) - Web interface usage
   - [CLI User Guide](cli-user-guide.md) - Command-line operations
   - [API User Guide](api-user-guide.md) - REST API usage and examples
