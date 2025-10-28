# GPU Memory Management Guide

This guide explains how to configure GPU memory usage for the CVDS system to avoid Out-of-Memory (OOM) issues when running both Cosmos Embed and Milvus services with CAGRA (GPU-accelerated vector search).

## Problem

When running both Cosmos Embed and Milvus GPU services on the same GPU, you may encounter memory conflicts:

- **Cosmos Embed**: Uses ~5GB GPU memory (Triton server + model loading)
- **Milvus GPU**: Requires additional GPU memory for vector operations
- **Total**: Can exceed available GPU memory (e.g., 12GB RTX 4070)

## Solutions

### Option 1: Use CPU Version of Milvus (For Development)

For development or when GPU memory is limited, use the CPU version:

```bash
# Set in your .env file or environment
MILVUS_IMAGE=milvusdb/milvus:v2.4.6  # CPU version
```

**Pros:**
- No GPU memory conflicts
- Stable and reliable
- Good performance for most use cases

**Cons:**
- No CAGRA acceleration
- Slower vector operations compared to GPU version

### Option 1.5: Use GPU Version with CAGRA (Recommended for Production)

For high-performance vector search with CAGRA acceleration:

```bash
# Set in your .env file or environment
MILVUS_IMAGE=milvusdb/milvus:v2.4.4-gpu  # GPU version with CAGRA support
```

**Pros:**
- CAGRA GPU acceleration for fast vector search
- Better performance for large-scale vector operations
- Optimized for production workloads

**Cons:**
- Requires careful GPU memory management
- More complex configuration

### Option 2: Configure GPU Memory with Automatic Allocation (Recommended for GPU Milvus)

When running GPU Milvus with Cosmos Embed on the same GPU, use automatic memory allocation to prevent conflicts:

**Key Requirements:**
1. **Ordered Startup**: Ensure cosmos-embed starts before Milvus (via docker-compose dependencies)
2. **Automatic Allocation**: Set Milvus GPU memory to `0/0` to auto-detect remaining GPU memory

**Configuration:**

In `docker-compose.build.yml`, add cosmos-embed as a dependency for Milvus:

```yaml
milvus:
  depends_on:
    cosmos-embed:
      condition: service_healthy  # Milvus waits for cosmos-embed to be ready
    etcd:
      condition: service_healthy
    localstack:
      condition: service_healthy
```

In `milvus_localstack.yaml`, set GPU memory to auto-allocate:

```yaml
gpu:
  initMemSize: 0  # Auto: half of remaining GPU memory after cosmos-embed
  maxMemSize: 0   # Auto: all remaining GPU memory after cosmos-embed
```

**How It Works:**
1. Cosmos Embed starts first and claims ~20-22GB GPU memory
2. Milvus starts after cosmos-embed is healthy
3. Milvus detects remaining GPU memory and allocates:
   - `initMemSize`: Half of remaining memory
   - `maxMemSize`: All remaining memory

**Pros:**
- Eliminates GPU memory race conditions
- Adapts to different GPU sizes automatically
- Maximizes GPU utilization without conflicts




### Option 3: Sequential GPU Usage (Manual Alternative)

If not using docker-compose dependencies, you can manually run services sequentially:

```bash
# Start only Cosmos Embed first
docker compose -f deploy/standalone/docker-compose.build.yml up -d cosmos-embed

# Wait for it to be ready, then start Milvus
docker compose -f deploy/standalone/docker-compose.build.yml up -d milvus
```

**Note:** This is less reliable than Option 2 because it requires manual intervention and doesn't prevent race conditions on restarts.

## Configuration Files

### Milvus Configuration

#### For Shared GPU Environment (with Cosmos Embed)

Update `milvus_localstack.yaml` for automatic GPU memory allocation and optimal storage settings:

```yaml
# GPU Memory Pool Configuration for Shared GPU Environment
# Setting both to 0 enables automatic allocation based on remaining GPU memory
# after cosmos-embed has claimed its share (~20-22GB)
gpu:
  initMemSize: 0  # Auto: half of remaining GPU memory after cosmos-embed
  maxMemSize: 0   # Auto: all remaining GPU memory after cosmos-embed
  overloadedMemoryThresholdPercentage: 95  # Prevent GPU OOM by limiting max usage to 95%

# CAGRA Configuration (GPU-accelerated vector search)
common:
  simdType: auto  # Use GPU acceleration when available
  storage:
    scheme: s3
    enablev2: true  # Enable Storage V2 for better reliability and segment loading
```

**Important Settings Explained:**

1. **GPU Memory Auto-Allocation** (`initMemSize: 0`, `maxMemSize: 0`):
   - When set to `0`, Milvus automatically detects available GPU memory after cosmos-embed loads
   - `initMemSize: 0` → Uses half of remaining GPU memory initially
   - `maxMemSize: 0` → Can expand to use all remaining GPU memory as needed
   - **Why this works**: Prevents hardcoded limits that might not match your actual GPU capacity

2. **GPU Memory Threshold** (`overloadedMemoryThresholdPercentage: 95`):
   - Limits GPU memory usage to 95% of the allocated pool
   - Prevents GPU Out-of-Memory (OOM) errors by reserving 5% buffer
   - Critical safety mechanism when running near GPU capacity limits
   - **Default**: 95 (recommended to keep this value)

3. **Storage V2** (`enablev2: true`):
   - **Critical** for reliable segment loading from S3/LocalStack
   - Provides better error handling and performance
   - Includes automatic retry logic for transient failures
   - Particularly important when using LocalStack or S3-compatible storage

## Monitoring GPU Memory

### Check Current Usage

```bash
# Monitor GPU memory usage
nvidia-smi

# Watch real-time usage
watch -n 1 nvidia-smi

# Check specific container usage
docker stats cosmos-embed milvus
```

### Troubleshooting

#### Common Issues

1. **LoadSegment Exception (Random Query Hangs)**
   ```
   [WARN] [delegator/delegator_data.go:360] ["failed to load growing segment"]
   [error="At LoadSegment: std::exception"]
   ```
   **Root Causes**: 
   - **Primary**: GPU memory allocation race condition between Milvus and cosmos-embed
   - **Secondary**: Storage V1 (`enablev2: false`) less robust for segment loading failures
   
   **Solution**: 
   - Add cosmos-embed dependency to Milvus in docker-compose (see Option 2)
   - Set GPU memory to `0/0` in milvus_localstack.yaml for automatic allocation
   - Enable Storage V2: Set `common.storage.enablev2: true` in milvus_localstack.yaml
   - This ensures cosmos-embed claims GPU memory first, then Milvus uses the remainder with better error handling

2. **CUDA Error: Insufficient Driver**
   ```
   cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version
   ```
   **Solution**: Use CPU version of Milvus or update CUDA drivers

3. **Out of Memory Error**
   ```
   CUDA out of memory. Tried to allocate X MB
   ```
   **Solution**: 
   - If using shared GPU: Ensure startup dependency is configured (Option 2)
   - If standalone Milvus: Reduce memory limits or use CPU version
   - Check `nvidia-smi` to verify available GPU memory

4. **Container Crash on Startup**
   ```
   terminate called after throwing an instance of 'raft::cuda_error'
   ```
   **Solution**: Check GPU memory availability and driver compatibility

#### Debug Commands

```bash
# Check container logs
docker logs cosmos-embed
docker logs milvus

# Check GPU processes
nvidia-smi pmon

# Check CUDA version compatibility
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi
```

## Best Practices & Recommendations

1. **For Shared GPU Setup (Cosmos Embed + Milvus GPU)**:
   - Use ordered startup with docker-compose dependencies
   - Set Milvus GPU memory to `0/0` for automatic allocation
   - Require at least 24GB GPU (cosmos-embed ~22GB + Milvus ~2GB minimum)
   - Monitor with `nvidia-smi` to verify memory distribution

2. **For Development**: 
   - Use `milvusdb/milvus:v2.4.6` (CPU) to avoid GPU memory complexity
   - Faster iteration without GPU driver concerns

3. **For Production with Limited GPU Memory (<16GB)**:
   - Use CPU Milvus + GPU Cosmos Embed
   - Acceptable search performance with CPU HNSW index

4. **For High-Performance Production (≥16GB GPU)**:
   - Use GPU Milvus with CAGRA + GPU Cosmos Embed
   - Implement automatic memory allocation (Option 2)
   - Enable monitoring and alerting on GPU memory usage

5. **General**:
   - Monitor Memory Usage: Always check `nvidia-smi` before starting services
   - Test Incrementally: Verify services start successfully one by one
   - Document Your Setup: Record working configurations for your specific hardware

## Hardware Recommendations

| Use Case | GPU Memory | Recommended Configuration |
|----------|------------|---------------------------|
| Development | 8GB+ | CPU Milvus + Cosmos Embed |
| Production (Small) | 12GB+ | CPU Milvus + Cosmos Embed |
| Production (Large) | 16GB+ | GPU Milvus + Cosmos Embed |
| High Performance | 24GB+ | GPU Milvus + Cosmos Embed |

