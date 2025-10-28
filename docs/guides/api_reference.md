# API Reference

## OpenAPI Schema

The complete OpenAPI schema is available in JSON format at:
- **File**: [openapi_schema_cvds.json](../api_reference/openapi_schema_cvds.json)
- **Live Documentation**: `http://localhost:8888/v1/docs` (when running locally)
- **Raw Schema Endpoint**: `http://localhost:8888/v1/openapi.json`

## Interactive Documentation

When the CVDS service is running, you can access the interactive API documentation:

- **Swagger UI**: `http://localhost:8888/v1/docs`
- **ReDoc**: `http://localhost:8888/v1/redoc`

## Practical Examples

For hands-on curl examples and practical usage, see the [API Guide](../guides/api.md).

## Key Endpoints Overview

The CVDS API provides the following main endpoint categories:

### Health & Status
- `GET /v1/health` - Service health check
- `GET /v1/pipelines` - List available pipelines

### Collection Management
- `POST /v1/collections` - Create new collection
- `GET /v1/collections` - List all collections
- `GET /v1/collections/{collection_id}` - Get collection details
- `DELETE /v1/collections/{collection_id}` - Delete collection

### Document Indexing
- `POST /v1/collections/{collection_id}/documents` - Index documents
- `POST /v1/collections/{collection_id}/embeddings` - Bulk embeddings ingestion
- `POST /v1/collections/{collection_id}/status` - Check ingestion status

### Search & Retrieval
- `POST /v1/collections/{collection_id}/search` - Semantic search
- `POST /v1/collections/{collection_id}/search/hybrid` - Hybrid search

### Pipeline Operations
- `GET /v1/pipelines/{pipeline_id}/collections` - Get pipeline collections
- `GET /v1/pipelines/{pipeline_id}/visualize` - Visualize pipeline

### Secrets Management
- `POST /v1/secrets` - Store secrets
- `GET /v1/secrets` - List secrets
- `DELETE /v1/secrets/{secret_name}` - Delete secret

## Authentication

Currently, the CVDS API does not require authentication for local development. For production deployments, refer to the [AWS EKS Deployment Guide](../docs/aws-eks-deployment.md) for security configuration.

## Rate Limits

See the [API Guide](../guides/api.md) for current rate limits and usage guidelines.