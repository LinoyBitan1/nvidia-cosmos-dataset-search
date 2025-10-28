# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for document indexing API changes supporting cosmos-embed base64 strategy."""

import base64
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from haystack import Document
from haystack.dataclasses import ByteStream

from src.visual_search.common.models import (
    Collection,
    DocumentUploadEmbedding,
    DocumentUploadJson,
    DocumentUploadUrl,
    MimeType,
)
from src.visual_search.common.pipelines import EnabledPipeline
from src.visual_search.v1.apis.document_indexing import (
    _convert_to_haystack_document,
    _download_url_data,
    _index_documents,
    _index_haystack_documents,
)

# --------------------------------------------------------------------------- #
# Test Fixtures                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_video_data():
    """Sample video data for testing."""
    return b"fake_video_data_for_document_indexing_tests"


@pytest.fixture
def sample_text_data():
    """Sample text data for testing."""
    return "This is sample text content for testing document indexing"


@pytest.fixture
def mock_collection():
    """Mock Collection for testing."""
    collection = MagicMock(spec=Collection)
    collection.id = "test_collection_123"
    collection.pipeline = "test-pipeline"
    return collection


@pytest.fixture
def mock_cosmos_pipeline():
    """Mock EnabledPipeline with cosmos-embed configuration."""
    pipeline = MagicMock(spec=EnabledPipeline)
    pipeline.id = "test-pipeline"
    pipeline.enabled = True

    # Mock index pipeline with run method
    mock_index_pipeline = MagicMock()

    # Configure the run method to return expected output
    def mock_run(pipeline_input):
        # Extract documents from the pipeline input to count them
        documents_count = 0
        for component_inputs in pipeline_input.values():
            if isinstance(component_inputs, dict):
                for input_value in component_inputs.values():
                    if isinstance(input_value, list):
                        # Assume this is a list of documents
                        documents_count = len(input_value)
                        break

        return {"writer": {"documents_written": documents_count}}

    mock_index_pipeline.run = MagicMock(side_effect=mock_run)
    pipeline.index_pipeline = mock_index_pipeline

    # Mock index pipeline inputs
    mock_index_inputs = MagicMock()
    mock_index_inputs.index_name = ["writer.index_name"]
    pipeline.index_pipeline_inputs = mock_index_inputs

    return pipeline


# --------------------------------------------------------------------------- #
# Base64 Strategy Tests                                                       #
# --------------------------------------------------------------------------- #


def test_convert_video_url_stores_url_directly():
    """Test that video URLs store URL directly for presigned URL strategy."""

    doc_upload = DocumentUploadUrl(
        url="https://s3.amazonaws.com/bucket/video.mp4",
        mime_type=MimeType.MP4,
        id="test_video_url",
    )

    result = _convert_to_haystack_document(doc_upload)

    # Verify URL is stored directly (presigned URL strategy)
    assert result.content == "https://s3.amazonaws.com/bucket/video.mp4"
    assert result.meta["source_url"] == "https://s3.amazonaws.com/bucket/video.mp4"
    assert result.blob is None
    assert result.id == "test_video_url"


def test_convert_image_url_stores_url_directly():
    """Test that image URLs also store URL directly."""

    doc_upload = DocumentUploadUrl(
        url="https://cdn.example.com/image.jpg",
        mime_type=MimeType.JPEG,
        id="test_image_url",
    )

    result = _convert_to_haystack_document(doc_upload)

    # Verify URL is stored directly (presigned URL strategy)
    assert result.content == "https://cdn.example.com/image.jpg"
    assert result.meta["source_url"] == "https://cdn.example.com/image.jpg"
    assert result.blob is None


def test_convert_text_url_downloads_as_content(sample_text_data):
    """Test that text URLs download as content (not blob) for text processing."""

    doc_upload = DocumentUploadUrl(
        url="https://example.com/document.txt",
        mime_type=MimeType.TEXT,
        id="test_text_content",
    )

    with patch(
        "src.visual_search.v1.apis.document_indexing._download_url_data",
        return_value=sample_text_data.encode(),
    ) as mock_download:
        result = _convert_to_haystack_document(doc_upload)

        # Verify download was called
        mock_download.assert_called_once_with("https://example.com/document.txt")

        # Verify text goes to content field (not blob)
        assert result.content == sample_text_data
        assert result.blob is None
        assert result.id == "test_text_content"


def test_convert_video_json_creates_data_uri():
    """Test that DocumentUploadJson with base64 video data creates data URI."""

    video_data = b"embedded_video_data"
    video_b64 = base64.b64encode(video_data).decode("utf-8")

    doc_upload = DocumentUploadJson(
        content=f"{video_b64}", mime_type=MimeType.MP4, id="test_embedded_video"
    )

    result = _convert_to_haystack_document(doc_upload)

    # Verify data URI was created in content field
    assert result.content == f"data:video/mp4;base64,{video_b64}"
    assert result.blob is None
    assert result.id == "test_embedded_video"


def test_convert_embedding_upload_preserves_embeddings():
    """Test that DocumentUploadEmbedding preserves pre-computed embeddings."""

    embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5]

    doc_upload = DocumentUploadEmbedding(
        embedding=embedding_vector, id="test_precomputed_embedding"
    )

    result = _convert_to_haystack_document(doc_upload)

    # Verify embedding is preserved
    assert result.embedding == embedding_vector
    assert result.id == "test_precomputed_embedding"
    assert result.blob is None
    assert result.content is None


# --------------------------------------------------------------------------- #
# Download Function Tests                                                     #
# --------------------------------------------------------------------------- #


def test_download_url_data_success():
    """Test successful URL data download."""

    test_data = b"downloaded_data_content"
    url = "https://example.com/file.data"

    with patch("requests.get") as mock_get:
        # Mock successful response
        mock_response = MagicMock()
        mock_response.content = test_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = _download_url_data(url)

        # Verify request was made correctly
        mock_get.assert_called_once_with(url, timeout=30)
        mock_response.raise_for_status.assert_called_once()

        # Verify data returned
        assert result == test_data


def test_download_url_data_timeout_handling():
    """Test download timeout handling."""

    url = "https://slow-server.com/large_video.mp4"

    with patch("requests.get") as mock_get:
        # Mock timeout exception
        mock_get.side_effect = Exception("Request timeout")

        with pytest.raises(Exception, match="Request timeout"):
            _download_url_data(url)


def test_download_url_data_http_error_handling():
    """Test HTTP error handling during download."""

    url = "https://example.com/nonexistent.mp4"

    with patch("requests.get") as mock_get:
        # Mock HTTP error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = HTTPException(
            status_code=404, detail="404 Not Found"
        )
        mock_get.return_value = mock_response

        with pytest.raises(HTTPException, match="404 Not Found"):
            _download_url_data(url)


# --------------------------------------------------------------------------- #
# Integration with Cosmos Pipeline Tests                                      #
# --------------------------------------------------------------------------- #


def test_index_documents_cosmos_pipeline_integration(
    mock_collection, mock_cosmos_pipeline, sample_video_data
):
    """Test document indexing with cosmos pipeline end-to-end."""

    # Create test documents
    documents = [
        DocumentUploadUrl(
            url=f"https://s3.bucket.com/video_{i}.mp4",
            mime_type=MimeType.MP4,
            id=f"video_{i}",
        )
        for i in range(3)
    ]

    # Mock the pipeline execution
    mock_pipeline_output = {"writer": {"documents_written": 3}}

    # Mock get_pipeline_by_collection and get_document_stores (patch where they're imported)
    with patch(
        "src.visual_search.v1.apis.document_indexing.get_pipeline_by_collection",
        return_value=mock_cosmos_pipeline,
    ):
        with patch(
            "src.visual_search.v1.apis.document_indexing.get_document_stores",
            return_value=[],
        ):
            with patch(
                "src.visual_search.common.pipelines.run_index_pipeline",
                return_value=mock_pipeline_output,
            ) as mock_run_pipeline:

                result = _index_documents(
                    collection=mock_collection,
                    pipeline=mock_cosmos_pipeline,
                    documents=documents,
                )

                # Verify pipeline was called with correct parameters
                mock_run_pipeline.assert_called_once()
                call_kwargs = mock_run_pipeline.call_args[1]

                # Check that documents were converted to Haystack format with URLs in content
                haystack_docs = call_kwargs["documents"]
                assert len(haystack_docs) == 3

                for i, doc in enumerate(haystack_docs):
                    assert doc.id == f"video_{i}"
                    assert doc.content == f"https://s3.bucket.com/video_{i}.mp4"
                    assert (
                        doc.meta["source_url"] == f"https://s3.bucket.com/video_{i}.mp4"
                    )
                    assert doc.blob is None

                # Check index name is passed correctly
                assert call_kwargs["index_name"] == "test_collection_123"


def test_index_haystack_documents_calls_run_index_pipeline(
    mock_collection, mock_cosmos_pipeline
):
    """Test that _index_haystack_documents properly calls run_index_pipeline."""

    # Create Haystack documents with URLs in content (as would come from _convert_to_haystack_document)
    haystack_docs = []
    for i in range(2):
        doc = Document(
            id=f"haystack_doc_{i}",
            content=f"https://example.com/video_{i}.mp4",
            meta={
                "source": f"test_{i}",
                "source_url": f"https://example.com/video_{i}.mp4",
            },
        )
        haystack_docs.append(doc)

    mock_pipeline_output = {"writer": {"documents_written": 2}}

    # Mock get_pipeline_by_collection to return our mock pipeline (patch where it's imported)
    with patch(
        "src.visual_search.v1.apis.document_indexing.get_pipeline_by_collection",
        return_value=mock_cosmos_pipeline,
    ) as mock_get_pipeline:
        with patch(
            "src.visual_search.common.pipelines.run_index_pipeline",
            return_value=mock_pipeline_output,
        ) as mock_run_pipeline:

            result = _index_haystack_documents(
                collection=mock_collection,
                pipeline=mock_cosmos_pipeline,
                haystack_documents=haystack_docs,
            )

            # Verify get_pipeline_by_collection was called with the collection
            mock_get_pipeline.assert_called_with(mock_collection)

            # Verify run_index_pipeline was called correctly
            mock_run_pipeline.assert_called_once()
            call_kwargs = mock_run_pipeline.call_args[1]

            assert call_kwargs["index_pipeline"] == mock_cosmos_pipeline.index_pipeline
            assert (
                call_kwargs["index_pipeline_inputs"]
                == mock_cosmos_pipeline.index_pipeline_inputs
            )
            assert call_kwargs["documents"] == haystack_docs
            assert call_kwargs["index_name"] == "test_collection_123"


# --------------------------------------------------------------------------- #
# Security and Performance Tests                                              #
# --------------------------------------------------------------------------- #


def test_presigned_url_strategy_passes_url_to_cosmos():
    """Test that presigned URL strategy passes URLs directly to cosmos-embed."""

    presigned_url = (
        "https://s3.amazonaws.com/secure-bucket/video.mp4?X-Amz-Signature=..."
    )

    doc_upload = DocumentUploadUrl(
        url=presigned_url, mime_type=MimeType.MP4, id="security_test_video"
    )

    result = _convert_to_haystack_document(doc_upload)

    # Verify URL is stored directly for cosmos-embed to access
    assert result.content == presigned_url
    assert result.meta["source_url"] == presigned_url
    assert result.blob is None


def test_mixed_document_types_processing(sample_video_data, sample_text_data):
    """Test processing mixed document types (video, text, pre-computed embeddings)."""

    mixed_documents = [
        DocumentUploadUrl(
            url="https://example.com/video.mp4",
            mime_type=MimeType.MP4,
            id="mixed_video",
        ),
        DocumentUploadUrl(
            url="https://example.com/document.txt",
            mime_type=MimeType.TEXT,
            id="mixed_text",
        ),
        DocumentUploadEmbedding(embedding=[0.1, 0.2, 0.3], id="mixed_embedding"),
    ]

    def mock_download_side_effect(url):
        if "document.txt" in url:
            return sample_text_data.encode()
        else:
            raise ValueError(f"Unexpected URL: {url}")

    with patch(
        "src.visual_search.v1.apis.document_indexing._download_url_data",
        side_effect=mock_download_side_effect,
    ) as mock_download:

        results = [_convert_to_haystack_document(doc) for doc in mixed_documents]

        # Verify video document (URL stored directly)
        video_doc = results[0]
        assert video_doc.id == "mixed_video"
        assert video_doc.content == "https://example.com/video.mp4"
        assert video_doc.meta["source_url"] == "https://example.com/video.mp4"
        assert video_doc.blob is None

        # Verify text document (downloaded)
        text_doc = results[1]
        assert text_doc.id == "mixed_text"
        assert text_doc.content == sample_text_data
        assert text_doc.blob is None

        # Verify embedding document
        embedding_doc = results[2]
        assert embedding_doc.id == "mixed_embedding"
        assert embedding_doc.embedding == [0.1, 0.2, 0.3]
        assert embedding_doc.blob is None
        assert embedding_doc.content is None

        # Verify downloads were called appropriately
        assert mock_download.call_count == 1  # Only text, not video or embedding


# --------------------------------------------------------------------------- #
# Error Handling Tests                                                        #
# --------------------------------------------------------------------------- #


def test_invalid_base64_data_handling():
    """Test that invalid base64 data is stored as data URI and validated later by embedder."""

    # Invalid base64 data
    doc_upload = DocumentUploadJson(
        content="invalid_base64_data!!!",
        mime_type=MimeType.MP4,
        id="invalid_base64_test",
    )

    # Should create data URI without validation (validation happens in embedder)
    result = _convert_to_haystack_document(doc_upload)
    assert result.content == "data:video/mp4;base64,invalid_base64_data!!!"
    assert result.blob is None


def test_unsupported_mime_type_handling():
    """Test handling of unsupported MIME types."""

    doc_upload = DocumentUploadUrl(
        url="https://example.com/file.unknown",
        mime_type=MimeType.OTHER,  # Generic binary type
        id="unsupported_mime_test",
    )

    result = _convert_to_haystack_document(doc_upload)

    # Should store URL directly regardless of mime type
    assert result.content == "https://example.com/file.unknown"
    assert result.meta["source_url"] == "https://example.com/file.unknown"
    assert result.blob is None


def test_large_video_file_handling():
    """Test handling of large video files with presigned URL strategy."""

    doc_upload = DocumentUploadUrl(
        url="https://example.com/large_video.mp4",
        mime_type=MimeType.MP4,
        id="large_video_test",
    )

    result = _convert_to_haystack_document(doc_upload)

    # Should handle large files efficiently by storing URL directly
    assert result.content == "https://example.com/large_video.mp4"
    assert result.meta["source_url"] == "https://example.com/large_video.mp4"
    assert result.blob is None


# --------------------------------------------------------------------------- #
# Backwards Compatibility Tests                                               #
# --------------------------------------------------------------------------- #


def test_backwards_compatibility_with_existing_uploads():
    """Test that existing upload methods still work with cosmos changes."""

    # Test that DocumentUploadJson with regular content still works
    doc_upload = DocumentUploadJson(
        content="Regular text content",
        mime_type=MimeType.TEXT,
        id="backwards_compat_text",
    )

    result = _convert_to_haystack_document(doc_upload)

    assert result.id == "backwards_compat_text"
    assert result.content == "Regular text content"
    assert result.blob is None


def test_migration_validation_uses_presigned_url_strategy():
    """Test that presigned URL strategy is properly implemented."""

    # This test verifies that we're using presigned URLs efficiently
    doc_upload = DocumentUploadUrl(
        url="https://s3.bucket.com/video.mp4",
        mime_type=MimeType.MP4,
        id="migration_validation",
    )

    result = _convert_to_haystack_document(doc_upload)

    # Verify URL is stored directly (presigned URL strategy)
    assert result.content == "https://s3.bucket.com/video.mp4"
    assert result.meta["source_url"] == "https://s3.bucket.com/video.mp4"
    assert result.blob is None

    # URLs should be passed to cosmos-embed for direct access
    assert result.content.startswith("https://")
    assert "source_url" in result.meta
