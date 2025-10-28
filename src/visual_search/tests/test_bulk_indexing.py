# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import types
from datetime import datetime
from unittest.mock import MagicMock, call

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.visual_search.v1.apis.bulk_indexing as bi
from src.visual_search.common.models import Collection
from src.visual_search.tests.conftest import FakeStore

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def make_store(**overrides):
    """Return a FakeStore with any overrides applied."""
    store = FakeStore()
    # Apply any overrides to the store's mock methods
    for key, value in overrides.items():
        if hasattr(store, key):
            # If it's already a MagicMock, configure it
            if isinstance(getattr(store, key), MagicMock):
                getattr(store, key).configure_mock(**{'side_effect': value} if callable(value) else {'return_value': value})
            else:
                setattr(store, key, value)
        else:
            setattr(store, key, value)
    return store


class FakeStored:
    def __init__(self, pipeline):
        self.pipeline = pipeline


class FakeState:
    def __init__(self, state, state_name="", failed_reason=None):
        self.state = state
        self.state_name = state_name
        self.failed_reason = failed_reason or ""


class FakeTask:
    def __init__(
        self,
        task_id,
        state,
        state_name="",
        failed_reason=None,
        progress=None,
        collection_name="",
        files=None,
    ):
        self.task_id = task_id
        self.state = state
        self.state_name = state_name
        # always a string for Pydantic
        self.failed_reason = failed_reason if failed_reason is not None else ""
        self.progress = progress
        self.collection_name = collection_name
        self.files = files or []


def setup_pipelines(monkeypatch, pipelines):
    # pipelines: dict of pipeline_name -> fake document-store
    pipes = {}
    for name in pipelines:
        # Create a mock index_pipeline with the necessary attributes
        index_pipeline = MagicMock()
        index_pipeline.name = name
        pipes[name] = types.SimpleNamespace(name=name, id=name, index_pipeline=index_pipeline)
    
    monkeypatch.setattr(bi, "enabled_pipelines", pipes)
    # Updated to match the fix - get_document_stores is called with index_pipeline
    # Use idx_pl.name to get the correct store from pipelines dict
    monkeypatch.setattr(
        bi, "get_document_stores", 
        lambda idx_pl: [pipelines.get(idx_pl.name)] if hasattr(idx_pl, 'name') and idx_pl.name in pipelines else []
    )
    monkeypatch.setattr(bi, "create_safe_name", lambda name: name)
    monkeypatch.setattr(bi, "build_storage_options", lambda ak, sk, eu: {})

    # avoid real Milvus connection/schema checks
    async def _noop_validate_parquet_schema(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bi, "validate_parquet_schema", _noop_validate_parquet_schema)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(bi.router)
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def mock_get_collections(monkeypatch):
    def mock_get(coll_id):
        return Collection(
            id=coll_id, pipeline="coll", name="test", created_at=datetime.utcnow()
        )

    monkeypatch.setattr(
        "src.visual_search.v1.apis.bulk_indexing.get_collections", mock_get
    )
    return mock_get


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_insert_data_success(monkeypatch, client, mock_get_collections):  # noqa: ARG001
    ds = make_store(bulk_insert_files=MagicMock(side_effect=[1, 2]))
    setup_pipelines(monkeypatch, {"coll": ds})

    payload = {
        "collection_name": "coll",
        "parquet_paths": ["s3://test-bucket/file1.parquet", "s3://test-bucket/file2.parquet"],
        "access_key": "ak",
        "secret_key": "sk",
        "endpoint_url": "http://example.com",
    }
    resp = client.post("/insert-data", json=payload)
    assert resp.status_code == 202
    assert resp.json() == {
        "status": "success",
        "message": "Data insertion started",
        "job_id": "1",
    }
    ds.bulk_insert_files.assert_has_calls(
        [
            call(collection_name="coll", file_paths=["file1.parquet"]),
            call(collection_name="coll", file_paths=["file2.parquet"]),
        ]
    )


def test_insert_data_pipeline_not_found(monkeypatch, client, mock_get_collections):  # noqa: ARG001
    ds = make_store(bulk_insert_files=MagicMock())
    setup_pipelines(monkeypatch, {"x": ds})
    monkeypatch.setattr(
        "src.visual_search.v1.apis.bulk_indexing.get_collections", lambda x: None
    )

    payload = {
        "collection_name": "does_not_exist",
        "parquet_paths": ["s3://test-bucket/test.parquet"],
        "access_key": None,
        "secret_key": None,
        "endpoint_url": None,
    }
    resp = client.post("/insert-data", json=payload)
    assert resp.status_code == 404


def test_insert_data_invalid_s3_path(monkeypatch, client, mock_get_collections):  # noqa: ARG001
    """Test that non-S3 paths or improperly formatted S3 paths are rejected."""
    ds = make_store(bulk_insert_files=MagicMock())
    setup_pipelines(monkeypatch, {"coll": ds})
    
    # Test invalid S3 paths
    invalid_paths = [
        "s3://bucket-only",  # No path after bucket
        "s3://",  # Empty bucket
        "/local/path/file.parquet",  # Not S3
        "https://example.com/file.parquet",  # Not S3
        "file.parquet",  # Relative path
    ]
    
    for invalid_path in invalid_paths:
        payload = {
            "collection_name": "coll",
            "parquet_paths": [invalid_path],
            "access_key": "ak",
            "secret_key": "sk",
            "endpoint_url": "http://example.com",
        }
        resp = client.post("/insert-data", json=payload)
        assert resp.status_code == 400
        assert "Invalid S3 path format" in resp.text
        assert "s3://bucket/path/to/file.parquet" in resp.text


def test_job_status_success(monkeypatch, client):
    ds1 = make_store(
        get_bulk_insert_state=MagicMock(
            side_effect=bi.MilvusException("can't find task")
        )
    )
    state = FakeState(bi.BulkInsertState.ImportStarted, state_name="ImportStarted")
    ds2 = make_store(get_bulk_insert_state=MagicMock(return_value=state))
    setup_pipelines(monkeypatch, {"p1": ds1, "p2": ds2})

    resp = client.get("/job-status/42")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "42"
    assert data["status"] == "in_progress"
    assert data["details"] == "ImportStarted"


def test_job_status_not_found(monkeypatch, client):
    ds = make_store(
        get_bulk_insert_state=MagicMock(
            side_effect=bi.MilvusException("can't find task")
        )
    )
    setup_pipelines(monkeypatch, {"p": ds})

    resp = client.get("/job-status/100")
    assert resp.status_code == 404


def test_job_status_milvus_error(monkeypatch, client):
    ds = make_store(
        get_bulk_insert_state=MagicMock(side_effect=bi.MilvusException("fatal error"))
    )
    setup_pipelines(monkeypatch, {"p": ds})

    resp = client.get("/job-status/1")
    assert resp.status_code == 500


def test_job_status_unknown_state(monkeypatch, client):
    state = FakeState(state="UNKNOWN_STATE", state_name="UNK")
    ds = make_store(get_bulk_insert_state=MagicMock(return_value=state))
    setup_pipelines(monkeypatch, {"p": ds})

    resp = client.get("/job-status/7")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "unknown"
    assert data["details"] == "UNK"


def test_list_jobs_success(monkeypatch, client):
    task1 = FakeTask(
        task_id=1,
        state=bi.BulkInsertState.ImportCompleted,
        state_name="Comp",
        progress=50,
        collection_name="col1",
        files=["a"],
    )
    ds1 = make_store(list_bulk_insert_tasks=MagicMock(return_value=[task1]))

    task2 = FakeTask(
        task_id=2,
        state=999,
        state_name="St",
        failed_reason="fail",
        progress=100,
        collection_name="col2",
        files=["b", "c"],
    )
    ds2 = make_store(list_bulk_insert_tasks=MagicMock(return_value=[task2]))

    setup_pipelines(monkeypatch, {"p1": ds1, "p2": ds2})

    resp = client.get("/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list) and len(data) == 2
    assert any(j["job_id"] == "1" and j["status"] == "completed" for j in data)
    assert any(
        j["job_id"] == "2" and j["status"] == "unknown" and j["details"] == "fail"
        for j in data
    )


def test_list_jobs_limit_and_filter(monkeypatch, client):
    tasks = [
        FakeTask(
            task_id=i,
            state=bi.BulkInsertState.ImportPending,
            state_name="Pending",
            progress=None,
            collection_name="xyz",
            files=[],
        )
        for i in range(5)
    ]
    ds = make_store(list_bulk_insert_tasks=MagicMock(return_value=tasks))
    setup_pipelines(monkeypatch, {"pdr": ds})

    resp = client.get("/jobs?limit=2")
    assert resp.status_code == 200
    assert len(resp.json()) == 2

    resp = client.get("/jobs?collection_name=collname")
    assert resp.status_code == 200
    ds.list_bulk_insert_tasks.assert_called_with(limit=None, collection_name="collname")


def test_list_jobs_exception(monkeypatch, client):
    ds = make_store(
        list_bulk_insert_tasks=MagicMock(side_effect=bi.MilvusException("oops"))
    )
    setup_pipelines(monkeypatch, {"pp": ds})

    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert resp.json() == []
