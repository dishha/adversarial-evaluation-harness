import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from adversarial_response_engine.output.storage import LocalStorage, S3Storage, AzureBlobStorage, make_storage


# ── LocalStorage ──────────────────────────────────────────────────────────────

def test_local_storage_creates_file():
    storage = LocalStorage()
    data = {"key": "value", "num": 42}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output" / "results.json"
        uri = storage.write_json(path, data)
        assert path.exists()
        assert uri == str(path)


def test_local_storage_valid_json_content():
    storage = LocalStorage()
    data = {"session_id": "abc", "scores": [1, 2, 3]}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "results.json"
        storage.write_json(path, data)
        with open(path) as f:
            loaded = json.load(f)
    assert loaded == data


def test_local_storage_creates_parent_dirs():
    storage = LocalStorage()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "deep" / "nested" / "dir" / "results.json"
        storage.write_json(path, {"x": 1})
        assert path.exists()


def test_local_storage_overwrites_existing_file():
    storage = LocalStorage()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "results.json"
        storage.write_json(path, {"v": 1})
        storage.write_json(path, {"v": 2})
        with open(path) as f:
            data = json.load(f)
        assert data == {"v": 2}


def test_local_storage_returns_path_string():
    storage = LocalStorage()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.json"
        result = storage.write_json(path, {})
        assert isinstance(result, str)
        assert result == str(path)


# ── S3Storage ──────────────────────────────────────────────────────────────────

def test_s3_storage_init():
    s3 = S3Storage(bucket="my-bucket", prefix="results", region="us-west-2")
    assert s3.bucket == "my-bucket"
    assert s3.prefix == "results"
    assert s3.region == "us-west-2"


def test_s3_storage_prefix_stripped():
    s3 = S3Storage(bucket="b", prefix="/results/", region="us-east-1")
    assert s3.prefix == "results"


def test_s3_storage_write_json():
    s3 = S3Storage(bucket="test-bucket", prefix="eval")
    mock_s3_client = MagicMock()
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3_client

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        uri = s3.write_json(Path("runs/result.json"), {"score": 1})

    assert uri == "s3://test-bucket/eval/runs/result.json"
    mock_s3_client.put_object.assert_called_once()
    call_kwargs = mock_s3_client.put_object.call_args.kwargs
    assert call_kwargs["Bucket"] == "test-bucket"
    assert "eval/runs/result.json" in call_kwargs["Key"]
    assert call_kwargs["ContentType"] == "application/json"


def test_s3_storage_write_json_no_prefix():
    s3 = S3Storage(bucket="test-bucket", prefix="")
    mock_s3_client = MagicMock()
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3_client

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        uri = s3.write_json(Path("result.json"), {})

    assert uri == "s3://test-bucket/result.json"


def test_s3_storage_body_is_valid_json():
    s3 = S3Storage(bucket="b")
    mock_s3_client = MagicMock()
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3_client
    data = {"hello": "world"}

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        s3.write_json(Path("out.json"), data)

    body = mock_s3_client.put_object.call_args.kwargs["Body"]
    assert json.loads(body.decode("utf-8")) == data


# ── AzureBlobStorage ──────────────────────────────────────────────────────────

def test_azure_storage_requires_connection_string():
    with pytest.raises(ValueError, match="AZURE_STORAGE_CONNECTION_STRING"):
        AzureBlobStorage(container="results", connection_string=None)


def test_azure_storage_from_env(monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "DefaultEndpointsProtocol=https;AccountName=test")
    az = AzureBlobStorage(container="results")
    assert az.container == "results"


def test_azure_storage_explicit_connection_string():
    az = AzureBlobStorage(container="c", connection_string="conn-str-value")
    assert az.connection_string == "conn-str-value"


def test_azure_storage_prefix_stripped():
    az = AzureBlobStorage(container="c", prefix="/eval/", connection_string="cs")
    assert az.prefix == "eval"


def test_azure_storage_write_json():
    az = AzureBlobStorage(container="my-container", prefix="eval", connection_string="cs")
    mock_blob_client = MagicMock()
    mock_service_client = MagicMock()
    mock_service_client.get_blob_client.return_value = mock_blob_client

    mock_module = MagicMock()
    mock_module.BlobServiceClient.from_connection_string.return_value = mock_service_client

    with patch.dict("sys.modules", {"azure.storage.blob": mock_module}):
        uri = az.write_json(Path("runs/result.json"), {"score": 2})

    assert uri == "azure-blob://my-container/eval/runs/result.json"
    mock_blob_client.upload_blob.assert_called_once()


# ── make_storage ──────────────────────────────────────────────────────────────

def test_make_storage_local():
    backend = make_storage("local")
    assert isinstance(backend, LocalStorage)


def test_make_storage_s3():
    backend = make_storage("s3", s3_bucket="my-bucket")
    assert isinstance(backend, S3Storage)
    assert backend.bucket == "my-bucket"


def test_make_storage_s3_requires_bucket():
    with pytest.raises(ValueError, match="--s3-bucket"):
        make_storage("s3", s3_bucket="")


def test_make_storage_azure():
    backend = make_storage(
        "azure-blob",
        azure_container="c",
        azure_connection_string="cs",
    )
    assert isinstance(backend, AzureBlobStorage)


def test_make_storage_azure_requires_container():
    with pytest.raises(ValueError, match="--azure-container"):
        make_storage("azure-blob", azure_container="", azure_connection_string="cs")


def test_make_storage_unknown_backend():
    with pytest.raises(ValueError, match="Unknown storage backend"):
        make_storage("ftp")
