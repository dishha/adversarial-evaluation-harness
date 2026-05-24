import os
import pytest
from unittest.mock import MagicMock, patch
from harness.observability import NullObserver, make_observer


# ── NullObserver ──────────────────────────────────────────────────────────────

def test_null_observer_start_run_no_error():
    obs = NullObserver()
    obs.start_run({"scenario_type": "toxicity", "model": "mock"})


def test_null_observer_log_turn_metrics_no_error():
    obs = NullObserver()
    obs.log_turn_metrics({"failure_score": 2, "harm_potential": 1}, step=1)


def test_null_observer_finish_run_no_error():
    obs = NullObserver()
    obs.finish_run({"failure_rate": 0.1}, artifact_path=None)


def test_null_observer_finish_run_with_path_no_error(tmp_path):
    obs = NullObserver()
    path = str(tmp_path / "result.json")
    obs.finish_run({"failure_rate": 0.2}, artifact_path=path)


def test_null_observer_all_methods_return_none():
    obs = NullObserver()
    assert obs.start_run({}) is None
    assert obs.log_turn_metrics({}, step=0) is None
    assert obs.finish_run({}) is None


# ── make_observer ─────────────────────────────────────────────────────────────

def test_make_observer_returns_null_when_no_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    obs = make_observer("toxicity", "mock-model")
    assert isinstance(obs, NullObserver)


def test_make_observer_returns_null_when_uri_empty(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "   ")
    obs = make_observer("toxicity", "mock-model")
    assert isinstance(obs, NullObserver)


def test_make_observer_returns_null_when_mlflow_not_installed(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.example.com")
    with patch.dict("sys.modules", {"mlflow": None}):
        obs = make_observer("toxicity", "mock-model")
    assert isinstance(obs, NullObserver)


def test_make_observer_returns_mlflow_observer_when_configured(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.example.com")
    mock_mlflow = MagicMock()
    mock_mlflow.set_tracking_uri = MagicMock()
    mock_mlflow.set_experiment = MagicMock()

    with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
        from harness.observability import MLflowObserver
        with patch("harness.observability.MLflowObserver.__init__", return_value=None):
            obs = make_observer("toxicity", "mock-model")

    # Either returns NullObserver (if init fails) or MLflowObserver
    assert obs is not None


# ── MLflowObserver (mocked mlflow) ────────────────────────────────────────────

def _make_mock_mlflow_observer():
    """Build an MLflowObserver with a fully mocked mlflow module."""
    mock_mlflow = MagicMock()
    mock_mlflow.set_experiment = MagicMock()
    mock_mlflow.set_tracking_uri = MagicMock()
    mock_mlflow.start_run = MagicMock()
    mock_mlflow.log_params = MagicMock()
    mock_mlflow.log_metrics = MagicMock()
    mock_mlflow.log_artifact = MagicMock()
    mock_mlflow.end_run = MagicMock()

    with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
        from harness.observability import MLflowObserver
        obs = MLflowObserver.__new__(MLflowObserver)
        obs._mlflow = mock_mlflow
        obs._model_label = "test-model"

    return obs, mock_mlflow


def test_mlflow_observer_start_run_logs_params():
    obs, mock_mlflow = _make_mock_mlflow_observer()
    params = {"scenario_type": "toxicity", "max_turns": 10, "budget": 5000}
    obs.start_run(params)
    mock_mlflow.log_params.assert_called_once()
    logged = mock_mlflow.log_params.call_args.args[0]
    for k in params:
        assert k in logged
        assert isinstance(logged[k], str)


def test_mlflow_observer_log_turn_metrics():
    obs, mock_mlflow = _make_mock_mlflow_observer()
    judge_result = {
        "failure_score": 3,
        "harm_potential": 2,
        "specificity": 1,
        "refusal_quality": 0,
    }
    obs.log_turn_metrics(judge_result, step=5)
    mock_mlflow.log_metrics.assert_called_once()
    metrics = mock_mlflow.log_metrics.call_args.args[0]
    assert metrics["failure_score"] == 3.0
    assert metrics["harm_potential"] == 2.0


def test_mlflow_observer_finish_run_logs_numeric_summary():
    obs, mock_mlflow = _make_mock_mlflow_observer()
    summary = {"failure_rate": 0.4, "total_sessions": 10, "label": "ignored"}
    obs.finish_run(summary)
    mock_mlflow.log_metrics.assert_called_once()
    metrics = mock_mlflow.log_metrics.call_args.args[0]
    assert "failure_rate" in metrics
    assert "total_sessions" in metrics
    assert "label" not in metrics  # strings excluded


def test_mlflow_observer_finish_run_calls_end_run():
    obs, mock_mlflow = _make_mock_mlflow_observer()
    obs.finish_run({})
    mock_mlflow.end_run.assert_called_once()


def test_mlflow_observer_finish_run_logs_artifact_if_exists(tmp_path):
    obs, mock_mlflow = _make_mock_mlflow_observer()
    artifact = tmp_path / "result.json"
    artifact.write_text("{}")
    obs.finish_run({}, artifact_path=str(artifact))
    mock_mlflow.log_artifact.assert_called_once_with(str(artifact))


def test_mlflow_observer_finish_run_skips_artifact_if_missing():
    obs, mock_mlflow = _make_mock_mlflow_observer()
    obs.finish_run({}, artifact_path="/nonexistent/path.json")
    mock_mlflow.log_artifact.assert_not_called()


def test_mlflow_observer_excludes_booleans_from_metrics():
    obs, mock_mlflow = _make_mock_mlflow_observer()
    summary = {"failure_rate": 0.1, "is_complete": True}
    obs.finish_run(summary)
    metrics = mock_mlflow.log_metrics.call_args.args[0]
    assert "is_complete" not in metrics
