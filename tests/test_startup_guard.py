"""Startup guard tests for multi-worker deployments."""

import pytest

from musicmixer import main


@pytest.fixture(autouse=True)
def _clear_worker_env(monkeypatch):
    """Keep worker env vars isolated per test."""
    monkeypatch.delenv("UVICORN_WORKERS", raising=False)
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)


def test_startup_guard_raises_for_uvicorn_workers_gt_one(monkeypatch):
    monkeypatch.setattr(
        main.settings, "distributed_limiter_enabled", False, raising=False
    )
    monkeypatch.setenv("UVICORN_WORKERS", "2")

    with pytest.raises(RuntimeError, match="UVICORN_WORKERS=2"):
        main._enforce_distributed_limiter_startup_guard()


def test_startup_guard_raises_for_web_concurrency_gt_one(monkeypatch):
    monkeypatch.setattr(
        main.settings, "distributed_limiter_enabled", False, raising=False
    )
    monkeypatch.setenv("WEB_CONCURRENCY", "3")

    with pytest.raises(RuntimeError, match="WEB_CONCURRENCY=3"):
        main._enforce_distributed_limiter_startup_guard()


def test_startup_guard_includes_both_env_vars_when_both_are_set(monkeypatch):
    monkeypatch.setattr(
        main.settings, "distributed_limiter_enabled", False, raising=False
    )
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    monkeypatch.setenv("WEB_CONCURRENCY", "3")

    with pytest.raises(RuntimeError) as exc_info:
        main._enforce_distributed_limiter_startup_guard()

    error_message = str(exc_info.value)
    assert "UVICORN_WORKERS=2" in error_message
    assert "WEB_CONCURRENCY=3" in error_message
    assert "DISTRIBUTED_LIMITER_ENABLED=true" in error_message


def test_startup_guard_allows_multi_worker_when_distributed_limiter_enabled(monkeypatch):
    monkeypatch.setattr(
        main.settings, "distributed_limiter_enabled", True, raising=False
    )
    monkeypatch.setenv("WEB_CONCURRENCY", "4")

    main._enforce_distributed_limiter_startup_guard()


def test_startup_guard_ignores_single_worker_or_invalid_values(monkeypatch):
    monkeypatch.setattr(
        main.settings, "distributed_limiter_enabled", False, raising=False
    )
    monkeypatch.setenv("UVICORN_WORKERS", "1")
    monkeypatch.setenv("WEB_CONCURRENCY", "invalid")

    main._enforce_distributed_limiter_startup_guard()
