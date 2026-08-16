import pytest

from sense_symbolic.app import DEFAULT_BACKEND, _default_backend, create_app


def test_default_backend_is_z3():
    assert DEFAULT_BACKEND == "z3"


def test_create_app_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown symbolic backend"):
        create_app("sql")


def test_default_backend_reads_env_var(monkeypatch):
    monkeypatch.setenv("SENSE_SYMBOLIC_BACKEND", "sparql")
    assert _default_backend() == "sparql"


def test_default_backend_falls_back_to_z3_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("SENSE_SYMBOLIC_BACKEND", raising=False)
    assert _default_backend() == "z3"


def test_default_backend_raises_on_unrecognized_env_var(monkeypatch):
    monkeypatch.setenv("SENSE_SYMBOLIC_BACKEND", "not-a-real-backend")
    with pytest.raises(ValueError, match="unknown symbolic backend"):
        _default_backend()
