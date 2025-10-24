# tests/test_session.py
# Unit tests for the PyO3 Session wrapper
# Ensures GPU session metadata is exposed to Python callers
# RELEVANT FILES: src/session.rs, src/gpu.rs, python/forge3d/__init__.py, tests/test_terrain_params.py
import pytest

import forge3d as f3d


if not f3d.has_gpu():
    pytest.skip("GPU adapter not available", allow_module_level=True)
if not hasattr(f3d, "Session"):
    pytest.skip("Session not available in this build", allow_module_level=True)


def test_session_creation() -> None:
    session = f3d.Session(window=False)
    assert isinstance(session.info(), str)


def test_session_properties() -> None:
    session = f3d.Session(window=False)
    assert session.adapter_name
    assert session.backend
    assert session.device_type


def test_window_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        f3d.Session(window=True)
