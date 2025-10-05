# python/forge3d/_gpu.py
# GPU capability helpers separated from the monolithic __init__ facade.
# Ensures detection logic can be reused by tests and other modules.
# RELEVANT FILES:python/forge3d/__init__.py,python/forge3d/_native.py,python/forge3d/_gpu.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._native import NATIVE_AVAILABLE, get_native_module


def _native_functions() -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
    native = get_native_module()
    if native is None:
        return None, None, None
    return (
        getattr(native, "enumerate_adapters", None),
        getattr(native, "device_probe", None),
        getattr(native, "get_device", None),
    )


def enumerate_adapters() -> List[Dict]:
    native_enum, native_probe, _ = _native_functions()
    if callable(native_enum):
        try:
            adapters = native_enum()
            if isinstance(adapters, list) and adapters:
                return adapters
        except Exception:
            adapters = []
    else:
        adapters = []
    probe = device_probe()
    if isinstance(probe, dict) and probe.get("status") == "ok":
        return [probe]
    return adapters


def device_probe(backend: Optional[str] = None) -> Dict:
    _, native_probe, _ = _native_functions()
    if callable(native_probe):
        try:
            if backend is not None:
                return native_probe(backend)
            return native_probe()
        except Exception:
            pass
    return {"status": "unavailable"}


def has_gpu() -> bool:
    if not NATIVE_AVAILABLE:
        return False
    adapters = enumerate_adapters()
    if adapters:
        return True
    probe = device_probe()
    return isinstance(probe, dict) and probe.get("status") == "ok"


def get_device() -> Any:
    _, _, native_get_device = _native_functions()
    if callable(native_get_device):
        try:
            return native_get_device()
        except Exception:
            pass

    class MockDevice:
        def __init__(self) -> None:
            self.name = "Fallback CPU Device"
            self.backend = "cpu"
            self.limits = {"max_texture_dimension": 16384}

        def create_virtual_texture(self, *args: Any, **kwargs: Any) -> "MockVirtualTexture":
            return MockVirtualTexture()

    return MockDevice()


class MockVirtualTexture:
    def __init__(self) -> None:
        self.width = 0
        self.height = 0

    def bind(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("Virtual textures require a GPU backend")
