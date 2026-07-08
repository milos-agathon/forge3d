# python/forge3d/_gpu.py
# GPU capability helpers separated from the monolithic __init__ facade.
# Ensures detection logic can be reused by tests and other modules.
# RELEVANT FILES:python/forge3d/__init__.py,python/forge3d/_native.py,python/forge3d/_gpu.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._native import NATIVE_AVAILABLE, get_native_module, native_import_error

_REMEDIATION_NATIVE = (
    "Reinstall the compiled extension (pip install --force-reinstall forge3d) "
    "or rebuild it from a checkout with: maturin develop --release"
)
_REMEDIATION_ADAPTER = (
    "Verify GPU drivers are installed, pin a backend via WGPU_BACKENDS "
    "(vulkan|dx12|metal|gl), or install a software rasterizer for headless use "
    "(Windows ships WARP; on Linux install Mesa's lavapipe)."
)


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
    """Probe GPU availability without raising.

    The returned dict always carries a ``status`` key:

    - ``"ok"``: an adapter was found; adapter fields (``name``, ``backend``,
      ``device_type``, ``software_fallback``, ...) are present.
    - ``"no_adapter"``: the native module works but no hardware or software
      adapter is exposed for the requested backends.
    - ``"probe_error"``: the native probe itself failed; see ``reason``.
    - ``"native_missing"``: the compiled ``forge3d._forge3d`` extension did not
      import; ``reason`` preserves the original import error.

    Every non-``"ok"`` status includes ``reason`` and ``remediation`` fields.
    """
    _, native_probe, _ = _native_functions()
    if not callable(native_probe):
        import_error = native_import_error()
        reason = (
            f"forge3d._forge3d failed to import: {import_error!r}"
            if import_error is not None
            else "forge3d._forge3d is not available (extension not built) or lacks device_probe"
        )
        return {
            "status": "native_missing",
            "reason": reason,
            "remediation": _REMEDIATION_NATIVE,
        }
    try:
        if backend is not None:
            probe = native_probe(backend)
        else:
            probe = native_probe()
    except Exception as exc:
        return {
            "status": "probe_error",
            "reason": f"native device_probe raised: {exc!r}",
            "remediation": _REMEDIATION_ADAPTER,
        }
    if isinstance(probe, dict) and probe.get("status") not in (None, "ok"):
        probe.setdefault("reason", "no GPU adapter (hardware or software) was found")
        probe.setdefault("remediation", _REMEDIATION_ADAPTER)
    return probe


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
            del args, kwargs
            return MockVirtualTexture()

    return MockDevice()


class MockVirtualTexture:
    def __init__(self) -> None:
        self.width = 0
        self.height = 0

    def bind(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("Virtual textures require a GPU backend")
