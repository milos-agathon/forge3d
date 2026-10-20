# python/forge3d/_gpu.py
# GPU capability helpers separated from the monolithic __init__ facade.
# Ensures detection logic can be reused by tests and other modules.
# RELEVANT FILES:python/forge3d/__init__.py,python/forge3d/_native.py,python/forge3d/_gpu.py

from __future__ import annotations

from typing import Dict, List, Optional

from ._native import get_native_module, get_native_symbol, native_import_error

_REMEDIATION_NATIVE = (
    "Reinstall the compiled extension (pip install --force-reinstall forge3d) "
    "or rebuild it from a checkout with: maturin develop --release"
)
_REMEDIATION_ADAPTER = (
    "Verify GPU drivers are installed, pin a backend via WGPU_BACKENDS "
    "(vulkan|dx12|metal|gl), or install a software rasterizer for headless use "
    "(Windows ships WARP; on Linux install Mesa's lavapipe)."
)


def enumerate_adapters() -> List[Dict]:
    native_enum = get_native_symbol("enumerate_adapters")
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
    native_probe = get_native_symbol("device_probe")
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


def determinism_probe() -> Dict:
    """Run the TERRA-DETERMINATA arithmetic canary on the active adapter.

    Returns a dict with ``status`` (``"ok"``/``"probe_error"``/``"native_missing"``),
    ``probe_sha256`` (SHA-256 of the 256-byte canary buffer), ``probe_bytes_hex``,
    ``wgsl_sha256`` (SHA-256 of the assembled probe WGSL source), and adapter
    metadata (``adapter_name``, ``adapter_backend``, ``software_fallback``, ...).

    The same WGSL runs in browser WebGPU via
    ``tools/determinism_browser/det_probe.js``, making ``probe_sha256`` a real
    cross-implementation arithmetic-identity leg. Requires the GPU context to
    be initialized already (backend pinned before first init in deterministic
    mode).
    """
    native = get_native_module()
    native_probe = getattr(native, "determinism_probe", None) if native is not None else None
    if not callable(native_probe):
        import_error = native_import_error()
        reason = (
            f"forge3d._forge3d failed to import: {import_error!r}"
            if import_error is not None
            else "forge3d._forge3d is not available (extension not built) or lacks determinism_probe"
        )
        return {
            "status": "native_missing",
            "reason": reason,
            "remediation": _REMEDIATION_NATIVE,
        }
    try:
        return native_probe()
    except Exception as exc:
        return {
            "status": "probe_error",
            "reason": f"native determinism_probe raised: {exc!r}",
            "remediation": _REMEDIATION_ADAPTER,
        }


def has_gpu() -> bool:
    if get_native_module() is None:
        return False
    adapters = enumerate_adapters()
    if adapters:
        return True
    probe = device_probe()
    return isinstance(probe, dict) and probe.get("status") == "ok"
