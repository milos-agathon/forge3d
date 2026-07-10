# python/forge3d/certificate.py
# CENSOR Task 10: native signing, pure-Python offline verification, and CLI for the
# forge3d RenderCertificate. The unsigned execution report is assembled by the
# native runtime (src/core/certificate.rs, exposed as
# forge3d._forge3d.render_execution_report); this module requests the Ed25519
# seal from Rust and provides a verifier that needs neither native code nor numpy.
# RELEVANT FILES: python/forge3d/_ed25519.py, python/forge3d/diagnostics.py,
# python/forge3d/provenance.py, src/core/certificate.rs

"""Sign and verify forge3d RenderCertificates (schema
``forge3d.render_certificate/1``).

Verification is intentionally pure Python and depends only on the standard
library plus the RFC 8032 Ed25519 verifier in :mod:`forge3d._ed25519`. It needs
neither the compiled ``_forge3d`` extension nor numpy, so a third party can
re-verify a signed certificate offline with a stock CPython interpreter::

    python -m forge3d.certificate verify cert.json --pubkey key.pub

Signing contract
----------------
The signed message is::

    SIGN_CONTEXT + sha256(canonical_payload_bytes(cert))

where :func:`canonical_payload_bytes` produces the *signed view* of the
certificate: the ``"signature"`` block is removed entirely and the per-pass
``"gpu_ms"`` live-timing values are removed (they are the only nondeterministic
fields in the native report, so excluding them lets the signature survive
re-timing the same render). The remaining structure is serialized with
``json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False)`` for a canonical, byte-stable payload.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping, Union

from . import _ed25519
from ._canonical_json import canonical_json_bytes

__all__ = [
    "SIGN_CONTEXT",
    "DEV_SIGNING_SEED",
    "SCHEMA",
    "canonical_payload_bytes",
    "emit_render_certificate",
    "payload_sha256",
    "sign_certificate",
    "verify",
    "write_certificate",
]

# Domain-separation tag for the Ed25519 message. Never reused across schemas.
SIGN_CONTEXT = b"forge3d.render_certificate.v1"

# Certificate schema this module signs/verifies.
SCHEMA = "forge3d.render_certificate/1"

# Environment variable that overrides the dev signing seed with a 64-char hex
# (32-byte) Ed25519 private seed.
_SIGNING_KEY_ENV = "FORGE3D_CERT_SIGNING_KEY"
_CAPTURE_STATE = threading.local()


@contextmanager
def _render_capture(entry_point: str, pass_label: str, draw_calls: int = 1):
    """Capture one pure-Python render without overwriting an outer capture."""
    depth = int(getattr(_CAPTURE_STATE, "depth", 0))
    _CAPTURE_STATE.depth = depth + 1
    if depth:
        try:
            yield
        finally:
            _CAPTURE_STATE.depth = depth
        return

    from ._native import get_native_module

    native = get_native_module()
    if native is None or not hasattr(native, "begin_render_execution_capture"):
        try:
            yield
        finally:
            _CAPTURE_STATE.depth = 0
        return

    native.begin_render_execution_capture(entry_point)
    try:
        yield
    except BaseException:
        native.abort_render_execution_capture()
        raise
    else:
        native.finish_render_execution_capture(pass_label, int(draw_calls))
    finally:
        _CAPTURE_STATE.depth = 0


def _dev_signing_seed() -> bytes:
    """Derive the fixed development signing seed.

    WARNING: this is a NON-PRODUCTION development key. It is deterministically
    derived from a public constant so tests and local tooling can produce and
    re-verify signatures without provisioning a keypair. Anything that needs a
    trustworthy signature MUST supply a real 32-byte seed via the
    ``seed`` argument to :func:`sign_certificate` or the
    ``FORGE3D_CERT_SIGNING_KEY`` environment variable (64-char hex).
    """
    return hashlib.sha256(b"forge3d.certificate.dev-signing-key.v1").digest()[:32]


#: Fixed 32-byte development signing seed. NON-PRODUCTION — see
#: :func:`_dev_signing_seed`. Overridable per call or via the
#: ``FORGE3D_CERT_SIGNING_KEY`` environment variable.
DEV_SIGNING_SEED: bytes = _dev_signing_seed()


def _resolve_seed(seed: Union[bytes, bytearray, None]) -> bytes:
    """Resolve the signing seed: explicit arg > env override > dev seed."""
    if seed is not None:
        seed = bytes(seed)
        if len(seed) != 32:
            raise ValueError("signing seed must be a 32-byte Ed25519 seed")
        return seed
    env = os.environ.get(_SIGNING_KEY_ENV)
    if env:
        env = env.strip()
        try:
            resolved = bytes.fromhex(env)
        except ValueError as exc:
            raise ValueError(
                f"{_SIGNING_KEY_ENV} must be 64-char hex (a 32-byte Ed25519 seed)"
            ) from exc
        if len(resolved) != 32:
            raise ValueError(
                f"{_SIGNING_KEY_ENV} must be 64-char hex (a 32-byte Ed25519 seed)"
            )
        return resolved
    return DEV_SIGNING_SEED


def canonical_payload_bytes(cert: Mapping[str, Any]) -> bytes:
    """Return the canonical, byte-stable *signed view* of ``cert``.

    Removes the ``"signature"`` block and every ``passes[*]["gpu_ms"]`` value,
    then serializes deterministically. Raises ``ValueError`` if the payload
    contains non-finite floats (``allow_nan=False``).
    """
    payload: Dict[str, Any] = copy.deepcopy(dict(cert))
    payload.pop("signature", None)
    passes = payload.get("passes")
    if isinstance(passes, list):
        for element in passes:
            if isinstance(element, dict):
                element.pop("gpu_ms", None)
    return canonical_json_bytes(payload, error_context="RenderCertificate serialization")


def _payload_digest(cert: Mapping[str, Any]) -> bytes:
    return hashlib.sha256(canonical_payload_bytes(cert)).digest()


def payload_sha256(cert: Mapping[str, Any]) -> str:
    """Hex SHA256 over :func:`canonical_payload_bytes`."""
    return _payload_digest(cert).hex()


def _signed_fields(cert: Mapping[str, Any]) -> list[str]:
    """Sorted top-level keys of the signed view, annotating ``passes`` to make
    the ``gpu_ms`` exclusion explicit to a reader of the signature block."""
    fields = sorted(str(key) for key in cert.keys() if key != "signature")
    return ["passes (gpu_ms excluded)" if key == "passes" else key for key in fields]


def sign_certificate(
    cert: Mapping[str, Any], seed: Union[bytes, bytearray, None] = None
) -> Dict[str, Any]:
    """Return a copy of ``cert`` with an Ed25519 ``"signature"`` block added.

    The message signed is ``SIGN_CONTEXT + sha256(canonical_payload_bytes)``.
    ``seed`` (32-byte Ed25519 seed) defaults to the ``FORGE3D_CERT_SIGNING_KEY``
    env override, then the non-production :data:`DEV_SIGNING_SEED`. Signing is
    deterministic: the same ``cert`` and ``seed`` always yield the same
    signature bytes.
    """
    private_key = _resolve_seed(seed)
    signed = copy.deepcopy(dict(cert))
    signed.pop("signature", None)

    from ._native import get_native_module

    native = get_native_module()
    if native is None or not hasattr(native, "sign_render_certificate_digest"):
        raise RuntimeError(
            "certificate signing requires the compiled forge3d native module "
            "with ed25519-dalek support"
        )
    signature, public_key = native.sign_render_certificate_digest(
        private_key, hashlib.sha256(canonical_payload_bytes(signed)).digest()
    )

    signed["signature"] = {
        "alg": "ed25519",
        "pubkey": public_key,
        "sig": signature,
        "signed_fields": _signed_fields(signed),
    }
    return signed


def _pubkey_from_text(text: str) -> bytes:
    key = bytes.fromhex(text.strip())
    if len(key) != 32:
        raise ValueError("public key must decode to 32 bytes")
    return key


def _pubkey_from_file(path: Path) -> bytes:
    raw = Path(path).read_bytes()
    if len(raw) == 32:
        return raw
    return _pubkey_from_text(raw.decode("ascii"))


def _resolve_pubkey(pubkey: Union[bytes, bytearray, str, "os.PathLike[str]"]) -> bytes:
    """Resolve ``pubkey`` to raw 32 bytes.

    Accepts raw 32 bytes, a 64-char hex string, or a path (``str`` /
    ``PathLike``) to a file containing either hex text or 32 raw bytes. A
    ``str`` that names an existing file is read from disk; otherwise it is
    interpreted as hex.
    """
    if isinstance(pubkey, (bytes, bytearray)):
        raw = bytes(pubkey)
        if len(raw) == 32:
            return raw
        return _pubkey_from_text(raw.decode("ascii"))
    if isinstance(pubkey, os.PathLike):
        return _pubkey_from_file(Path(pubkey))
    if isinstance(pubkey, str):
        candidate = Path(pubkey)
        try:
            is_file = candidate.is_file()
        except OSError:
            is_file = False
        if is_file:
            return _pubkey_from_file(candidate)
        return _pubkey_from_text(pubkey)
    raise TypeError("pubkey must be bytes, a hex str, or a path to a key file")


def verify(
    path: Union[str, "os.PathLike[str]"],
    pubkey: Union[bytes, bytearray, str, "os.PathLike[str]"],
) -> bool:
    """Verify the certificate at ``path`` against the trusted ``pubkey``.

    ``pubkey`` may be raw 32 bytes, a 64-char hex string, or a path to a file
    containing hex text or 32 raw bytes. Verification uses the CALLER-SUPPLIED
    public key, not the one embedded in the certificate, so tampering with the
    embedded ``pubkey`` cannot make an invalid certificate verify.

    Returns ``True`` only when the signature is valid over the canonical signed
    view. Any structural error (missing file, malformed JSON, missing signature
    block, bad hex, unusable key) returns ``False`` — this never raises for a
    malformed certificate.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
        cert = json.loads(text)
        if not isinstance(cert, dict):
            return False
        signature_block = cert.get("signature")
        if not isinstance(signature_block, dict):
            return False
        if signature_block.get("alg") != "ed25519":
            return False
        sig = bytes.fromhex(str(signature_block["sig"]))
        public_key = _resolve_pubkey(pubkey)
        message = SIGN_CONTEXT + hashlib.sha256(canonical_payload_bytes(cert)).digest()
        return bool(_ed25519.verify(public_key, message, sig))
    except Exception:
        return False


def emit_render_certificate(
    certificate: "bool | str | os.PathLike[str] | None",
) -> "str | None":
    """Handle a ``certificate=`` render kwarg for the LAST completed render.

    ``certificate`` follows the render entry-point contract:

    * a falsy value (``False``, ``None``, ``""``) → no certificate is built and
      ``None`` is returned;
    * ``True`` → a signed certificate is assembled via
      :func:`forge3d.diagnostics.render_certificate` and its payload SHA256 is
      returned (nothing is written to disk);
    * a path (``str`` / :class:`os.PathLike`) → the signed certificate is
      assembled and written there via :func:`write_certificate`, and its
      payload SHA256 is returned.

    The returned value is the deterministic ``payload_sha256`` of the signed
    certificate, suitable for stashing into render metadata. Requires a
    completed native render in this process (see
    :func:`forge3d.diagnostics.render_certificate`).
    """
    if not certificate:
        return None
    from . import diagnostics as _diagnostics

    cert = _diagnostics.render_certificate()
    if not isinstance(certificate, bool):
        write_certificate(cert, os.fspath(certificate))
    return payload_sha256(cert)


def write_certificate(
    cert: Mapping[str, Any], path: Union[str, "os.PathLike[str]"]
) -> None:
    """Write ``cert`` to ``path`` as human-diffable, deterministic JSON.

    Uses ``indent=2, sort_keys=True`` with a trailing newline. The signed view
    is derived by :func:`canonical_payload_bytes`, so this pretty-printing is
    free to reorder keys without affecting verification.
    """
    text = json.dumps(cert, indent=2, sort_keys=True) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def _main(argv: "list[str] | None" = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m forge3d.certificate",
        description="Sign and verify forge3d RenderCertificates offline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify", help="Verify a signed certificate against a trusted public key."
    )
    verify_parser.add_argument("cert", help="Path to the certificate JSON file.")
    verify_parser.add_argument(
        "--pubkey",
        required=True,
        help="Trusted Ed25519 public key: 64-char hex, or a path to a file "
        "containing hex text or 32 raw bytes.",
    )

    args = parser.parse_args(argv)

    if args.command == "verify":
        ok = verify(args.cert, args.pubkey)
        print("VALID" if ok else "INVALID")
        return 0 if ok else 1
    parser.error(f"unknown command: {args.command!r}")
    return 2  # unreachable; parser.error exits


if __name__ == "__main__":
    import sys

    sys.exit(_main())
