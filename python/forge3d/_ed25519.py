"""Minimal Ed25519 signing and verification.

This is a pure-Python fallback used when the compiled Rust extension is
not available (e.g. during development or testing).  The primary
verification path is through the Rust binary via ``_forge3d.verify_license_signature``.

Implementation follows RFC 8032 Section 5.1.6-5.1.7.
"""

import hashlib

# -- Ed25519 curve constants --
_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P


def _sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


def _mod_inv(x: int, p: int = _P) -> int:
    return pow(x, p - 2, p)


def _recover_x(y: int) -> int:
    y2 = y * y % _P
    x2 = (y2 - 1) * _mod_inv(_D * y2 + 1) % _P
    if x2 == 0:
        return 0
    # Tonelli-Shanks for p = 5 mod 8
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        _I = pow(2, (_P - 1) // 4, _P)
        x = x * _I % _P
    if (x * x - x2) % _P != 0:
        raise ValueError("no square root")
    if x & 1:
        x = _P - x
    return x


# Base point
_By = 4 * _mod_inv(5) % _P
_Bx = _recover_x(_By)
_B = (_Bx, _By, 1, _Bx * _By % _P)


def _point_add(P: tuple, Q: tuple) -> tuple:
    x1, y1, z1, t1 = P
    x2, y2, z2, t2 = Q
    a = (y1 - x1) * (y2 - x2) % _P
    b = (y1 + x1) * (y2 + x2) % _P
    c = 2 * t1 * t2 * _D % _P
    d = 2 * z1 * z2 % _P
    e = b - a
    f = d - c
    g = d + c
    h = b + a
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _scalar_mult(s: int, P: tuple) -> tuple:
    Q = (0, 1, 1, 0)  # identity
    while s > 0:
        if s & 1:
            Q = _point_add(Q, P)
        P = _point_add(P, P)
        s >>= 1
    return Q


def _encode_point(P: tuple) -> bytes:
    x, y, z, _ = P
    zi = _mod_inv(z)
    x = x * zi % _P
    y = y * zi % _P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _decode_point(s: bytes) -> tuple:
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y)
    if (x & 1) != sign:
        x = _P - x
    t = x * y % _P
    return (x, y, 1, t)


def _clamped_scalar(seed: bytes) -> int:
    if len(seed) != 32:
        raise ValueError("Ed25519 private key seeds must be 32 bytes")
    h = bytearray(_sha512(seed))
    h[0] &= 248
    h[31] &= 63
    h[31] |= 64
    return int.from_bytes(h[:32], "little")


def public_key_from_private(private_key: bytes) -> bytes:
    """Derive a 32-byte Ed25519 public key from a 32-byte private seed."""
    a = _clamped_scalar(private_key)
    return _encode_point(_scalar_mult(a, _B))


def sign(private_key: bytes, message: bytes) -> bytes:
    """Sign *message* with a 32-byte Ed25519 private seed."""
    if len(private_key) != 32:
        raise ValueError("Ed25519 private key seeds must be 32 bytes")

    h = _sha512(private_key)
    a = _clamped_scalar(private_key)
    prefix = h[32:]
    public_key = _encode_point(_scalar_mult(a, _B))

    r = int.from_bytes(_sha512(prefix + message), "little") % _L
    R_bytes = _encode_point(_scalar_mult(r, _B))
    k = int.from_bytes(_sha512(R_bytes + public_key + message), "little") % _L
    s = (r + k * a) % _L
    return R_bytes + s.to_bytes(32, "little")


def verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """Verify an Ed25519 signature.

    Parameters
    ----------
    public_key : bytes
        32-byte Ed25519 public key.
    message : bytes
        The signed message.
    signature : bytes
        64-byte Ed25519 signature.

    Returns
    -------
    bool
        ``True`` when valid, ``False`` otherwise.
    """
    if len(public_key) != 32 or len(signature) != 64:
        return False

    try:
        A = _decode_point(public_key)
    except (ValueError, ZeroDivisionError):
        return False

    R_bytes = signature[:32]
    try:
        R = _decode_point(R_bytes)
    except (ValueError, ZeroDivisionError):
        return False

    s = int.from_bytes(signature[32:], "little")
    if s >= _L:
        return False

    h = int.from_bytes(
        _sha512(R_bytes + public_key + message), "little"
    ) % _L

    # Check:  [s]B == R + [h]A
    sB = _scalar_mult(s, _B)
    hA = _scalar_mult(h, A)
    RhA = _point_add(R, hA)

    return _encode_point(sB) == _encode_point(RhA)
