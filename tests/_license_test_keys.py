"""Test helpers for generating properly signed license keys.

Uses the DEVELOPMENT keypair whose public key is embedded in the binary.
This private key is intentionally included in the test suite — it is a
development key, not the production signing key.
"""

_DEV_PRIVATE_KEY_HEX = "469ba4e901ab3b0dee8b0a59eb390bcbe16230355d0995fe2d5d1d86024414c4"
_DEFAULT_CUSTOMER_ID = "forge3d-ci"


def sign_test_key(
    tier: str,
    expiry: str,
    customer_id: str = _DEFAULT_CUSTOMER_ID,
) -> str:
    """Return a complete license key string with a valid Ed25519 signature.

    Parameters
    ----------
    tier : str
        ``"PRO"`` or ``"ENTERPRISE"``.
    expiry : str
        Date in ``YYYYMMDD`` format.
    customer_id : str
        Stable customer identifier included in the signed payload.
    """
    from forge3d import _ed25519

    priv_bytes = bytes.fromhex(_DEV_PRIVATE_KEY_HEX)
    message = f"F3D-{tier}-{customer_id}-{expiry}".encode()
    sig = _ed25519.sign(priv_bytes, message)
    return f"F3D-{tier}-{customer_id}-{expiry}-{sig.hex()}"
