#!/usr/bin/env python3
"""Sign a forge3d license key using the Ed25519 private key.

Usage:
    python scripts/sign_license_key.py --key-file forge3d_license.key --tier PRO --customer acme-co --expiry 20261231
    python scripts/sign_license_key.py --key-hex <hex> --tier ENTERPRISE --customer newsroom-42 --expiry 20270630

The output is a complete license key string ready for:
    forge3d.set_license_key("F3D-PRO-acme-co-20261231-<signature>")
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Sign a forge3d license key")
    key_group = parser.add_mutually_exclusive_group(required=True)
    key_group.add_argument("--key-file", help="Path to private key file (hex)")
    key_group.add_argument("--key-hex", help="Private key as hex string")
    parser.add_argument(
        "--tier",
        required=True,
        choices=["PRO", "ENTERPRISE"],
        help="License tier",
    )
    parser.add_argument(
        "--customer",
        required=True,
        help="Stable customer identifier embedded in the signed payload",
    )
    parser.add_argument(
        "--expiry",
        required=True,
        help="Expiry date in YYYYMMDD format",
    )
    args = parser.parse_args()

    # Load private key
    if args.key_file:
        with open(args.key_file) as f:
            priv_hex = f.read().strip()
    else:
        priv_hex = args.key_hex.strip()

    try:
        priv_bytes = bytes.fromhex(priv_hex)
    except ValueError:
        print("ERROR: Invalid hex in private key", file=sys.stderr)
        sys.exit(1)

    if len(priv_bytes) != 32:
        print(f"ERROR: Private key must be 32 bytes, got {len(priv_bytes)}", file=sys.stderr)
        sys.exit(1)

    # Validate expiry format
    if len(args.expiry) != 8 or not args.expiry.isdigit():
        print("ERROR: Expiry must be YYYYMMDD format", file=sys.stderr)
        sys.exit(1)

    if not args.customer.strip():
        print("ERROR: Customer identifier must be non-empty", file=sys.stderr)
        sys.exit(1)

    repo_python = Path(__file__).resolve().parents[1] / "python"
    if str(repo_python) not in sys.path:
        sys.path.insert(0, str(repo_python))
    from forge3d import _ed25519

    # Sign
    message = f"F3D-{args.tier}-{args.customer}-{args.expiry}".encode()
    signature = _ed25519.sign(priv_bytes, message)
    sig_hex = signature.hex()

    license_key = f"F3D-{args.tier}-{args.customer}-{args.expiry}-{sig_hex}"
    print(license_key)


if __name__ == "__main__":
    main()
