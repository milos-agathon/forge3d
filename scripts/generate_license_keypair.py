#!/usr/bin/env python3
"""Generate an Ed25519 keypair for forge3d license signing.

Usage:
    python scripts/generate_license_keypair.py [--out-dir DIR]

Outputs two files:
    forge3d_license.key     - 32-byte private key (hex, keep secret!)
    forge3d_license.pub     - 32-byte public key  (hex, embed in code)

The public key must be embedded in:
    1. src/license/mod.rs   (PUBLIC_KEY_BYTES constant)
    2. python/forge3d/_license.py (_PUBLIC_KEY_HEX constant)
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate Ed25519 license keypair")
    parser.add_argument(
        "--out-dir", default=".", help="Directory for output files (default: cwd)"
    )
    args = parser.parse_args()

    repo_python = Path(__file__).resolve().parents[1] / "python"
    if str(repo_python) not in sys.path:
        sys.path.insert(0, str(repo_python))
    from forge3d import _ed25519

    priv_bytes = os.urandom(32)
    pub_bytes = _ed25519.public_key_from_private(priv_bytes)

    priv_hex = priv_bytes.hex()
    pub_hex = pub_bytes.hex()

    priv_path = os.path.join(args.out_dir, "forge3d_license.key")
    pub_path = os.path.join(args.out_dir, "forge3d_license.pub")

    with open(priv_path, "w") as f:
        f.write(priv_hex + "\n")
    os.chmod(priv_path, 0o600)

    with open(pub_path, "w") as f:
        f.write(pub_hex + "\n")

    # Print Rust constant for easy embedding
    rust_array = ", ".join(f"0x{b:02x}" for b in pub_bytes)

    print(f"Private key: {priv_path}")
    print(f"Public key:  {pub_path}")
    print()
    print("Embed in src/license/mod.rs:")
    print(f"const PUBLIC_KEY_BYTES: [u8; 32] = [{rust_array}];")
    print()
    print("Embed in python/forge3d/_license.py:")
    print(f'_PUBLIC_KEY_HEX = "{pub_hex}"')


if __name__ == "__main__":
    main()
