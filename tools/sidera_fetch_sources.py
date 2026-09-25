"""Fetch the exact official source bytes pinned in SIDERA's manifest.

This is an offline asset-maintenance tool, not a runtime dependency. Network
access is needed only when regenerating the committed compact binaries.
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/sidera/MANIFEST.md"
CACHE = ROOT / "data/sidera/_source_cache"


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \| (https://[^ |]+) \|$", MANIFEST.read_text(encoding="utf-8"), re.M)
    if len(rows) < 40:
        raise ValueError("SIDERA source manifest is incomplete")
    for name, digest, url in rows:
        target = CACHE / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            continue
        content = urllib.request.urlopen(url).read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"source digest changed: {name} from {url}")
        target.write_bytes(content)
        print(name, len(content))


if __name__ == "__main__":
    main()
