#!/usr/bin/env python3
"""Check the public forge3d install and purchase funnel."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request


DOCS_URL = "https://milos-agathon.github.io/forge3d/"
REPO_URL = "https://github.com/milos-agathon/forge3d"
LICENSE_INFO_URL = f"{REPO_URL}#license"

TARGETS = [
    ("homepage", REPO_URL),
    ("docs", DOCS_URL),
    ("license", LICENSE_INFO_URL),
    ("pypi", "https://pypi.org/project/forge3d/"),
    (
        "datasets",
        "https://raw.githubusercontent.com/milos-agathon/forge3d/main/assets/tif/dem_rainier.tif",
    ),
]


def _check_url(name: str, url: str) -> tuple[bool, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "forge3d-public-funnel-monitor/1.0",
            "Range": "bytes=0-0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            status = getattr(response, "status", response.getcode())
            if 200 <= status < 400:
                return True, f"{name}: {status} {url}"
            return False, f"{name}: unexpected status {status} {url}"
    except urllib.error.HTTPError as exc:
        return False, f"{name}: HTTP {exc.code} {url}"
    except urllib.error.URLError as exc:
        return False, f"{name}: URL error {exc.reason} {url}"


def main() -> int:
    failures: list[str] = []
    for name, url in TARGETS:
        ok, message = _check_url(name, url)
        print(message)
        if not ok:
            failures.append(message)

    if failures:
        print("\nPublic funnel check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nPublic funnel check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
