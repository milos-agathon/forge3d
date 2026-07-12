from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bidi_conformance() -> None:
    result = subprocess.run(
        [
            "cargo",
            "test",
            "bidi_conformance_corpus",
            "--lib",
            "--features",
            "extension-module",
            "--",
            "--nocapture",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    match = re.search(
        r"BidiCharacterTest\.txt=(\d+) BidiTest\.txt=(\d+) failures=(\d+)", output
    )
    assert match is not None, output
    character_lines, bidi_lines, failures = map(int, match.groups())
    assert character_lines >= 2_000
    assert bidi_lines >= 2_000
    assert failures == 0
