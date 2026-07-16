from __future__ import annotations

import hashlib
import importlib.util
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_table_generator_module():
    path = ROOT / "src" / "labels" / "shape" / "generate_unicode_tables.py"
    spec = importlib.util.spec_from_file_location("generate_unicode_tables", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bidi_conformance() -> None:
    result = subprocess.run(
        [
            "cargo",
            "test",
            "bidi_conformance_corpus",
            "--lib",
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


@pytest.mark.parametrize("corrupted_name", ["BidiBrackets.txt", "emoji-data.txt"])
def test_table_generator_rejects_each_corrupted_unicode_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupted_name: str
) -> None:
    generator = _load_table_generator_module()
    brackets = tmp_path / "BidiBrackets.txt"
    emoji = tmp_path / "emoji-data.txt"
    brackets.write_bytes(b"valid brackets")
    emoji.write_bytes(b"valid emoji")
    monkeypatch.setattr(
        generator,
        "SOURCE_HASHES",
        {
            brackets.name: hashlib.sha256(brackets.read_bytes()).hexdigest(),
            emoji.name: hashlib.sha256(emoji.read_bytes()).hexdigest(),
        },
    )
    (brackets if corrupted_name == brackets.name else emoji).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match=rf"SHA-256 mismatch for {corrupted_name}"):
        generator.verify_sources(brackets, emoji)
