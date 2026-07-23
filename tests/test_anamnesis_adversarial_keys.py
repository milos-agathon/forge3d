from __future__ import annotations

import json

import forge3d
import pytest
from forge3d.anamnesis import (
    _Store,
    capability_fingerprint,
    engine_fingerprint,
    leaf_key,
    pass_key,
    render_sequence,
    verify,
)


def _key(pipeline: bytes, uniforms: bytes = b"uniform\0\0\0\0", caps: bytes | None = None) -> str:
    return pass_key(
        "terrain.forward",
        pipeline,
        uniforms,
        [],
        caps or capability_fingerprint({}, backend="vulkan"),
        engine_fingerprint(),
    )


def test_adversarial_key_pairs_are_distinct():
    padding_a = b"value\0\0\0"
    padding_b = b"value\0\0\1"
    assert _key(b"pipeline", padding_a) != _key(b"pipeline", padding_b)
    assert _key(b"sampler=nearest") != _key(b"sampler=linear")
    assert _key(b"blend=replace") != _key(b"blend=alpha")
    caps_all = capability_fingerprint({}, backend="vulkan", naga_capabilities=["all"])
    caps_restricted = capability_fingerprint({}, backend="vulkan", naga_capabilities=["restricted"])
    assert _key(b"wgsl=same", caps=caps_all) != _key(b"wgsl=same", caps=caps_restricted)


def test_corrupt_blob_is_quarantined_then_recomputed(tmp_path):
    recipe = {"terrain": {"dem": [0, 1]}, "output": {"samples": 1}}
    render_sequence(recipe, frames=[0], cache=tmp_path)
    output_entry = None
    for meta_path in tmp_path.glob("??/*/meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("pass_label") == "frame.output":
            output_entry = meta_path.parent
            break
    assert output_entry is not None
    blob_path = output_entry / "blob"
    blob = bytearray(blob_path.read_bytes())
    blob[0] ^= 0x01
    blob_path.write_bytes(blob)

    result = verify(tmp_path)
    assert result["quarantined"] == 1
    assert any((tmp_path / "quarantine").iterdir())
    rerender = render_sequence(recipe, frames=[0], cache=tmp_path)
    assert (0, "frame.output") in rerender.observed_recompute


def test_corrupt_metadata_cannot_bypass_blob_integrity(tmp_path):
    recipe = {"terrain": {"dem": [2, 3]}, "output": {"samples": 1}}
    render_sequence(recipe, frames=[0], cache=tmp_path, verify_reads=False)
    output_entry = None
    for meta_path in tmp_path.glob("??/*/meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("pass_label") == "frame.output":
            output_entry = meta_path.parent
            break
    assert output_entry is not None
    meta_path = output_entry / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["derivation"]["material"]["uniform_hex"] = "00"
    meta_path.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match="prediction mismatch"):
        render_sequence(recipe, frames=[0], cache=tmp_path, verify_reads=False)
    rerender = render_sequence(recipe, frames=[0], cache=tmp_path, verify_reads=False)
    assert (0, "frame.output") in rerender.observed_recompute
    assert any((tmp_path / "quarantine").iterdir())


def _complete_footprint(path) -> int:
    return sum(item.stat().st_size for item in [path, *path.rglob("*")])


def test_complete_store_footprint_is_hard_bounded(tmp_path):
    max_bytes = 64 * 1024
    render_sequence(
        {"terrain": {"dem": list(range(64))}},
        frames=range(8),
        cache=tmp_path,
        max_bytes=max_bytes,
        verify_reads=False,
    )
    assert _complete_footprint(tmp_path) <= max_bytes


def test_tiny_budget_rejects_unrepresentable_self_describing_entry(tmp_path):
    try:
        render_sequence(
            {"terrain": {"dem": [0, 1]}},
            frames=[0],
            cache=tmp_path,
            max_bytes=1024,
        )
    except ValueError as error:
        assert "max_bytes" in str(error) or "footprint" in str(error)
    else:
        assert _complete_footprint(tmp_path) <= 1024


def test_semantic_input_role_swap_changes_key():
    height = "11" * 32
    water = "22" * 32
    common = (
        b"pipeline",
        b"uniform",
        capability_fingerprint({}, backend="vulkan"),
        engine_fingerprint(),
    )
    left = pass_key(
        "terrain.forward",
        common[0],
        common[1],
        [("heightmap@0", height), ("water_mask@1", water)],
        common[2],
        common[3],
    )
    right = pass_key(
        "terrain.forward",
        common[0],
        common[1],
        [("heightmap@0", water), ("water_mask@1", height)],
        common[2],
        common[3],
    )
    assert left != right


def test_rust_and_python_share_one_store_schema_bidirectionally(tmp_path):
    max_bytes = 1024 * 1024
    native_blob = b"written-by-rust"
    native_key = forge3d.anamnesis_store_put_leaf(
        tmp_path, native_blob, "interop.native", max_bytes
    )
    python_store = _Store(tmp_path, max_bytes, True)
    assert python_store.get(native_key)[0] == native_blob

    python_blob = b"written-by-python"
    python_key = leaf_key(python_blob)
    python_store.put_leaf(python_key, python_blob, label="interop.python")
    assert (
        forge3d.anamnesis_store_get(tmp_path, python_key, max_bytes) == python_blob
    )
    report = forge3d.anamnesis_store_verify(tmp_path, max_bytes)
    assert report == {
        "valid": 2,
        "quarantined": 0,
        "bytes_checked": len(native_blob) + len(python_blob),
    }
