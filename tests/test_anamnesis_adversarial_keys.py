from __future__ import annotations

import json

from forge3d.anamnesis import (
    capability_fingerprint,
    engine_fingerprint,
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


def test_fast_pack_cannot_bypass_blob_integrity(tmp_path):
    recipe = {"terrain": {"dem": [2, 3]}, "output": {"samples": 1}}
    render_sequence(recipe, frames=[0], cache=tmp_path, verify_reads=False)
    output_entry = None
    for meta_path in tmp_path.glob("??/*/meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("pass_label") == "frame.output":
            output_entry = meta_path.parent
            break
    assert output_entry is not None
    output_key = output_entry.name

    blob_path = output_entry / "blob"
    blob = bytearray(blob_path.read_bytes())
    blob[0] ^= 0x01
    blob_path.write_bytes(blob)
    fast_path = tmp_path / "fastpack.json"
    fast = json.loads(fast_path.read_text(encoding="utf-8"))
    fast_blob = bytearray.fromhex(fast["entries"][output_key]["blob_hex"])
    fast_blob[0] ^= 0x02
    fast["entries"][output_key]["blob_hex"] = fast_blob.hex()
    fast_path.write_text(json.dumps(fast, sort_keys=True), encoding="utf-8")

    rerender = render_sequence(recipe, frames=[0], cache=tmp_path, verify_reads=False)
    assert (0, "frame.output") in rerender.observed_recompute
    assert any((tmp_path / "quarantine").iterdir())
