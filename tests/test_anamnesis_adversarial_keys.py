from __future__ import annotations
from anamnesis_packed_store import metadata_entries, metadata, write_metadata, read_blob, write_blob

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


@pytest.mark.parametrize("container", [bytes, bytearray, list, tuple])
def test_native_key_preserves_python_parity_and_byte_sequence_inputs(container, monkeypatch):
    import forge3d.anamnesis as anamnesis
    from forge3d._native import get_native_module

    native = get_native_module()
    assert native is not None
    inputs = [("z@1", leaf_key(b"z")), ("a@0", leaf_key(b"a"))]
    pipeline, uniforms, caps, engine = b"pipeline\0\xff", b"uniform\0padding", b"caps", b"engine"
    monkeypatch.setattr(anamnesis, "get_native_module", lambda: None)
    expected = pass_key("terrain.render", pipeline, uniforms, inputs, caps, engine)
    assert native.anamnesis_pass_key(
        "terrain.render", container(pipeline), container(uniforms), inputs,
        container(caps), container(engine),
    ) == expected


def test_sequence_descriptor_templates_preserve_exact_canonical_bytes(tmp_path):
    from forge3d.anamnesis import _pass_descriptor

    state = {"backend": "unit-cpu", "seed": 7, "sampler": {"accumulation_frame_index": None}}
    render_sequence(
        {"anamnesis_state": state, "layers": [{"kind": "label", "text": "x"}]},
        frames=[-3, 0, 19, 100000], cache=tmp_path,
        capabilities={"backend": "unit-cpu", "granted": [], "limits": {}},
     reference_only=True)
    entries = metadata_entries(tmp_path)
    assert entries
    for entry_metadata in entries.values():
        metadata = entry_metadata
        material = metadata["derivation"]["material"]
        frame = metadata["frame"]
        expected = _pass_descriptor(material["label"], None if frame < 0 else frame, state)
        assert bytes.fromhex(material["pipeline_descriptor_hex"]) == expected


def test_corrupt_blob_is_quarantined_then_recomputed(tmp_path):
    recipe = {"terrain": {"dem": [0, 1]}, "output": {"samples": 1}}
    render_sequence(recipe, frames=[0], cache=tmp_path, reference_only=True)
    output_key = next(key for key, meta in metadata_entries(tmp_path).items() if meta["pass_label"] == "frame.output")
    blob = bytearray(read_blob(tmp_path, output_key))
    blob[0] ^= 0x01
    write_blob(tmp_path, output_key, blob)

    result = verify(tmp_path)
    assert result["quarantined"] == 1
    assert any((tmp_path / "quarantine").iterdir())
    rerender = render_sequence(recipe, frames=[0], cache=tmp_path, reference_only=True)
    assert (0, "frame.output") in rerender.observed_recompute


def test_corrupt_metadata_cannot_bypass_blob_integrity(tmp_path):
    recipe = {"terrain": {"dem": [2, 3]}, "output": {"samples": 1}}
    render_sequence(recipe, frames=[0], cache=tmp_path, verify_reads=False, reference_only=True)
    output_key = next(key for key, meta in metadata_entries(tmp_path).items() if meta["pass_label"] == "frame.output")
    meta = metadata(tmp_path, output_key)
    meta["derivation"]["material"]["uniform_hex"] = "00"
    write_metadata(tmp_path, output_key, meta)

    with pytest.raises(RuntimeError, match="prediction mismatch"):
        render_sequence(recipe, frames=[0], cache=tmp_path, verify_reads=False, reference_only=True)
    rerender = render_sequence(recipe, frames=[0], cache=tmp_path, verify_reads=False, reference_only=True)
    assert (0, "frame.output") in rerender.observed_recompute
    assert any((tmp_path / "quarantine").iterdir())


def _complete_footprint(path) -> int:
    return sum(item.stat().st_size for item in [path, *path.rglob("*")])


def test_complete_store_footprint_is_hard_bounded(tmp_path):
    # Linux filesystems commonly allocate 4 KiB for every hash-prefix and
    # entry directory; the test budget must fit a self-describing sequence
    # before it can meaningfully assert the hard upper bound.
    max_bytes = 512 * 1024
    recipe = {"terrain": {"dem": list(range(64))}}
    render_sequence(
        recipe,
        frames=range(8),
        cache=tmp_path,
        max_bytes=max_bytes,
        verify_reads=False,
     reference_only=True)
    # A warm read creates the batched cross-language LRU journal; it is a
    # control file and must remain inside the same complete-footprint bound.
    render_sequence(
        recipe,
        frames=range(8),
        cache=tmp_path,
        max_bytes=max_bytes,
        verify_reads=False,
     reference_only=True)
    assert _complete_footprint(tmp_path) <= max_bytes


def test_tiny_budget_rejects_unrepresentable_self_describing_entry(tmp_path):
    try:
        render_sequence(
            {"terrain": {"dem": [0, 1]}},
            frames=[0],
            cache=tmp_path,
            max_bytes=1024,
         reference_only=True)
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
