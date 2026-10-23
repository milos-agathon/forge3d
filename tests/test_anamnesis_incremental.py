from __future__ import annotations

import copy
import json
import os

import pytest

from forge3d.anamnesis import _Store, explain, render_sequence
from anamnesis_packed_store import (resident as packed_resident, metadata, metadata_entries, write_metadata, write_blob, remove_entry)


def _recipe(text: str) -> dict:
    return {
        "terrain": {"dem_sha256": "11" * 32, "z_scale": 1.5},
        "atmosphere": {"model": "clear"},
        "lighting": {"azimuth": 215.0, "elevation": 35.0},
        "camera": {"path": "deterministic-flythrough-v1"},
        "layers": [
            {
                "kind": "label",
                "id": "summit",
                "text": text,
                "visible_frames": list(range(250, 260)),
            }
        ],
        "output": {"width": 64, "height": 64, "samples": 4},
    }


def _watch_complete_bound(monkeypatch):
    """Independent footprint at every eviction entry."""
    admitting, admissions = set(), []
    initialize, collect = _Store.__init__, _Store.gc

    def footprint(root):
        return sum(path.stat().st_size for path in (root, *root.rglob("*")))

    def initialize_checked(self, *args, **kwargs):
        admitting.add(id(self))
        try:
            initialize(self, *args, **kwargs)
        finally:
            admitting.discard(id(self))

    def collect_checked(self, target):
        actual = footprint(self.root)
        if actual > self.max_bytes:
            # A smaller bound on an already populated store needs admission
            # cleanup. Milos approved reporting this separately on 2026-10-22.
            assert id(self) in admitting, (actual, self.max_bytes)
            admissions.append(actual - self.max_bytes)
        result = collect(self, target)
        assert footprint(self.root) <= self.max_bytes
        return result

    monkeypatch.setattr(_Store, "__init__", initialize_checked)
    monkeypatch.setattr(_Store, "gc", collect_checked)
    return admissions


@pytest.mark.slow
def test_600_frame_incremental_matches_cold_and_recomputes_exact_set(tmp_path):
    warm_store = tmp_path / "warm"
    cold_store = tmp_path / "cold-modified"
    original = _recipe("Old summit")
    modified = copy.deepcopy(original)
    modified["layers"][0]["text"] = "New summit"

    render_sequence(original, cache=warm_store, reference_only=True)
    incremental = render_sequence(modified, cache=warm_store, reference_only=True)
    cold_modified = render_sequence(modified, cache=cold_store, reference_only=True)

    assert incremental.frame_hashes == cold_modified.frame_hashes
    assert len(incremental.frame_hashes) == 600
    assert incremental.prediction_matches
    assert set(incremental.predicted_recompute) == set(incremental.observed_recompute)
    assert len(incremental.observed_recompute) == 21
    labels = {label for _, label in incremental.observed_recompute}
    assert labels == {"label.compile", "label.composite", "frame.output"}
    assert incremental.cache_report.hit_rate > 0.99


def test_eviction_invalidates_sequence_manifest_and_label_edit_recovers(tmp_path):
    original, modified = _recipe("Old summit"), _recipe("New summit")
    frames = range(40)
    for recipe in (original, modified):
        recipe["layers"][0]["visible_frames"] = list(range(10, 20))
    unbounded = tmp_path / "unbounded"
    render_sequence(original, frames=frames, cache=unbounded, reference_only=True)
    footprint = _Store._tree_bytes(unbounded)
    bounded = tmp_path / "bounded"
    budget = footprint // 2
    render_sequence(original, frames=frames, cache=bounded, max_bytes=budget, reference_only=True)
    assert _Store._tree_bytes(bounded) <= budget
    resident = json.loads((bounded / "last_manifest.json").read_text())["entries"]
    assert resident
    assert all(packed_resident(bounded, key) for key in resident.values())
    result = render_sequence(modified, frames=frames, cache=bounded, max_bytes=budget, reference_only=True)
    cold = render_sequence(modified, frames=frames, cache=None, reference_only=True)
    assert result.frame_blobs == cold.frame_blobs
    assert result.prediction_matches
    assert _Store._tree_bytes(bounded) <= budget


def test_dry_run_predicts_without_writing(tmp_path):
    store = tmp_path / "dry"
    result = render_sequence(_recipe("Summit"), frames=range(3), cache=store, dry_run=True, reference_only=True)
    assert result.predicted_recompute
    assert result.observed_recompute == []
    assert not any(path.name == "blob" for path in store.rglob("blob"))


def test_explain_reconstructs_recursive_pass_derivation(tmp_path, capsys):
    result = render_sequence(_recipe("Summit"), frames=[250], cache=tmp_path, reference_only=True)
    key = result.pass_keys["250:frame.output"]

    tree = explain(key, tmp_path)
    capsys.readouterr()

    assert tree["key"] == key
    assert tree["reconstructed_key"] == key
    assert tree["reconstructs"] is True
    assert tree["pipeline_descriptor_hex"]
    assert tree["uniform_hex"]
    assert tree["capability_fingerprint_hex"]
    assert tree["engine_fingerprint_hex"]
    assert tree["inputs"]
    assert all(item["binding"] and item["derivation"] for item in tree["inputs"])


def test_structured_scheduler_executes_only_changed_passes(tmp_path):
    original = _recipe("Old summit")
    original["layers"][0]["visible_frames"] = [250]
    modified = copy.deepcopy(original)
    modified["layers"][0]["text"] = "New summit"
    calls: list[tuple[str, int]] = []

    def identity(state, frame, inputs):
        calls.append(("identity", frame))
        return inputs[0] if inputs else repr(state).encode("utf-8")

    executors = {
        "terrain.shade": identity,
        "accumulation": identity,
        "label.compile": identity,
        "label.composite": identity,
        "frame.output": identity,
    }
    fingerprints = {
        label: f"test-executor:{label}".encode("utf-8") for label in executors
    }
    options = {
        "capabilities": {"backend": "test-cpu", "granted": [], "limits": {}},
        "frames": [250],
        "verify_reads": False,
        "pass_executors": executors,
        "pass_executor_fingerprints": fingerprints,
        "pass_executor_contexts": {
            label: f"test-context:{label}".encode("utf-8")
            for label in executors
        },
    }

    render_sequence(original, cache=tmp_path / "warm-speed", **options, reference_only=True)
    calls.clear()
    incremental = render_sequence(modified, cache=tmp_path / "warm-speed", **options, reference_only=True)
    incremental_calls = list(calls)
    calls.clear()
    cold = render_sequence(modified, cache=tmp_path / "cold-speed", **options, reference_only=True)

    assert incremental.frame_hashes == cold.frame_hashes
    assert incremental.prediction_matches
    assert incremental_calls == [
        ("identity", -1),
        ("identity", 250),
        ("identity", 250),
    ]


def test_incremental_elides_unchanged_ancestors_and_batches_lru_touches(
    tmp_path, monkeypatch
):
    original = _recipe("Old summit")
    original["layers"][0]["visible_frames"] = [5]
    modified = copy.deepcopy(original)
    modified["layers"][0]["text"] = "New summit"
    frames = range(12)

    render_sequence(original, frames=frames, cache=tmp_path, verify_reads=False, reference_only=True)
    metadata_before = metadata_entries(tmp_path)
    read_keys: list[str] = []
    original_get = _Store.get

    def recording_get(self, key, **kwargs):
        read_keys.append(key)
        return original_get(self, key, **kwargs)

    monkeypatch.setattr(_Store, "get", recording_get)
    incremental = render_sequence(
        modified,
        frames=frames,
        cache=tmp_path,
        verify_reads=False,
     reference_only=True)

    # Eleven unchanged terminal frames plus the unchanged accumulation input
    # to the one changed label composite. Shadow/shade ancestors are committed
    # transitively by those keys and require no filesystem read.
    assert len(read_keys) == 12
    assert incremental.prediction_matches
    assert (tmp_path / "access.log").is_file()
    after = metadata_entries(tmp_path)
    assert all(after[key] == meta for key, meta in metadata_before.items())


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="set FORGE3D_RUN_GPU_ANAMNESIS=1 on a hardware-backed runner",
)
def test_real_gpu_600_frame_acceptance(tmp_path):
    from anamnesis_gpu_acceptance import run_acceptance

    result = run_acceptance(tmp_path)
    assert result["matching_frames"] == 600
    assert result["symmetric_difference"] == []
    assert result["speedup"] >= 20.0


@pytest.mark.parametrize("visible", [range(10, 20), range(40), range(30, 40)])
@pytest.mark.parametrize("new_text", ["New summit", "N" * 20_000, "N" * 200_000],
                         ids=["short", "20k", "200k"])
def test_full_store_preserves_unread_frames_during_larger_label_edit(tmp_path, visible, new_text, monkeypatch):
    """CPU reference/store regression; this is not physical rendering proof."""
    _watch_complete_bound(monkeypatch)
    original, modified = _recipe("Old summit"), _recipe(new_text)
    for recipe in (original, modified):
        recipe["layers"][0]["visible_frames"] = list(visible)
    frames = range(40)
    cache = tmp_path / "bounded"
    render_sequence(original, frames=frames, cache=cache, reference_only=True)
    # Measure this exact store: timestamp/timing metadata makes separately
    # seeded stores differ in size even for the same recipe.
    budget = _Store._tree_bytes(cache)
    assert (cache / "last_manifest.json").exists()
    assert _Store._tree_bytes(cache) == budget
    result = render_sequence(modified, frames=frames, cache=cache, max_bytes=budget, reference_only=True)
    fresh = render_sequence(modified, frames=frames, cache=None, reference_only=True)
    assert result.frame_blobs == fresh.frame_blobs
    assert result.prediction_matches
    assert _Store._tree_bytes(cache) <= budget
    # A partial store still publishes an honest reusable plan after eviction.
    resident = json.loads((cache / "last_manifest.json").read_text())["entries"]
    assert resident
    assert all(packed_resident(cache, key) for key in resident.values())
    unchanged = render_sequence(modified, frames=frames, cache=cache, max_bytes=budget, reference_only=True)
    assert unchanged.cache_report.hits
    assert unchanged.frame_blobs == fresh.frame_blobs
    assert unchanged.prediction_matches


@pytest.mark.parametrize("case", ["smaller_budget", "consecutive_edits"])
def test_store_bound_during_smaller_budget_and_consecutive_edits(tmp_path, monkeypatch, case):
    admissions = _watch_complete_bound(monkeypatch)
    visible = range(40) if case == "smaller_budget" else range(10, 20)
    original, modified = _recipe("Old summit"), _recipe("N" * 200_000)
    for recipe in (original, modified):
        recipe["layers"][0]["visible_frames"] = list(visible)
    cache = tmp_path / "bounded"
    render_sequence(original, frames=range(40), cache=cache, reference_only=True)
    budget = _Store._tree_bytes(cache) - (4096 if case == "smaller_budget" else 0)
    result = render_sequence(modified, frames=range(40), cache=cache, max_bytes=budget, reference_only=True)
    assert result.frame_blobs == render_sequence(modified, frames=range(40), cache=None, reference_only=True).frame_blobs
    assert result.prediction_matches
    if case == "consecutive_edits":
        modified["layers"][0]["text"] = "Short again"
        result = render_sequence(modified, frames=range(40), cache=cache, max_bytes=budget, reference_only=True)
        assert result.frame_blobs == render_sequence(modified, frames=range(40), cache=None, reference_only=True).frame_blobs
        assert result.prediction_matches
        assert not admissions
    else:
        assert admissions == [4096]
    assert _Store._tree_bytes(cache) <= budget


def test_gc_preserved_reads_are_scoped_and_verified(tmp_path):
    from forge3d.anamnesis import leaf_key
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    blob = b"unread pixels"
    key = leaf_key(blob)
    store.put_leaf(key, blob, label="test.unread")
    with pytest.raises(RuntimeError, match="executor failure"):
        with store.protect_reads([key]):
            with store.protect_reads([]):
                store.gc(0)
            assert not packed_resident(store.root, key)
            assert store.get(key)[0] == blob
            raise RuntimeError("executor failure")
    assert store._protected_reads == {}
    assert store.get(key) is None

    store.put_leaf(key, blob, label="test.unread")
    store._memory_entries.clear()
    write_blob(store.root, key, b"corrupt")
    with pytest.raises(RuntimeError, match="prediction mismatch"):
        with store.protect_reads([key]):
            store.gc(0)
    assert store._protected_reads == {}
    assert not packed_resident(store.root, key)


def test_batched_native_signatures_match_python_after_file_changes(tmp_path):
    from forge3d._native import get_native_module
    native = get_native_module()
    paths = [tmp_path / "present", tmp_path / "missing"]
    paths[0].mkdir()
    (paths[0] / "blob").write_bytes(b"original")
    (paths[0] / "meta.json").write_bytes(b"metadata")
    inspect = native._anamnesis_entry_signatures
    assert inspect(paths) == [_Store._entry_signature(path) for path in paths]
    (paths[0] / "blob").write_bytes(b"new content")
    assert inspect(paths) == [_Store._entry_signature(path) for path in paths]
    (paths[0] / "meta.json").unlink()
    assert inspect(paths) == [None, None]


def test_store_reserves_payload_and_journal_space_before_writing(tmp_path, monkeypatch):
    from forge3d.anamnesis import leaf_key, _DEFAULT_MAX_BYTES
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    original = b"a" * 403_440  # The transient excess measured in the review.
    store.put_leaf(leaf_key(original), original, label="peak")
    store.max_bytes = store.disk_bytes()
    # Observe the actual packed payload/index/journal opens and writes, rather
    # than the former directory-store metadata/staging helpers.
    events = _observe_store_writes(monkeypatch, tmp_path, lambda: store.max_bytes)
    changed = b"b" * len(original)
    key = leaf_key(changed)
    store.put_leaf(key, changed, label="peak")
    assert store.get(key)[0] == changed
    store.flush_accesses()
    assert events
    assert store.disk_bytes() <= store.max_bytes


@pytest.mark.parametrize("damage", ["none", "blob", "metadata", "changed_after_prefetch", "missing_after_prefetch"])
def test_persisted_prefetch_preserves_integrity_and_quarantine(tmp_path, damage):
    from forge3d.anamnesis import leaf_key, _DEFAULT_MAX_BYTES

    blob = b"persisted bytes"
    key = leaf_key(blob)
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, False)
    store.put_leaf(key, blob, label="test.persisted")
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, False)
    if damage == "blob":
        write_blob(store.root, key, b"corrupt bytes!!")
    elif damage == "metadata":
        meta = metadata(store.root, key)
        meta["key"] = "00" * 32
        write_metadata(store.root, key, meta)
    with store.prefetch_reads([key, key]):
        store._pending_reads[key].result()
        if damage == "changed_after_prefetch":
            write_blob(store.root, key, b"changed after validation")
        elif damage == "missing_after_prefetch":
            write_metadata(store.root, key, {})
        # Worker verification cannot quarantine, touch or publish memory hits.
        assert not store._memory_entries and not store._touched_keys
        assert not list((tmp_path / "quarantine").iterdir())
        restored = store.get(key)
    if damage == "none":
        assert restored[0] == blob
        assert store._touched_keys == {key}
    else:
        assert restored is None
        assert not packed_resident(store.root, key)
        assert len(list((tmp_path / "quarantine").iterdir())) == 1
    assert not store._pending_reads


def test_complete_tree_scan_counts_files_directories_and_controls(tmp_path):
    directory = tmp_path / "prefix" / "entry"
    directory.mkdir(parents=True)
    for name, payload in (("blob", b"data"), ("meta.json", b"{}")):
        (directory / name).write_bytes(payload)
    (tmp_path / "access.log").write_bytes(b"journal")
    (tmp_path / "quarantine").mkdir()
    expected = sum(path.stat().st_size for path in (tmp_path, *tmp_path.rglob("*")))
    assert _Store._tree_bytes(tmp_path) == expected


def _observe_store_writes(monkeypatch, root, limit):
    """Observe real filesystem growth at open, write, flush, mkdir and rename."""
    from pathlib import Path
    events = []
    original_open, original_mkdir, original_replace = Path.open, os.mkdir, Path.replace

    def within(path):
        return path == root or root in path.parents

    def footprint():
        return sum(path.stat().st_size for path in (root, *root.rglob("*"))) if root.exists() else 0

    def check(operation, growth=0):
        actual = footprint()
        events.append((operation, actual, growth))
        assert actual + growth <= limit(), (operation, actual, growth, limit())

    class ObservedFile:
        def __init__(self, stream, path, mode):
            self.stream, self.path, self.mode = stream, path, mode
        def __getattr__(self, name):
            return getattr(self.stream, name)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.flush()
            check("before close")
            result = self.stream.__exit__(*args)
            check("after close")
            return result
        def write(self, payload):
            size = len(payload.encode(self.stream.encoding)) if isinstance(payload, str) else len(memoryview(payload))
            old_size = self.path.stat().st_size
            growth = size if "a" in self.mode else max(0, self.stream.tell() + size - old_size)
            check("before " + self.path.name, growth)
            result = self.stream.write(payload)
            self.stream.flush()
            check("after " + self.path.name)
            return result

    def open_checked(path, mode="r", *args, **kwargs):
        writing = within(path) and any(flag in mode for flag in "wax+")
        if writing:
            check("before open " + path.name)
        stream = original_open(path, mode, *args, **kwargs)
        if writing:
            check("after open " + path.name)
            return ObservedFile(stream, path, mode)
        return stream

    def mkdir_checked(path, *args, **kwargs):
        path = Path(path)
        if within(path):
            check("before mkdir")
        result = original_mkdir(path, *args, **kwargs)
        if within(path):
            check("after mkdir")
        return result

    def replace_checked(path, target):
        if within(path):
            check("before rename")
        result = original_replace(path, target)
        if within(path):
            check("after rename")
        return result

    monkeypatch.setattr(Path, "open", open_checked)
    # tempfile.mkdtemp creates staging through os.mkdir directly.
    monkeypatch.setattr(os, "mkdir", mkdir_checked)
    monkeypatch.setattr(Path, "replace", replace_checked)
    return events


@pytest.mark.parametrize("operation", ["payload", "manifest", "access_log", "control_staging", "quarantine"])
def test_every_store_write_reserves_real_tree_growth(tmp_path, monkeypatch, operation):
    from forge3d.anamnesis import leaf_key, _DEFAULT_MAX_BYTES
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    keys = []
    for byte in (1, 2):
        blob = bytes([byte]) * 4096
        keys.append(leaf_key(blob))
        store.put_leaf(keys[-1], blob, label="write-bound")
    if operation == "quarantine":
        write_blob(store.root, keys[0], bytes([3]) * 4096)
    if operation == "access_log":
        for key in keys:
            assert store.get(key) is not None
    store.max_bytes = store.disk_bytes()
    events = _observe_store_writes(monkeypatch, tmp_path, lambda: store.max_bytes)
    if operation == "payload":
        blob = bytes([4]) * 4096
        store.put_leaf(leaf_key(blob), blob, label="write-bound")
    elif operation == "manifest":
        store.write_control("last_manifest.json", json.dumps({"entries": {key: key for key in keys}}).encode())
    elif operation == "access_log":
        store.flush_accesses()
    elif operation == "control_staging":
        store.write_control("state.json", b'{"state":"new"}')
    else:
        assert store.get(keys[0]) is None
        assert len(list((tmp_path / "quarantine").iterdir())) == 1
    assert events
    assert store.disk_bytes() <= store.max_bytes
    assert store.admission_cleanup_bytes == 0
    print(json.dumps(dict(operation=operation, observations=len(events), peak=max(row[1] for row in events), max_bytes=store.max_bytes)))


@pytest.mark.parametrize("reopen", [False, True], ids=["first-use", "reopen"])
def test_admission_cleans_lowered_bound_before_any_write(tmp_path, monkeypatch, reopen):
    from forge3d.anamnesis import leaf_key, _DEFAULT_MAX_BYTES
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    for byte in (1, 2):
        blob = bytes([byte]) * 4096
        store.put_leaf(leaf_key(blob), blob, label="admission")
    filled = store.disk_bytes()
    store.max_bytes = filled - 4096
    events = _observe_store_writes(monkeypatch, tmp_path, lambda: store.max_bytes)
    if reopen:
        store = _Store(tmp_path, store.max_bytes, True)
    blob = bytes([3]) * 4096
    store.put_leaf(leaf_key(blob), blob, label="admission")
    assert store.admission_cleanup_bytes == 4096
    assert events and store.disk_bytes() <= store.max_bytes


def test_unreclaimable_admission_raises_without_any_write(tmp_path, monkeypatch):
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    control = tmp_path / "unmanaged-control"
    control.write_bytes(bytes(4096 + 1))
    bound = store.disk_bytes() - 4096
    events = _observe_store_writes(monkeypatch, tmp_path, lambda: bound)
    with pytest.raises(ValueError, match="footprint exceeds max_bytes"):
        _Store(tmp_path, bound, True)
    assert not events
    assert control.stat().st_size == 4096 + 1


def test_bounded_manifest_remains_resident_across_three_processes(tmp_path):
    """A reopened all-hit store must not republish elided, evicted ancestors."""
    import subprocess
    import sys
    from pathlib import Path
    import forge3d.anamnesis as anamnesis

    recipe = _recipe("Old summit")
    recipe["layers"][0]["visible_frames"] = list(range(10, 20))
    (tmp_path / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    script = r"""
import json, sys
from pathlib import Path
from forge3d.anamnesis import _Store, render_sequence
from anamnesis_packed_store import resident as packed_resident
phase, root = sys.argv[1], Path(sys.argv[2])
cache = root / "bounded"
recipe = json.loads((root / "recipe.json").read_text(encoding="utf-8"))
if phase == "seed":
    render_sequence(recipe, frames=range(40), cache=cache, reference_only=True)
    (root / "budget.txt").write_text(str(_Store._tree_bytes(cache)))
budget = int((root / "budget.txt").read_text())
recipe["layers"][0]["text"] = "Edited summit" if phase == "edit" else "N" * 200_000
fresh = render_sequence(recipe, frames=range(40), cache=None, reference_only=True)
for _ in range(2 if phase == "seed" else 1):
    result = render_sequence(recipe, frames=range(40), cache=cache, max_bytes=budget, reference_only=True)
    assert result.frame_blobs == fresh.frame_blobs
    assert result.prediction_matches
    assert set(result.predicted_recompute) == set(result.observed_recompute)
    assert _Store._tree_bytes(cache) <= budget
entries = json.loads((cache / "last_manifest.json").read_text())["entries"]
missing = [instance for instance, key in entries.items()
           if not packed_resident(cache, key)]
print(json.dumps({"phase": phase, "missing": missing}))
(root / (phase + ".json")).write_text(json.dumps(missing))
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(Path(anamnesis.__file__).parents[1]), str(Path(__file__).parent))),
               FORGE3D_NO_BOOTSTRAP="1")
    for phase in ("seed", "reopen", "edit"):
        result = subprocess.run([sys.executable, "-c", script, phase, str(tmp_path)],
                                env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    for phase in ("seed", "reopen", "edit"):
        assert json.loads((tmp_path / (phase + ".json")).read_text()) == []


def test_resident_manifest_uses_inventory_and_excludes_protected_evictions(tmp_path, monkeypatch):
    from pathlib import Path
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key

    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    persisted, added, absent = (leaf_key(blob) for blob in (b"persisted", b"added", b"absent"))
    store.put_leaf(persisted, b"persisted", label="test.resident")
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    store.put_leaf(added, b"added", label="test.resident")
    payload = json.dumps({"entries": {key: key for key in (persisted, added, absent)}}).encode()

    def resident_without_filesystem():
        def forbidden(*args, **kwargs):
            raise AssertionError("manifest filtering must not query the filesystem")
        with monkeypatch.context() as patch:
            for method in ("stat", "is_dir", "exists", "open", "iterdir"):
                patch.setattr(Path, method, forbidden)
            patch.setattr(os, "scandir", forbidden)
            return json.loads(store._resident_manifest(payload))["entries"]

    assert set(resident_without_filesystem().values()) == {persisted, added}
    with store.protect_reads([persisted]):
        store.gc(0)
        assert store.get(persisted)[0] == b"persisted"
        assert not packed_resident(store.root, persisted)
        assert resident_without_filesystem() == {}


@pytest.mark.parametrize("damage", ["removed", "prefetch_removed", "corrupt"])
def test_resident_manifest_drops_missing_and_quarantined_entries(tmp_path, damage):
    import shutil
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key

    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    key = leaf_key(b"entry")
    store.put_leaf(key, b"entry", label="test.resident")
    if damage in {"removed", "prefetch_removed"}:
        remove_entry(store.root, key)
    else:
        write_blob(store.root, key, b"corrupt")
    if damage == "prefetch_removed":
        with store.prefetch_reads([key]):
            assert store.get(key) is None
    else:
        assert store.get(key) is None
    payload = json.dumps({"entries": {"entry": key}}).encode()
    assert json.loads(store._resident_manifest(payload))["entries"] == {}


@pytest.mark.parametrize("maintenance", ["sequence", "public_gc"])
def test_pooled_manifest_stays_resident_across_external_eviction_and_edit(tmp_path, maintenance):
    """P seeds, Q evicts, the same P store restores, then independent R edits."""
    import subprocess
    import sys
    from pathlib import Path
    import forge3d.anamnesis as an

    original = _recipe("Old summit")
    original["layers"][0]["visible_frames"] = list(range(10, 20))
    render_sequence(original, frames=range(40), cache=tmp_path / "measure", reference_only=True)
    budget = _Store._tree_bytes(tmp_path / "measure")
    cache = tmp_path / "bounded"
    render_sequence(original, frames=range(40), cache=cache, max_bytes=budget, reference_only=True)
    pool_key = an._store_pool_key(cache, budget, True)
    remembered = an._STORE_POOL[pool_key]
    root_identity = remembered._identity()
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(original), encoding="utf-8")
    script = r"""
import json, sys
from pathlib import Path
import forge3d.anamnesis as an
from anamnesis_packed_store import resident as packed_resident
role, recipe_path, cache, budget, maintenance = sys.argv[1:]
cache, budget = Path(cache), int(budget)
recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
removed = None
if role == "evict" and maintenance == "public_gc":
    removed = an.gc(0, cache)
    assert removed > 0
recipe["layers"][0]["text"] = "N" * 200_000 if role == "evict" else "Edited summit"
fresh = an.render_sequence(recipe, frames=range(40), cache=None, reference_only=True)
for _ in range(2 if role == "evict" else 1):
    result = an.render_sequence(recipe, frames=range(40), cache=cache, max_bytes=budget, reference_only=True)
    assert result.frame_blobs == fresh.frame_blobs
    assert result.prediction_matches
    assert set(result.predicted_recompute) == set(result.observed_recompute)
    assert an._Store._tree_bytes(cache) <= budget
entries = json.loads((cache / "last_manifest.json").read_text(encoding="utf-8"))["entries"]
assert entries and all(packed_resident(cache, key) for key in entries.values())
print(json.dumps({"role": role, "pid": __import__("os").getpid(),
                  "removed": removed, "resident_entries": len(entries)}))
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(Path(an.__file__).parents[1]), str(Path(__file__).parent))), FORGE3D_NO_BOOTSTRAP="1")

    def child(role):
        result = subprocess.run(
            [sys.executable, "-c", script, role, str(recipe_path), str(cache), str(budget), maintenance],
            env=env, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)

    evictor = child("evict")
    assert remembered._identity() == root_identity
    modified = copy.deepcopy(original)
    modified["layers"][0]["text"] = "N" * 200_000
    rerun = render_sequence(modified, frames=range(40), cache=cache, max_bytes=budget, reference_only=True)
    assert an._STORE_POOL[pool_key] is remembered  # Prove refresh, not pool replacement.
    assert rerun.prediction_matches and not rerun.cache_report.misses
    fresh = render_sequence(modified, frames=range(40), cache=None, reference_only=True)
    assert rerun.frame_blobs == fresh.frame_blobs
    entries = json.loads((cache / "last_manifest.json").read_text(encoding="utf-8"))["entries"]
    assert entries
    missing = [key for key in entries.values() if not packed_resident(cache, key)]
    # Run R even if P published a bad manifest, retaining both failure symptoms.
    editor = child("edit")
    assert not missing, missing
    assert evictor["pid"] != editor["pid"]
    assert {evictor["pid"], editor["pid"]}.isdisjoint({os.getpid()})


@pytest.mark.parametrize("maintenance", ["gc", "verify"])
def test_pooled_residency_refreshes_after_public_store_maintenance(tmp_path, monkeypatch, maintenance):
    import forge3d.anamnesis as an

    budget = an._DEFAULT_MAX_BYTES
    store = an._pooled_store(tmp_path, budget, True)
    removed, retained = an.leaf_key(b"removed"), an.leaf_key(b"retained")
    store.put_leaf(removed, b"removed", label="test.inventory")
    store.put_leaf(retained, b"retained", label="test.inventory")
    root_identity = store._identity()
    native = an.get_native_module()
    native_call = getattr(native, "anamnesis_store_" + maintenance, None)
    calls = []
    if native_call is not None:
        def observe(*args, **kwargs):
            calls.append(True)
            return native_call(*args, **kwargs)
        monkeypatch.setattr(native, "anamnesis_store_" + maintenance, observe)
    if maintenance == "gc":
        assert an.gc(0, tmp_path) > 0
        expected = {}
    else:
        write_blob(store.root, removed, b"corrupt")
        assert an.verify(tmp_path)["quarantined"] == 1
        expected = {"retained": retained}
    if native_call is not None:
        assert calls  # Exercise the real native public-maintenance route.
    assert store._identity() == root_identity
    assert an._pooled_store(tmp_path, budget, True) is store
    payload = json.dumps({"entries": {"removed": removed, "retained": retained}}).encode()
    assert json.loads(store._resident_manifest(payload))["entries"] == expected


def test_residency_refresh_uses_only_the_two_packed_file_signatures(tmp_path, monkeypatch):
    from pathlib import Path
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key

    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    keys = {leaf_key(blob) for blob in (b"one", b"two")}
    for blob in (b"one", b"two"):
        store.put_leaf(leaf_key(blob), blob, label="test.inventory")
    stat, queried = Path.stat, []

    def observe(path, *args, **kwargs):
        queried.append(path)
        return stat(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("packed residency refresh must not enumerate per-entry files")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", observe)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(os, "scandir", forbidden)
        store._refresh_residency()
    assert store._resident_keys == keys
    assert queried == [tmp_path / _Store.DATA, tmp_path / _Store.INDEX]


def test_packed_schema_ignores_legacy_entries_in_both_stores(tmp_path):
    import forge3d
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key

    blob = b"old schema must miss"
    key = leaf_key(blob)
    seed = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    seed.put_leaf(key, blob, label="test.legacy")
    legacy_metadata = metadata(tmp_path, key)
    legacy_metadata["schema"] = "forge3d.anamnesis.store/1"
    for name in (_Store.DATA, _Store.INDEX):
        (tmp_path / name).unlink()
    legacy = tmp_path / key[:2] / key
    legacy.mkdir(parents=True)
    (legacy / "blob").write_bytes(blob)
    (legacy / "meta.json").write_text(json.dumps(legacy_metadata), encoding="utf-8")
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    assert store.get(key) is None
    assert forge3d.anamnesis_store_get(tmp_path, key, _DEFAULT_MAX_BYTES) is None
    store.put_leaf(key, blob, label="test.schema2")
    assert metadata(tmp_path, key)["schema"] == "forge3d.anamnesis.store/2"
    assert forge3d.anamnesis_store_get(tmp_path, key, _DEFAULT_MAX_BYTES) == blob
    # The legacy payload was ignored, not migrated or silently trusted.
    assert (legacy / "blob").read_bytes() == blob


@pytest.mark.parametrize("field", ["offset", "length"])
def test_packed_record_bounds_are_verified_before_serving(tmp_path, field):
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key
    from anamnesis_packed_store import index, INDEX

    key = leaf_key(b"verified")
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, False)
    store.put_leaf(key, b"verified", label="test.bounds")
    value = index(tmp_path)
    value["entries"][key][field] = -1 if field == "offset" else len(b"verified") + 1
    (tmp_path / INDEX).write_text(json.dumps(value), encoding="utf-8")
    store._refresh_residency()
    payload = json.dumps({"entries": {"invalid": key}}).encode()
    assert json.loads(store._resident_manifest(payload))["entries"] == {}
    assert store.get(key) is None
    assert not packed_resident(tmp_path, key)
    assert list((tmp_path / "quarantine").iterdir())


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("damage", ["syntax", "missing"])
def test_invalid_packed_index_is_quarantined_and_new_entries_round_trip(tmp_path, implementation, damage):
    import forge3d
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, False)
    key = leaf_key(b"before")
    store.put_leaf(key, b"before", label="test.invalid-index")
    if damage == "syntax":
        (tmp_path / _Store.INDEX).write_bytes(b"{broken index")
    else:
        (tmp_path / _Store.INDEX).unlink()
    result = (store.verify() if implementation == "python" else
              forge3d.anamnesis_store_verify(tmp_path, _DEFAULT_MAX_BYTES))
    assert result["valid"] == 0 and result["quarantined"] == 1
    assert list((tmp_path / "quarantine").iterdir())
    new_key = leaf_key(b"after")
    store.put_leaf(new_key, b"after", label="test.after-invalid-index")
    assert _Store(tmp_path, _DEFAULT_MAX_BYTES, False).get(new_key)[0] == b"after"
    assert forge3d.anamnesis_store_get(tmp_path, new_key, _DEFAULT_MAX_BYTES) == b"after"
    assert store.get(key) is None


def test_native_verify_rejects_an_invalid_range_for_an_empty_payload(tmp_path):
    import forge3d
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES
    from anamnesis_packed_store import index, INDEX

    key = forge3d.anamnesis_store_put_leaf(tmp_path, b"", "test.empty-range", _DEFAULT_MAX_BYTES)
    value = index(tmp_path)
    value["entries"][key]["length"] = 1  # No such byte exists in the data file.
    (tmp_path / INDEX).write_text(json.dumps(value), encoding="utf-8")
    report = forge3d.anamnesis_store_verify(tmp_path, _DEFAULT_MAX_BYTES)
    assert report["valid"] == 0
    assert report["quarantined"] == 1
    assert forge3d.anamnesis_store_get(tmp_path, key, _DEFAULT_MAX_BYTES) is None


def test_packed_append_preserves_prefix_and_native_interoperability(tmp_path, monkeypatch):
    import forge3d
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    store.put_leaf(leaf_key(b"first"), b"first", label="test.append")
    before = (tmp_path / _Store.INDEX).read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("adding one record must not serialize the entire resident index")
    monkeypatch.setattr(store, "_encoded_index", forbidden)
    store.put_leaf(leaf_key(b"second"), b"second", label="test.append")
    after = (tmp_path / _Store.INDEX).read_bytes()
    assert after.startswith(before[:-2])
    assert forge3d.anamnesis_store_get(tmp_path, leaf_key(b"first"), _DEFAULT_MAX_BYTES) == b"first"
    assert forge3d.anamnesis_store_get(tmp_path, leaf_key(b"second"), _DEFAULT_MAX_BYTES) == b"second"
    # A native rewrite must remain appendable by a freshly opened Python store.
    native_key = forge3d.anamnesis_store_put_leaf(tmp_path, b"native", "test.append", _DEFAULT_MAX_BYTES)
    reopened = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    monkeypatch.setattr(reopened, "_encoded_index", forbidden)
    reopened.put_leaf(leaf_key(b"third"), b"third", label="test.append")
    assert reopened.get(native_key)[0] == b"native"
    assert forge3d.anamnesis_store_get(tmp_path, leaf_key(b"third"), _DEFAULT_MAX_BYTES) == b"third"


@pytest.mark.parametrize("evict", [False, True])
def test_packed_preload_is_read_only_and_rechecks_residency(tmp_path, monkeypatch, evict):
    from pathlib import Path
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, _preload_packed_store, _PACKED_PRELOADS, leaf_key
    import forge3d
    key = leaf_key(b"snapshot")
    seed = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    seed.put_leaf(key, b"snapshot", label="test.preload")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    with _preload_packed_store(tmp_path):
        snapshot = _PACKED_PRELOADS.pending[os.path.abspath(tmp_path)].result()
        assert snapshot is not None
        assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
        if evict:
            assert forge3d.anamnesis_store_gc(tmp_path, 0) > 0
            assert _Store(tmp_path, _DEFAULT_MAX_BYTES, True).get(key) is None
        else:
            read_bytes = Path.read_bytes
            def forbidden(path):
                assert path.name not in {_Store.DATA, _Store.INDEX}, "prepared snapshot should be reused"
                return read_bytes(path)
            monkeypatch.setattr(Path, "read_bytes", forbidden)
            assert _Store(tmp_path, _DEFAULT_MAX_BYTES, True).get(key)[0] == b"snapshot"


def test_warm_pooled_store_does_not_repeat_packed_preload(tmp_path, monkeypatch):
    import forge3d.anamnesis as an
    store = an._pooled_store(tmp_path, an._DEFAULT_MAX_BYTES, True)
    key = an.leaf_key(b"warm")
    store.put_leaf(key, b"warm", label="test.preload")
    def forbidden(*args):
        raise AssertionError("an open pooled store must not reload the packed files speculatively")
    monkeypatch.setattr(an, "_load_packed_snapshot", forbidden)
    with an._preload_packed_store(tmp_path):
        assert an._pooled_store(tmp_path, an._DEFAULT_MAX_BYTES, True) is store
        assert store.get(key)[0] == b"warm"


def test_packed_growing_buffer_returns_owned_immutable_bytes(tmp_path):
    from forge3d.anamnesis import leaf_key, _DEFAULT_MAX_BYTES
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    first = leaf_key(b"owned first")
    store.put_leaf(first, b"owned first", label="test.buffer")
    restored = store.get(first)[0]
    assert type(restored) is bytes
    store.put_leaf(leaf_key(b"appended"), b"appended", label="test.buffer")
    store.gc(0)
    assert restored == b"owned first"


def test_packed_segment_slices_match_contiguous_bytes():
    from forge3d.anamnesis import _PackedBytes
    resident = b"resident"
    data = _PackedBytes(resident)
    data.extend(b"append")
    reference = resident + b"append"
    assert data[:] == reference
    assert data[len(resident)-1:len(resident)+1] == reference[len(resident)-1:len(resident)+1]
    assert data[len(reference):] == b""
    assert data[::-1] == reference[::-1]
    assert data.chunks[0] is resident  # The resident prefix was not copied.


def test_packed_reserved_appends_track_full_footprint_without_rescanning(tmp_path, monkeypatch):
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    store.put_leaf(leaf_key(b"resident"), b"resident", label="test.accounting")
    def forbidden():
        raise AssertionError("an admitted append to existing files must not rescan the tree")
    with store.protect_reads([]), monkeypatch.context() as patch:
        patch.setattr(store, "disk_bytes", forbidden)
        store.put_leaf(leaf_key(b"append"), b"append", label="test.accounting")
        actual = sum(path.stat().st_size for path in (tmp_path, *tmp_path.rglob("*")))
        assert store._current_bytes == actual
        assert actual <= store.max_bytes
    assert type(store.get(leaf_key(b"resident"))[0]) is bytes
    assert store.get(leaf_key(b"append"))[0] == b"append"


def test_native_packed_snapshot_verifies_bytes_without_file_io(tmp_path, monkeypatch):
    from pathlib import Path
    from forge3d._native import get_native_module
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key
    native = get_native_module()
    key = leaf_key(b"verified snapshot")
    seed = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    seed.put_leaf(key, b"verified snapshot", label="test.native-snapshot")
    raw_index = (tmp_path / _Store.INDEX).read_bytes()
    raw_data = (tmp_path / _Store.DATA).read_bytes()
    snapshot = native._anamnesis_packed_snapshot(raw_index, raw_data)
    absent = leaf_key(b"absent")
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("native snapshot must perform no file I/O"))
    assert snapshot.verify([key, absent]) == {key: b"verified snapshot"}
    damaged = native._anamnesis_packed_snapshot(raw_index, b"!" + raw_data[1:])
    assert damaged.verify([key]) == {key: None}


def test_packed_prefetch_uses_shared_native_integrity_validation(tmp_path, monkeypatch):
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key
    key = leaf_key(b"required")
    seed = _Store(tmp_path, _DEFAULT_MAX_BYTES, False)
    seed.put_leaf(key, b"required", label="test.native-prefetch")
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, False)
    assert store._native_snapshot is not None
    monkeypatch.setattr(store, "_read_entry", lambda *args: pytest.fail("required packed keys should use native batch validation"))
    with store.protect_reads([key]), store.prefetch_reads([key]):
        assert store.get(key)[0] == b"required"
        assert store._native_snapshot is None


def test_native_label_scene_preparation_rejects_zero_dimensions_before_gpu_work():
    from forge3d._native import get_native_module
    with pytest.raises(ValueError, match="dimensions must be non-zero"):
        get_native_module()._anamnesis_prepare_label_scene(0, 64)



def _batch_pass_write(label, blob, frame=0):
    from forge3d.anamnesis import _pass_material, capability_fingerprint, engine_fingerprint, pass_key
    pipeline, uniforms = b"batch.pipeline", label.encode("utf-8")
    caps = capability_fingerprint({"backend": "reference-cpu"})
    engine = engine_fingerprint()
    key = pass_key(label, pipeline, uniforms, (), caps, engine)
    material = _pass_material(label, pipeline, uniforms, (), caps, engine)
    return key, blob, material, frame, 0.0


def test_packed_pass_batch_opens_one_file_pair_and_round_trips_natively(tmp_path, monkeypatch):
    from pathlib import Path
    import forge3d
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    seed = leaf_key(b"seed")
    store.put_leaf(seed, b"seed", label="batch.seed")
    writes = [_batch_pass_write(label, blob) for label, blob in [
        ("batch.first", b"first"), ("batch.second", b"second"), ("batch.empty", b"")]]
    prefix = (tmp_path / store.INDEX).read_bytes()[:-2]
    open_path, opened = Path.open, []
    def observe(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path.parent == tmp_path and path.name in {store.DATA, store.INDEX} and mode in {"ab", "r+b"}:
            opened.append((path.name, mode))
            assert store.disk_bytes() <= store.max_bytes
        return open_path(path, *args, **kwargs)
    with store.protect_reads(()), monkeypatch.context() as patch:
        patch.setattr(Path, "open", observe)
        store.put_pass_batch(writes)
    assert opened == [(store.DATA, "ab"), (store.INDEX, "r+b")]
    assert (tmp_path / store.INDEX).read_bytes().startswith(prefix)
    reopened = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    for key, blob, _, _, _ in writes:
        assert reopened.get(key, touch=False)[0] == blob
        assert forge3d.anamnesis_store_get(tmp_path, key, _DEFAULT_MAX_BYTES) == blob
    assert reopened.get(seed)[0] == b"seed"
    assert store.disk_bytes() <= store.max_bytes


def test_packed_pass_batch_reserves_complete_growth_before_writing(tmp_path, monkeypatch):
    from pathlib import Path
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    store.put_leaf(leaf_key(b"seed"), b"seed", label="batch.seed")
    writes = [_batch_pass_write("batch.first", b"first"), _batch_pass_write("batch.second", b"second")]
    reserve, open_path, reservations = store._reserve_growth, Path.open, []
    start = store.disk_bytes()
    def reserve_checked(growth):
        assert store.disk_bytes() + growth <= store.max_bytes
        reservations.append(growth)
        return reserve(growth)
    def open_checked(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path.parent == tmp_path and path.name in {store.DATA, store.INDEX} and mode in {"ab", "r+b"}:
            assert reservations, "the full write set must be reserved first"
        return open_path(path, *args, **kwargs)
    with store.protect_reads(()), monkeypatch.context() as patch:
        patch.setattr(store, "_reserve_growth", reserve_checked)
        patch.setattr(Path, "open", open_checked)
        store.put_pass_batch(writes)
    assert reservations == [store.disk_bytes() - start]


def test_packed_pass_batch_rejects_bad_material_without_a_write(tmp_path):
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    store.put_leaf(leaf_key(b"seed"), b"seed", label="batch.seed")
    key, blob, material, frame, elapsed = _batch_pass_write("batch.bad", b"blob")
    material["uniform_sha256"] = "00" * 32
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    with pytest.raises(ValueError, match="batch pass key"):
        store.put_pass_batch([(key, blob, material, frame, elapsed)])
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before


def test_packed_pass_batch_capacity_pressure_uses_protected_serial_path(tmp_path, monkeypatch):
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    old_blob = b"protected" * 1024
    old_key = leaf_key(old_blob)
    store.put_leaf(old_key, old_blob, label="batch.protected")
    writes = [_batch_pass_write("batch.replacement", b"new" * 2048)]
    # The measured existing footprint leaves no append space, but collection
    # can replace the old payload while its verified read remains protected.
    store.max_bytes = store.disk_bytes()
    ordinary, observed = store.put_pass, []
    def observe(*args, **kwargs):
        observed.append(args[0])
        return ordinary(*args, **kwargs)
    with store.protect_reads([old_key]), store.prefetch_reads([old_key]):
        assert store.get(old_key)[0] == old_blob
        monkeypatch.setattr(store, "put_pass", observe)
        store.put_pass_batch(writes)
        assert store.get(old_key)[0] == old_blob
        assert store.disk_bytes() <= store.max_bytes
    assert observed == [writes[0][0]]
    assert _Store(tmp_path, store.max_bytes, True).get(writes[0][0])[0] == writes[0][1]


def test_packed_pass_batch_existing_key_keeps_serial_verification(tmp_path, monkeypatch):
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    write = _batch_pass_write("batch.existing", b"existing")
    key, blob, material, frame, elapsed = write
    store.put_pass(key, blob, material, frame=frame, measured_wall_ms=elapsed)
    ordinary, seen = store.put_pass, []
    def observe(*args, **kwargs):
        seen.append(args[0]);return ordinary(*args, **kwargs)
    monkeypatch.setattr(store, "put_pass", observe)
    store.put_pass_batch([write])
    assert seen == [key] and store.get(key)[0] == blob


def test_packed_pass_batch_failed_index_write_never_publishes_manifest(tmp_path, monkeypatch):
    from pathlib import Path
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, _LAST_MANIFEST, leaf_key
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    seed = leaf_key(b"seed")
    store.put_leaf(seed, b"seed", label="batch.seed")
    store.write_control(_LAST_MANIFEST, json.dumps({"entries": {"seed": seed}}).encode())
    write = _batch_pass_write("batch.failure", b"not published")
    open_path = Path.open
    def fail_index(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == tmp_path / store.INDEX and mode == "r+b":
            raise OSError("injected index failure")
        return open_path(path, *args, **kwargs)
    with store.protect_reads(()), monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_index)
        with pytest.raises(OSError, match="injected index failure"):
            store.put_pass_batch([write])
    assert not (tmp_path / _LAST_MANIFEST).exists()
    reopened = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    assert reopened.get(seed)[0] == b"seed" and reopened.get(write[0]) is None
    assert store.disk_bytes() <= store.max_bytes



def test_packed_pass_batch_replans_if_reservation_relocates_entries(tmp_path, monkeypatch):
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key
    from anamnesis_packed_store import index
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    old_blob = b"old-seed" * 1024
    old_key = leaf_key(old_blob)
    store.put_leaf(old_key, old_blob, label="batch.relocation")
    write = _batch_pass_write("batch.after-relocation", b"new")
    reserve, ordinary = store._reserve_growth, store.put_pass
    relocated, fallback = [], []
    def relocate_once(growth):
        result = reserve(growth)
        if not relocated:
            relocated.append(True)
            store.gc(store.disk_bytes() - len(old_blob))
        return result
    def observe(*args, **kwargs):
        fallback.append(args[0]);return ordinary(*args, **kwargs)
    with store.protect_reads([old_key]), store.prefetch_reads([old_key]):
        assert store.get(old_key)[0] == old_blob
        monkeypatch.setattr(store, "_reserve_growth", relocate_once)
        monkeypatch.setattr(store, "put_pass", observe)
        store.put_pass_batch([write])
        assert store.get(old_key)[0] == old_blob
    assert fallback == [write[0]]
    assert index(tmp_path)["entries"][write[0]]["offset"] == 0
    assert _Store(tmp_path, _DEFAULT_MAX_BYTES, True).get(write[0])[0] == b"new"


def test_packed_pass_batch_does_not_overwrite_a_concurrent_native_append(tmp_path):
    import forge3d
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    seed = leaf_key(b"seed")
    store.put_leaf(seed, b"seed", label="batch.seed")
    write = _batch_pass_write("batch.concurrent", b"new")
    with store.protect_reads([seed]), store.prefetch_reads([seed]):
        assert store.get(seed)[0] == b"seed"
        foreign = forge3d.anamnesis_store_put_leaf(tmp_path, b"foreign", "batch.foreign", _DEFAULT_MAX_BYTES)
        before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
        with pytest.raises(RuntimeError, match="changed during batched append"):
            store.put_pass_batch([write])
        assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before
    assert forge3d.anamnesis_store_get(tmp_path, foreign, _DEFAULT_MAX_BYTES) == b"foreign"
    # Reuse outside the active frozen run refreshes and can append normally.
    store.put_pass_batch([write])
    assert forge3d.anamnesis_store_get(tmp_path, write[0], _DEFAULT_MAX_BYTES) == b"new"


@pytest.mark.parametrize("field", ["key", "self_hash"])
@pytest.mark.parametrize("invalid", ["\U0001f4a9" * 16, "\u20ac" * 21 + "0", "+0" * 32])
def test_native_packed_snapshot_rejects_malformed_key_encodings(tmp_path, field, invalid):
    from forge3d._native import get_native_module
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key

    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    store.put_leaf(leaf_key(b"owned"), b"owned", label="test.invalid-key")
    index = json.loads((tmp_path / _Store.INDEX).read_bytes())
    record = next(iter(index["entries"].values()))
    record["metadata"][field] = invalid
    assert len(invalid.encode("utf-8")) == 64
    with pytest.raises(ValueError, match="non-hexadecimal"):
        get_native_module()._anamnesis_packed_snapshot(
            json.dumps(index, ensure_ascii=False).encode("utf-8"),
            (tmp_path / _Store.DATA).read_bytes(),
        )


def test_public_verify_quarantines_unicode_key_metadata_without_panicking(tmp_path):
    from forge3d.anamnesis import _DEFAULT_MAX_BYTES, leaf_key, verify

    key = leaf_key(b"owned")
    store = _Store(tmp_path, _DEFAULT_MAX_BYTES, True)
    store.put_leaf(key, b"owned", label="test.invalid-key")
    index = json.loads((tmp_path / _Store.INDEX).read_bytes())
    next(iter(index["entries"].values()))["metadata"]["key"] = "\U0001f4a9" * 16
    (tmp_path / _Store.INDEX).write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    report = verify(tmp_path)
    assert report["valid"] == 0
    assert any((tmp_path / "quarantine").iterdir())
    assert _Store(tmp_path, _DEFAULT_MAX_BYTES, True).get(key) is None
