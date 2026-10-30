"""Reference-executor key propagation, not native production pixel coverage.

The 10,000 mutations below exercise a synthetic byte corpus. Passing them does
not satisfy the separate production-input/rendered-pixel hermeticity requirement.
"""

from __future__ import annotations

import random
import os
import json

import pytest

import forge3d.anamnesis as anamnesis


def test_native_parameter_inventory_covers_optional_inputs_and_validator_domains(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from anamnesis_native_mutations import (
        _EXPLICIT_OPTIONAL_CONTROLS, _finite_choices, _mutation_config,
        _optional_paths, _parameter_mutations, _paths,
    )
    from forge3d import terrain_params as tp

    config, _, _ = _mutation_config(tmp_path)
    categories, rejected, bypasses, inventory = _parameter_mutations(config, tmp_path, random.Random(0xA11A_17), None)
    assert set(inventory) == {".".join(map(str, path)) for path, _ in _paths(config)}
    optional = {".".join(map(str, path)) for path, _ in _optional_paths(config)}
    assert optional == ({name.removeprefix("optional.") for name, _, _, _ in categories
                         if name.startswith("optional.")} | set(_EXPLICIT_OPTIONAL_CONTROLS) |
                        {item["field"].removeprefix("optional.") for item in rejected
                         if item["field"].startswith("optional.")})
    assert {name for name, _, _, reason in bypasses if reason == "material_virtual_textures"} == {"vt.enabled"}
    assert set(_finite_choices(config, ("sampling", "address_u"))) == {"Repeat", "ClampToEdge", "MirrorRepeat"}

    def evolved_validator(self):
        if self.mag_filter not in {"Linear", "Nearest", "future-test-filter"}:
            raise ValueError("invalid filter")
    monkeypatch.setattr(tp.SamplingSettings, "__post_init__", evolved_validator)
    assert "future-test-filter" in _finite_choices(config, ("sampling", "mag_filter"))
    monkeypatch.undo()

    @dataclass
    class EvolvedConfig(type(config)):
        unclassified_optional: float | None = None
    evolved = EvolvedConfig(**vars(config))
    with pytest.raises(AssertionError, match="unclassified native parameter: unclassified_optional"):
        _parameter_mutations(evolved, tmp_path, random.Random(0xA11A_17), None)


def _mutate(data: bytes, rng: random.Random) -> bytes:
    changed = bytearray(data)
    index = rng.randrange(len(changed))
    changed[index] ^= rng.randrange(1, 256)
    return bytes(changed)


def _run_mutations(count: int) -> None:
    rng = random.Random(0xA11A_17)
    pixel_inputs = {
        "dem": b"DEM bytes representing exact little-endian f32 texels",
        "uniform": bytes(range(64)),
        "wgsl": b"@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1); }",
        "sampler": b"min=nearest|mag=nearest|mip=nearest|address=clamp",
        "blend": b"color=replace|alpha=replace|write_mask=rgba",
        "depth": b"format=depth32float|compare=less|write=true|bias=0,0,0",
        "viewport": b"0,0,64,64,0,1",
        "scissor": b"0,0,64,64",
        "clear": b"color=0,0,0,0|depth=1|stencil=0",
        "texture": b"rgba8unorm|mips=1|samples=1|dimension=2d",
        "primitive": b"tri-list|ccw|cull=back|unclipped=false",
        "seed": (0xA11A17).to_bytes(8, "little"),
        "frame_index": (250).to_bytes(8, "little"),
        "capability": b'{"granted":["timestamp_query"],"limits":{"max_bind_groups":4}}',
        "backend": b"vulkan",
        "dx12_compiler": b"n/a",
        "engine": b'{"crate_version":"1.33.0","git_sha":"test","naga_version":"0.19.2"}',
    }

    def build_for(inputs: dict[str, bytes]) -> tuple[str, str]:
        frame = int.from_bytes(inputs["frame_index"], "little")
        state = {
            "shader_hashes": {"mutation": inputs["wgsl"].hex()},
            "sampler": inputs["sampler"].hex(),
            "blend": inputs["blend"].hex(),
            "depth": inputs["depth"].hex(),
            "viewport": inputs["viewport"].hex(),
            "scissor": inputs["scissor"].hex(),
            "clear": inputs["clear"].hex(),
            "formats": [inputs["texture"].hex()],
            "primitive": inputs["primitive"].hex(),
            "seed": int.from_bytes(inputs["seed"], "little"),
            "backend": inputs["backend"].hex(),
            "dx12_compiler": inputs["dx12_compiler"].hex(),
        }
        recipe = {
            "terrain": {"dem_bytes": inputs["dem"].hex()},
            "camera": {"uniform_bytes": inputs["uniform"].hex()},
            "output": {"texture_bytes": inputs["texture"].hex()},
            "anamnesis_state": state,
        }
        original_engine = anamnesis.engine_fingerprint
        anamnesis.engine_fingerprint = lambda: inputs["engine"]
        try:
            result = anamnesis.render_sequence(
                recipe,
                frames=[frame],
                cache=None,
                capabilities={
                    "granted": [inputs["capability"].hex()],
                    "naga_capabilities": [inputs["capability"].hex()],
                },
             reference_only=True)
        finally:
            anamnesis.engine_fingerprint = original_engine
        return result.pass_keys[f"{frame}:frame.output"], result.frame_hashes[0]

    base_key, base_output = build_for(pixel_inputs)
    categories = tuple(pixel_inputs)
    seen = set()
    for index in range(count):
        # Round-robin selection guarantees that even the 500-case CI lane
        # exercises the entire declared pixel-input space. Mutation offsets
        # and replacement bits remain independently seeded.
        category = categories[index % len(categories)]
        seen.add(category)
        mutated = dict(pixel_inputs)
        mutated[category] = _mutate(mutated[category], rng)
        key, output = build_for(mutated)
        assert key != base_key, f"mutation escaped key: {category}"
        assert output != base_output, f"pixel-relevant mutation escaped output: {category}"
    assert seen == set(categories)


def test_hermeticity_fast_lane_500_mutations():
    _run_mutations(500)


@pytest.mark.slow
def test_hermeticity_full_10000_mutations():
    _run_mutations(10_000)


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="native production mutations require a physical GPU lane",
)
def test_native_production_10000_mutations_match_independent_fresh_frames(tmp_path, monkeypatch):
    from anamnesis_native_mutations import run_native_mutations

    monkeypatch.delenv("FORGE3D_ANAMNESIS_COMPATIBILITY_PROFILE", raising=False)
    result = run_native_mutations(tmp_path, 10_000)
    assert result["mutations"] == 10_000
    (tmp_path / "native-mutations.json").write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    if destination := os.environ.get("FORGE3D_ANAMNESIS_MUTATION_RESULT_PATH"):
        from pathlib import Path

        Path(destination).write_text(json.dumps(result, sort_keys=True), encoding="utf-8")


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="native production mutations require a physical GPU lane",
)
def test_native_production_fast_500_mutations(tmp_path, monkeypatch):
    from anamnesis_native_mutations import run_native_mutations

    monkeypatch.delenv("FORGE3D_ANAMNESIS_COMPATIBILITY_PROFILE", raising=False)
    result = run_native_mutations(tmp_path, 500)
    (tmp_path / "native-500-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert result["mutations"] == 500


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="retained parameter identity requires a physical GPU")
@pytest.mark.parametrize("case", ["detail_noise", "msaa"])
def test_retained_parameter_snapshot_cannot_alias_new_native_state(tmp_path, case):
    from dataclasses import replace
    import numpy as np
    import forge3d as f3d
    from anamnesis_native_mutations import native_config
    from forge3d.determinism import canonical_heightmap, write_canonical_hdr

    hdr = tmp_path / "environment.hdr"
    write_canonical_hdr(hdr)
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material = f3d.MaterialSet.terrain_default()
    environment = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
    height = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)

    def render(params, cache=None):
        f3d.clear_native_degradations()
        data = renderer.render_terrain_pbr_pom(material, environment, params, height, cache=cache).to_numpy()
        assert not [item for item in f3d.native_degradations() if item["kind"] in {"uncaptured_gpu_error", "validation_error"}]
        return data, dict(renderer.last_anamnesis_cache_report)

    config = native_config()
    if case == "detail_noise":
        start = config.cam_radius + config.terrain_span
        config.detail = replace(config.detail, enabled=True, albedo_noise=0.0,
                                normal_strength=0.0, fade_start=start,
                                fade_end=start + config.terrain_span)
    held = f3d.TerrainRenderParams(config)
    assert held.python_object is config
    before, _ = render(held)
    if case == "detail_noise":
        config.detail.albedo_noise = 0.5
    else:
        config.msaa_samples = 4
    assert held.python_object is config
    stored, stored_report = render(held, tmp_path / "shared")
    revised = f3d.TerrainRenderParams(config)
    recalled, recalled_report = render(revised, tmp_path / "shared")
    restored, warm_report = render(revised, tmp_path / "shared")
    fresh, _ = render(revised)
    assert np.array_equal(before, stored)
    assert not np.array_equal(stored, fresh), "control must expose the decoded state change"
    assert stored_report["pass_keys"] != recalled_report["pass_keys"]
    assert np.array_equal(recalled, fresh) and np.array_equal(restored, fresh)
    assert len(warm_report["hits"]) == 4 and warm_report["graph_command_submissions"] == 0
    assert warm_report["bypass_reason"] is None
    print(json.dumps({"case": case, "changed_pixels": int(np.count_nonzero(np.any(stored != fresh, axis=2)))}))


def test_parameter_snapshot_owns_arrays_and_preserves_float_bits():
    from dataclasses import dataclass
    import numpy as np
    from forge3d._anamnesis_params import input_snapshot

    @dataclass
    class Input:
        array: object
        number: float

    value = Input(np.array([0.25, 0.5], dtype=np.float32), 0.0)
    original = input_snapshot(value)
    assert input_snapshot(value) == original
    value.array[0] = 0.75
    assert input_snapshot(value) != original
    value.array[0] = 0.25
    value.number = -0.0
    assert input_snapshot(value) != original
