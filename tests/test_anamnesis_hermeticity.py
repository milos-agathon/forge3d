from __future__ import annotations

import hashlib
import random

import pytest

import forge3d.anamnesis as anamnesis


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

    def key_for(inputs: dict[str, bytes]) -> str:
        pipeline = b"".join(
            inputs[name]
            for name in (
                "wgsl",
                "sampler",
                "blend",
                "depth",
                "viewport",
                "scissor",
                "clear",
                "texture",
                "primitive",
            )
        )
        uniform = inputs["uniform"] + inputs["seed"] + inputs["frame_index"]
        capability = (
            inputs["capability"] + inputs["backend"] + inputs["dx12_compiler"]
        )
        return anamnesis.pass_key(
            "terrain.forward",
            pipeline,
            uniform,
            [hashlib.sha256(inputs["dem"]).hexdigest()],
            capability,
            inputs["engine"],
        )

    def render_for(inputs: dict[str, bytes]) -> str:
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
                frames=[int.from_bytes(inputs["frame_index"], "little")],
                cache=None,
                capabilities={"granted": [inputs["capability"].hex()]},
            )
        finally:
            anamnesis.engine_fingerprint = original_engine
        return result.frame_hashes[0]

    base_key = key_for(pixel_inputs)
    base_output = render_for(pixel_inputs)
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
        key = key_for(mutated)
        output = render_for(mutated)
        assert key != base_key, f"mutation escaped key: {category}"
        assert output != base_output, f"pixel-relevant mutation escaped output: {category}"
    assert seen == set(categories)


def test_hermeticity_fast_lane_500_mutations():
    _run_mutations(500)


@pytest.mark.slow
def test_hermeticity_full_10000_mutations():
    _run_mutations(10_000)
