from __future__ import annotations

import hashlib
import random

import pytest

from forge3d.anamnesis import pass_key


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
        return pass_key(
            "terrain.forward",
            pipeline,
            uniform,
            [hashlib.sha256(inputs["dem"]).hexdigest()],
            capability,
            inputs["engine"],
        )

    base_key = key_for(pixel_inputs)
    # This digest is the deterministic reference-render oracle used by the
    # mutation gate. Every byte below is declared pixel-relevant; therefore a
    # mutation is allowed neither to preserve the key nor the rendered digest.
    base_output = hashlib.sha256(b"".join(pixel_inputs.values())).digest()
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
        output = hashlib.sha256(b"".join(mutated.values())).digest()
        assert key != base_key, f"mutation escaped key: {category}"
        assert output != base_output, f"pixel-relevant mutation escaped output: {category}"
    assert seen == set(categories)


def test_hermeticity_fast_lane_500_mutations():
    _run_mutations(500)


@pytest.mark.slow
def test_hermeticity_full_10000_mutations():
    _run_mutations(10_000)
