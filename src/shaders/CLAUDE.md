# shaders/ — Memory for WGSL conventions

We keep shaders **deterministic**, **readable**, and **backend‑friendly** (Vulkan/Metal/DX12).

## General rules
- WGSL modules live under `src/shaders/*.wgsl` and are loaded via `include_str!` in Rust.
- Keep modules small and composable (e.g., `tonemap.wgsl`, `normals.wgsl`, `colormap.wgsl`).
- Document entry points and expected bind groups in comments.

## Determinism
- Prefer `textureSampleLevel(tex, samp, uv, 0.0)` to avoid derivative ambiguity.
- Disable MSAA in MVP; AA is done analytically in shader (smoothstep/SDF) for lines/points.
- Avoid dynamic branching for edge conditions where possible; use continuous masks.

## Color & tonemap
- Perform lighting in **linear**. Target format is **sRGB**; do **not** re‑gamma before write.
- Provide simple tonemap functions:
  - `reinhard(x)` for MVP
  - (optional, later) ACES approx
- Keep exposure uniform; default 1.0.

## Terrain specifics
- VS reconstructs `world_z` from an `R32Float` height texture. Use `(dx, dy)` spacing and `exaggeration` to scale Z.
- FS computes normals by forward differences of height with explicit LOD 0; normalize the cross product carefully.
- Colormap via 256×1 LUT texture: map normalized height `((h - h_min) / (h_max - h_min))` to UV.x, clamp to [0,1].

## Lines (screen‑space quads)
- VS expands per‑segment **unit quad** in clip space using viewport size; supply `cap/join` data via instance attributes.
- FS computes edge distance and uses **smoothstep** for antialiasing.
- Implement `miter` with a limit; fall back to `bevel` when exceeded.

## Points (instanced sprites)
- VS expands a unit quad sized in **pixels**; uses viewport to convert to NDC.
- FS uses SDF circle/square; AA via smooth edge width in device pixels.

## Layout & bindings
- Keep a clear mapping: `@group(0) @binding(0)` → globals; `@group(1..)` textures/samplers.
- Ensure host/Rust uniform structs match WGSL alignment (16‑byte multiples).

## Testing
- Compile on all CI backends; avoid backend‑specific extensions in shaders.
- Provide small scene renders as goldens for shader changes.
