# Agent Reflections

## P0.2: Wrapper/Native Callsite Mismatch (2026-02-19)

- When probing for native API methods, always verify whether a function is a **module-level free function** or an **instance method on a class**. The old `offscreen.py` checked `hasattr(_native, "render_rgba")` but `render_rgba` is a method on `Scene` instances, not a module-level export. This pattern of dead probes can persist silently for a long time because the fallback path works.
- The PyO3 `#[pyo3(text_signature = "($self)")]` annotation on `Scene.render_rgba` confirms it takes no positional arguments beyond `self` -- width and height are baked in at `Scene` construction time. Always read the Rust signature before wiring up Python calls.
- When removing dead code that probed nonexistent module-level functions, also clean up imports that become unused (e.g., `_forge3d`, `warnings` in `viewer.py`).
- Contract tests that assert "function X does NOT exist at module level" are valuable for documenting architectural decisions and preventing future developers from re-introducing the same mistake.

## P0.3: Register Orphaned PyO3 Classes (2026-02-19)

- When a Rust struct has `#[pyclass]` but no corresponding `m.add_class::<T>()?;` in the `#[pymodule]` init function, the class is invisible to Python even though it compiles fine. Always check that every `#[pyclass]` has a matching registration in `src/lib.rs`.
- When importing multiple types from the same module (e.g., `crate::sdf::py`), consolidate into a single `use` statement with a braced group rather than adding separate `use` lines per type.
- Negative contract tests ("X is NOT registered") should be flipped to positive assertions ("X IS registered") when the registration is intentionally added. Also add construction tests that verify the class is not just importable but actually usable (constructible, methods callable).
- The `EXPECTED_CLASSES` list in Section 1 of `test_api_contracts.py` must be updated whenever new `m.add_class` registrations are added, otherwise the parametrized existence test won't cover them.

## P0.4: Resolve Mesh TBN Drift (2026-02-19)

- When Rust code is gated behind a Cargo feature flag (e.g., `#[cfg(feature = "enable-tbn")]`), you must also add that feature to the maturin build features in `pyproject.toml`. Otherwise the module compiles but the functions are excluded from the extension. The `pyproject.toml` `[tool.maturin].features` list is the single source of truth for which features are included in the Python wheel.
- PyO3 `#[pyfunction]` wrappers that convert Rust structs to Python dicts should use a shared helper function (e.g., `tbn_result_to_py_dict`) to avoid duplicating dict-building logic across multiple wrapper functions.
- When the Python wrapper (e.g., `mesh.py`) has a feature-detection guard like `_HAS_TBN = hasattr(_forge3d, 'mesh_generate_cube_tbn')`, simply registering the function in the pymodule is enough to flip the flag. No changes to the Python wrapper itself are needed.
- For feature-gated registrations in `lib.rs`, wrap related function registrations in a `#[cfg(feature = "...")]` block to keep them conditional, matching the module-level gating in `mod.rs`.
- The `EXPECTED_FUNCTIONS` list in the contract tests should be updated alongside registration to maintain the contract lock.

## P1.1: SSGI/SSR Settings Wiring (2026-02-19)

- When adding state tracking to `Scene` for a new feature (SSGI, SSR, bloom, etc.), follow the pattern: add fields to the struct, initialize with defaults in the constructor, add `#[pymethods]` for enable/disable/is_enabled/set_settings/get_settings, and update the `.pyi` type stubs.
- `get_*_settings()` should return a dict (not a typed object) for maximum flexibility and compatibility with the Python side. Use `pyo3::types::PyDict::new(py)` and `dict.set_item(...)`.
- Behavior tests for Scene methods should test at the class level (`hasattr(_native.Scene, "method_name")`) when instance construction is blocked by GPU/shader issues. This avoids test fragility while still asserting the API contract.

## P1.2: Bloom Execute Wiring via Resource Pool (2026-02-19)

- The `PostFxEffect::execute()` trait method needs `queue: &Queue` to upload uniform data before dispatching compute passes. When modifying a trait signature, check that the only implementor is `BloomEffect` to avoid a wider refactor.
- `BloomEffect` differs from `TerrainBloomProcessor` in that it must use the `PostFxResourcePool`'s ping-pong texture pairs for intermediate storage rather than owning its own textures. Allocate pairs during `initialize()` and retrieve views during `execute()`.
- The bloom composite pass requires a **4-binding layout** (original + bloom + output + uniforms), distinct from the 3-binding brightpass/blur layouts. This was missing from the original stub and had to be added alongside the composite pipeline and shader loading.
- Bloom default-off semantics are critical for backward compatibility: `BloomConfig::default().enabled == false`, and `execute()` returns `Ok(())` immediately when disabled.
- The `PostFxChain::execute_chain()` must also accept `queue` and forward it to each effect's `execute()` call.

## P2.1: Point Cloud GPU Rendering Path (2026-02-19)

- The Rust `PointBuffer` stores positions as flat `Vec<f32>` [x,y,z,...] and colors as `Option<Vec<u8>>` [r,g,b,...]. The `create_gpu_buffer()` method interleaves them into [x,y,z,r,g,b] per point, normalizing u8 colors to 0..1 f32. Default is white when no colors are present.
- PyO3 bindings for simple data structs work best as thin wrapper types (e.g., `PyPointBuffer` wrapping `pointcloud::PointBuffer`) rather than deriving `#[pyclass]` directly on the Rust struct, because the inner struct lacks `Clone`/`Copy` and holds non-PyO3-compatible fields.
- When returning numpy arrays from PyO3, prefer `PyArray1::from_vec_bound(py, data)` over the deprecated `from_vec(py, data)` in pyo3 0.21+.
- Constructor validation for `PointBuffer` (positions length divisible by 3, colors matching point count) prevents downstream GPU buffer mismatches. Always validate at the boundary.
- The `renderer.rs` file (353 lines) is slightly over the 300-line guideline due to pre-existing duplication between `load_copc_points` and `load_ept_points`. A future refactor could extract a common `load_points_generic` helper.
- `MemoryReport` separates observation from policy: it reports cache_used, cache_budget, utilization, and entry_count without deciding what to do about high utilization. This keeps the renderer testable without GPU access.
