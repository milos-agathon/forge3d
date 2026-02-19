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
