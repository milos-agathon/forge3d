# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Rust crate (core, terrain, vector, shaders). Entry is `src/lib.rs`.
- `python/forge3d/`: Python API layer and utilities (`_forge3d` via PyO3, `py.typed`, stubs `__init__.pyi`).
- `examples/`: Python demos (e.g. `terrain_single_tile.py`) and Rust example(s) under `examples/perf/`.
- `tests/`: PyTest suite; GPU-aware tests auto-skip if no adapter.
- `docs/`, `assets/`, `data/`, `bench/`: Documentation, logos/colormaps, sample RGBA data, and benches.

## Build, Test, and Development Commands
```bash
# Build Python extension in-place (recommended for dev)
pip install -U maturin
maturin develop --release

# Run tests (quiet)
pytest -q

# Run examples
python examples/terrain_single_tile.py
cargo run --example split_vs_single_bg --release

# Optional: Rust build/lints
cargo build --release
cargo fmt && cargo clippy --all-targets --all-features -D warnings
```
Tips: set `RUST_LOG=info` for debug logs; force a backend with `WGPU_BACKEND=dx12|vulkan|metal` if needed.

## Coding Style & Naming Conventions
- Rust: format with `cargo fmt`; lint with `clippy`. `snake_case` for modules/functions, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for consts.
- Python: follow PEP 8 and use type hints. Keep public signatures in sync with `python/forge3d/__init__.pyi`; the package is typed via `py.typed`.
- API: prefer explicit shapes/dtypes; ensure NumPy arrays are C‑contiguous where required.

## Testing Guidelines
- Framework: PyTest (`pytest.ini` sets `tests/`). Name tests `test_*.py`.
- Coverage: not enforced, but add tests alongside features (determinism, NumPy interop, GPU parity).
- GPU: tests skip automatically if no compatible adapter; design tests to be deterministic.

## Commit & Pull Request Guidelines
- Changelog: update `CHANGELOG.md` under “Unreleased” (Keep a Changelog; SemVer).
- Commits: imperative mood, focused scope; include rationale and affected modules (e.g., "vector: fix AA line joins").
- PRs: include summary, linked issues, test results (`pytest -q`), and artifacts when relevant (PNG outputs, bench JSON). Note feature flags used (e.g., `wsI_bigbuf`).

## Security & Configuration Tips
- Validate shapes/dtypes and ranges at boundaries; avoid panics in FFI.
- Respect memory budget utilities; don’t check in large binaries; keep assets in `assets/`/`data/`.

# HEADER COMMENTS
- EVERY file HAS TO start with 4 lines of comments!
1. exact file location in codebase
2. clear description of what this file does
3. clear description of WHY this file exists
4. RELEVANT FILES:comma-separated list of 2-4 most relevant files
- NEVER delete these "header comments" from the files you're editing.

# SIMPLICITY
- Always prioritize writing clean, simple, and modular code.
- do not add unnecessary complications. SIMPLE = GOOD, COMPLEX = BAD.
- Implement precisely what the user asks for, without additional features or complexity.

# READING FILES
- always read the file in full, do not be lazy
- before making any code changes, start by finding & reading ALL of the relevant files
- never make changes without reading the entire file

# EGO
- do not make assumption. do not jump to conclusions.
- you are just a Large Language Model, you are very limited.
- always consider multiple different approaches, just like a Senior Developer would

# WRITING STYLE
- each long sentence should be followed by two newline characters
- avoid long bullet lists
- write in natural, plain English. be conversational.
- avoid using overly complex language, and super long sentences
- use simple & easy-to-understand language. be concise.

# OUTPUT STYLE
- write in complete, clear sentences. like a Senior Developer when talking to a junior engineer
- always provide enough context for the user to understand -- in a simple & short way
- make sure to clearly explain your assumptions, and your conclusions