# Contributing to forge3d

## Development Setup

1. Install Rust from <https://rustup.rs>.
2. Clone the repo: `git clone https://github.com/forge3d/forge3d`
3. Create a Python environment for Python 3.10+.
4. Install build tooling: `python -m pip install -U pip maturin pytest`
5. Build the extension in development mode: `maturin develop`

Optional extras:

- `python -m pip install -e ".[jupyter]"`
- `python -m pip install -e ".[datasets]"`
- `python -m pip install -e ".[all]"`

## Running Tests

```bash
# Python tests
python -m pytest tests/ -v --tb=short

# Rust tests
cargo test --workspace --all-features

# Focused package smoke checks
python -m pytest tests/test_install_smoke.py -v
```

## Code Style

- Python: keep public signatures typed and match the existing module layout.
- Rust: run `cargo fmt` and keep `cargo clippy` clean when touching Rust code.
- Docs: keep tutorial snippets copy-pasteable and aligned with the current API.

## Pull Requests

- Keep changes scoped to one feature or fix.
- Include tests for public API changes.
- Update docs when behavior or packaging changes.
- Do not revert unrelated user work in the tree.
