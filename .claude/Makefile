.PHONY: claude-setup fmt lint license-report

# Install optional dev tools used by hooks and CI docs
claude-setup:
	@echo "Installing Python formatters (black, ruff) and Rust cargo-about..."
	@python3 -m pip install --upgrade pip >/dev/null 2>&1 || true
	@python3 -m pip install -q black ruff || python -m pip install -q black ruff
	@command -v cargo >/dev/null 2>&1 && cargo install cargo-about --locked || { echo "cargo not found; skipping cargo-about (install Rust to enable)"; true; }
	@echo "Done. If cargo-about was skipped, install Rust (https://www.rust-lang.org/tools/install) and re-run."

# Convenience targets
fmt:
	@echo "Running formatters (Rust, Python)..."
	@cargo fmt || true
	@python3 -m black . || true
	@python3 -m ruff check --fix . || true

lint:
	@echo "Running ruff linter..."
	@python3 -m ruff check . || true
	@echo "Running clippy..."
	@cargo clippy -- -D warnings || true

# Generate third-party license report (requires cargo-about)
license-report:
	@echo "Generating LICENSES-THIRD-PARTY.txt (requires cargo-about)..."
	@cargo about generate about.hbs > LICENSES-THIRD-PARTY.txt || { echo "cargo-about not available or template missing"; exit 0; }
	@echo "Wrote LICENSES-THIRD-PARTY.txt"
