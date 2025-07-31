# Claude Code hooks used in this project

This project defines **hooks** in `.claude/settings.json` to keep sub‑agents safe, consistent, and reproducible.

## PreToolUse hooks

1. **block-sensitive-writes**  
   Blocks `Edit`/`Write` to sensitive paths:
   - `**/secrets/**`, `.pypirc`, `.npmrc`, `id_rsa`, `credentials*`, `.cargo/credentials`  
   **Why:** Prevents accidental credential leakage.

2. **block-dangerous-bash**  
   Blocks dangerous shell patterns:
   - `sudo *`, `rm -rf /*`, `curl|sh`, `wget|sh`, and system package managers (`apt-get`, `yum`)  
   **Why:** Avoids destructive or privileged commands in CI/dev.

3. **auto-format-before-large-diffs**  
   Before applying edits to `.rs`, `.toml`, `.py`, `.md`, run best‑effort formatters:
   - `cargo fmt` (Rust), `black` and `ruff --fix` (Python)  
   **Behavior:** Non‑fatal — if tools are missing, the hook continues. Install tools via `make claude-setup`.

## Subagent Stop hook

**log-ssim-and-metrics**  
Aggregates any `**/metrics*.json` and PNG SHA‑256 hashes into `.claude/logs/last_run.json` at the end of a sub‑agent run.  
**Why:** Keeps a lightweight audit trail of determinism checks and outputs without polluting PRs.

---

## Recommended workflow

- Create/modify code → hooks auto‑format and guard dangerous actions.
- Run determinism tests (SSIM) or examples → on stop, metrics + image hashes are logged.
- Use the **critic** agent to verify acceptance criteria before merging.

## Troubleshooting

- If formatting doesn’t run, install tools: `make claude-setup`.
- To bypass a blocked action, request explicit approval or use the `release-captain` agent for publishes.
