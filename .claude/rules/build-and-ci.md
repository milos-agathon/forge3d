---
paths: ["Cargo.toml", "pyproject.toml", ".cargo/**", ".github/workflows/**", "pytest.ini", "**/conftest.py"]
---

# Build, test, and CI facts

- Build with maturin/PyO3. Wheels use the `release-lto` profile.
- Use `cargo forge3d-clippy`, never plain `cargo clippy`.
- The portable CI feature list is:
  `default,async_readback,copc_laz,cog_streaming,gis-remote,geos-topology,weighted-oit,wsI_bigbuf,wsI_double_buf,enable-pbr,enable-tbn,enable-normal-mapping,enable-hdr-offscreen,enable-renderer-config,enable-staging-rings,shader-contract-asserts`.
  `.github/workflows/ci.yml` is authoritative and the honesty gate locks every
  duplicate list. Ubuntu separately installs PROJ prerequisites and checks the
  `proj` feature.
- Python CI runs every tracked `tests/test_*.py` through
  `scripts/ci_pytest_lane.py` except dated, owned entries in `tests/UNRUN.toml`.
- Visual goldens run on macOS Metal. A successful probe executes pixel and WGSL
  provenance checks; a genuinely absent adapter emits an ABSENT marker; a probe
  crash or pixel mismatch fails the aggregate.
- Protected internal/release golden runs require the Actions secret
  `FORGE3D_CERT_SIGNING_KEY` and verify committed certificates against the
  pinned production public key. Fork PRs are explicitly untrusted and do not
  claim production-signing validation.
- `determinism-matrix` is an experimental, non-required diagnostic until the
  documented fixed-function filtering and downstream compiler differences in
  `src/shaders/includes/determinism.wgsl` are eliminated. Its per-backend
  render jobs and zero-byte diff must stay loud; cross-adapter hash divergence
  is permitted as diagnostic evidence and must never be presented as a green
  determinism guarantee or used to replace the committed hash casually. The
  protected release gate remains the `CI Success` aggregate.
