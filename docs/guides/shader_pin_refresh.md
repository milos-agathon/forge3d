# Owner-approved terrain source pin refresh

The arithmetic rewrite command documented in `src/verify/det_rewrite.rs` rewrites
WGSL and does not refresh proof pins. Use the computed terrain-pin rewrite below;
never copy a printed hash into the Rust constant by hand. This narrowly scoped
command updates only `PINNED_TERRAIN_SOURCE_HASH`, using the assembled module and
the proof engine's existing line-ending-independent hash.

Obtain the owner's authorization for the source change and pin refresh first.
For C01, Milos's 2026-10-15 approval requires all 22 default recipe images to stay
byte-identical. Retain that evidence and the shader diff. Signed certificates
record shader hashes separately; this command neither updates nor signs them.
Ask before any re-signing.

Use the native feature list selected for the build. A dry run is read-only and
fails with the old and computed values if a refresh is needed:

```powershell
cargo test --locked --lib --features $c01features terrain_source_pin_rewrite -- --nocapture
$env:FORGE3D_REFRESH_TERRAIN_PIN = '1'
cargo test --locked --lib --features $c01features terrain_source_pin_rewrite -- --nocapture
Remove-Item Env:FORGE3D_REFRESH_TERRAIN_PIN
cargo test --locked --lib --features $c01features terrain_renderer_source_hash_is_pinned -- --nocapture
```

The final command recompiles the changed constant and checks the strict pin.
Then execute shader assembly/proof checks and rebuild the private extension
before Python/GPU validation. A pin rewrite is not proof of pixel compatibility.
