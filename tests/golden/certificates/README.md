# Recipe golden certificates

This directory holds one committed, Ed25519-signed **RenderCertificate** per
`RECIPE_GOLDENS` scene in `tests/test_recipe_goldens.py`, plus the public key
(`signing.pub`) used to verify them.

## What these files are

Each `<scene_id>.json` is the full execution report for that scene's golden
render — engine version + per-module WGSL source hashes, adapter identity,
negotiated GPU capabilities, the per-pass timing ledger, the peak allocation
ledger, the (empty, for a clean golden) degradation list, and an Ed25519
`signature` block. The certificate is assembled by
`forge3d.diagnostics.render_certificate()` and written by
`forge3d.certificate.write_certificate()`.

`signing.pub` is the 64-char hex Ed25519 **public** key that verifies every
certificate in this directory.

## How they are used by the test suite

In normal (hardware-backed) mode, `test_recipe_goldens_render_and_match`:

1. asserts the committed `<scene_id>.json` exists and
   `forge3d.certificate.verify(cert, signing.pub)` is `True`;
2. asserts the committed certificate's `degradations` are empty; and
3. asserts the FRESH in-process render's `engine.wgsl_module_hashes` match the
   committed certificate's — the load-bearing **shader tamper** check that ties
   the committed certificates to the current WGSL sources. If a shader changes,
   this assertion fails with a message directing you to regenerate.

## How to regenerate

Run the recipe golden suite with the update flag (this also refreshes the
committed pixel goldens under `tests/golden/recipes/`):

```bash
FORGE3D_UPDATE_RECIPE_GOLDENS=1 .venv/Scripts/python -m pytest tests/test_recipe_goldens.py -q
```

Regenerate only after you have verified the pixel goldens are correct. The WGSL
hash check exists precisely so that a shader edit forces a conscious
regeneration rather than silently drifting.

## The signing key is a NON-PRODUCTION development key

By default the certificates are signed with the fixed development seed in
`forge3d.certificate.DEV_SIGNING_SEED`, derived deterministically from a public
constant (see `forge3d.certificate._dev_signing_seed`). This lets anyone
reproduce and re-verify these signatures offline with a stock CPython
interpreter and no secrets:

```bash
python -m forge3d.certificate verify tests/golden/certificates/mapscene_terrain_raster.json \
    --pubkey tests/golden/certificates/signing.pub
```

The dev key provides **tamper evidence and reproducibility, not trust**. It is
NOT a production signing key — it proves the certificate was produced by the
committed pipeline and has not been altered, nothing more. A real deployment
must supply its own 32-byte Ed25519 seed via the `FORGE3D_CERT_SIGNING_KEY`
environment variable (64-char hex) or the `seed=` argument to
`forge3d.certificate.sign_certificate`, and publish the matching public key
out of band.
