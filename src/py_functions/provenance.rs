// src/py_functions/provenance.rs
// VERITAS: PyO3 surface for sealing and verifying per-pixel provenance.
// Builds the SHA256 Merkle commitment over contributing VT tiles + the
// source-id map, signs the root with Ed25519, and emits/verifies the
// versioned provenance.json manifest.
// RELEVANT FILES: src/core/provenance.rs, python/forge3d/provenance.py,
// tools/verify_provenance.py

use super::super::*;

#[cfg(feature = "extension-module")]
use numpy::PyUntypedArrayMethods;

#[cfg(feature = "extension-module")]
use crate::core::provenance::{
    encode_source_map_leaf, encode_tile_leaf, from_hex, merkle_root, sign_root, to_hex,
    verify_root, ContributingTile, FAMILY_NAMES,
};

/// Manifest schema version emitted by `seal_provenance` and accepted by
/// `verify_provenance` / the offline verifier.
#[cfg(feature = "extension-module")]
pub(crate) const PROVENANCE_SCHEMA_VERSION: u64 = 1;

/// SHA256 over the row-major little-endian u32 raster of the source map.
#[cfg(feature = "extension-module")]
fn source_map_digest(source_map: &numpy::PyReadonlyArray2<'_, u32>) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let array = source_map.as_array();
    let mut hasher = Sha256::new();
    for value in array.iter() {
        hasher.update(value.to_le_bytes());
    }
    hasher.finalize().into()
}

#[cfg(feature = "extension-module")]
fn parse_tile_dict(index: usize, tile: &Bound<'_, PyAny>) -> PyResult<ContributingTile> {
    let get_u32 = |key: &str| -> PyResult<u32> {
        tile.get_item(key)
            .map_err(|_| {
                PyValueError::new_err(format!("contributing tile {index} missing '{key}'"))
            })?
            .extract::<u32>()
            .map_err(|_| {
                PyValueError::new_err(format!(
                    "contributing tile {index}: '{key}' must be a non-negative integer"
                ))
            })
    };
    let content_hash_hex: String = tile
        .get_item("content_hash")
        .map_err(|_| {
            PyValueError::new_err(format!("contributing tile {index} missing 'content_hash'"))
        })?
        .extract()
        .map_err(|_| {
            PyValueError::new_err(format!(
                "contributing tile {index}: 'content_hash' must be a hex string"
            ))
        })?;
    let content_hash_bytes = from_hex(&content_hash_hex).map_err(|e| {
        PyValueError::new_err(format!("contributing tile {index}: bad content_hash: {e}"))
    })?;
    let content_hash: [u8; 32] = content_hash_bytes.as_slice().try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "contributing tile {index}: content_hash must be 32 bytes"
        ))
    })?;
    Ok(ContributingTile {
        family_slot: get_u32("family_slot")?,
        source_id: get_u32("source_id")?,
        tile_x: get_u32("tile_x")?,
        tile_y: get_u32("tile_y")?,
        mip_level: get_u32("mip_level")?,
        content_hash,
    })
}

/// Seal a rendered frame's provenance.
///
/// Builds the sorted SHA256 Merkle tree over the contributing-tile leaves
/// plus one source-map leaf, signs the root with the Ed25519 32-byte seed
/// `private_key`, and returns the `provenance.json` manifest bytes.
///
/// Parameters
/// ----------
/// source_map : numpy.ndarray
///     `(H, W)` uint32 per-pixel source-id map (from `AovFrame.source_id()`).
/// contributing_tiles : list[dict]
///     Records from `TerrainRenderer.read_contributing_tiles()`.
/// private_key : bytes
///     32-byte Ed25519 seed.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (source_map, contributing_tiles, private_key))]
pub(crate) fn seal_provenance(
    py: Python<'_>,
    source_map: numpy::PyReadonlyArray2<'_, u32>,
    contributing_tiles: &Bound<'_, PyAny>,
    private_key: Vec<u8>,
) -> PyResult<Py<PyAny>> {
    use pyo3::types::PyBytes;

    let seed: [u8; 32] = private_key
        .as_slice()
        .try_into()
        .map_err(|_| PyValueError::new_err("private_key must be a 32-byte Ed25519 seed"))?;

    let shape = source_map.shape();
    let (height, width) = (shape[0] as u32, shape[1] as u32);
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("source_map must be non-empty"));
    }

    let mut tiles = Vec::new();
    for (index, tile) in contributing_tiles.iter()?.enumerate() {
        tiles.push(parse_tile_dict(index, &tile?)?);
    }
    tiles.sort_by_key(|t| encode_tile_leaf(t));
    tiles.dedup();

    let map_digest = source_map_digest(&source_map);
    let mut leaves: Vec<Vec<u8>> = tiles.iter().map(|t| encode_tile_leaf(t).to_vec()).collect();
    leaves.push(encode_source_map_leaf(width, height, &map_digest).to_vec());
    let root = merkle_root(&leaves);
    let (signature, public_key) = sign_root(&root, &seed);

    // source_table: unique (source_id, family, content_hash) triples.
    let mut source_table: Vec<(u32, u32, [u8; 32])> = tiles
        .iter()
        .map(|t| (t.source_id, t.family_slot, t.content_hash))
        .collect();
    source_table.sort();
    source_table.dedup();

    let family_name = |slot: u32| {
        FAMILY_NAMES
            .get(slot as usize)
            .copied()
            .unwrap_or("unknown")
    };
    let manifest = serde_json::json!({
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "merkle_root": to_hex(&root),
        "signature": to_hex(&signature),
        "public_key": to_hex(&public_key),
        "image_dims": [width, height],
        "albedo_family_index": 0,
        "source_map_encoding": "u32le-row-major",
        "source_map_sha256": to_hex(&map_digest),
        "leaves": tiles
            .iter()
            .map(|t| serde_json::json!({
                "family": family_name(t.family_slot),
                "family_slot": t.family_slot,
                "source_id": t.source_id,
                "tile_x": t.tile_x,
                "tile_y": t.tile_y,
                "mip_level": t.mip_level,
                "content_hash": to_hex(&t.content_hash),
            }))
            .collect::<Vec<_>>(),
        "source_table": source_table
            .iter()
            .map(|(source_id, family_slot, content_hash)| serde_json::json!({
                "source_id": source_id,
                "family": family_name(*family_slot),
                "content_hash": to_hex(content_hash),
            }))
            .collect::<Vec<_>>(),
    });
    let bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to serialize manifest: {e}")))?;
    Ok(PyBytes::new_bound(py, &bytes).into_py(py))
}

/// Verify a provenance manifest against a source map.
///
/// Rebuilds the Merkle root from the manifest's leaf records plus the
/// source-map leaf recomputed from `source_map`, and checks it matches the
/// signed root and that the Ed25519 signature verifies against the embedded
/// public key. Returns `False` on any mismatch; raises `ValueError` only for
/// a structurally malformed manifest.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (source_map, manifest))]
pub(crate) fn verify_provenance(
    source_map: numpy::PyReadonlyArray2<'_, u32>,
    manifest: Vec<u8>,
) -> PyResult<bool> {
    let manifest: serde_json::Value = serde_json::from_slice(&manifest)
        .map_err(|e| PyValueError::new_err(format!("malformed provenance manifest: {e}")))?;

    let schema_version = manifest
        .get("schema_version")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| PyValueError::new_err("manifest missing schema_version"))?;
    if schema_version != PROVENANCE_SCHEMA_VERSION {
        return Err(PyValueError::new_err(format!(
            "unsupported provenance schema_version {schema_version} (expected {PROVENANCE_SCHEMA_VERSION})"
        )));
    }

    let hex_field = |key: &str, expected_len: usize| -> PyResult<Vec<u8>> {
        let text = manifest
            .get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| PyValueError::new_err(format!("manifest missing '{key}'")))?;
        let bytes =
            from_hex(text).map_err(|e| PyValueError::new_err(format!("bad '{key}': {e}")))?;
        if bytes.len() != expected_len {
            return Err(PyValueError::new_err(format!(
                "'{key}' must be {expected_len} bytes"
            )));
        }
        Ok(bytes)
    };
    let signed_root: [u8; 32] = hex_field("merkle_root", 32)?.try_into().unwrap();
    let signature: [u8; 64] = hex_field("signature", 64)?.try_into().unwrap();
    let public_key: [u8; 32] = hex_field("public_key", 32)?.try_into().unwrap();

    let dims = manifest
        .get("image_dims")
        .and_then(|v| v.as_array())
        .ok_or_else(|| PyValueError::new_err("manifest missing image_dims"))?;
    if dims.len() != 2 {
        return Err(PyValueError::new_err("image_dims must be [width, height]"));
    }
    let manifest_width = dims[0].as_u64().unwrap_or(0) as u32;
    let manifest_height = dims[1].as_u64().unwrap_or(0) as u32;

    let shape = source_map.shape();
    let (height, width) = (shape[0] as u32, shape[1] as u32);
    if (manifest_width, manifest_height) != (width, height) {
        return Ok(false);
    }

    let leaves_json = manifest
        .get("leaves")
        .and_then(|v| v.as_array())
        .ok_or_else(|| PyValueError::new_err("manifest missing leaves"))?;
    let mut leaves: Vec<Vec<u8>> = Vec::with_capacity(leaves_json.len() + 1);
    for (index, leaf) in leaves_json.iter().enumerate() {
        let get_u32 = |key: &str| -> PyResult<u32> {
            leaf.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .ok_or_else(|| PyValueError::new_err(format!("leaf {index} missing '{key}'")))
        };
        let content_hash_text = leaf
            .get("content_hash")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PyValueError::new_err(format!("leaf {index} missing content_hash")))?;
        let content_hash: [u8; 32] = from_hex(content_hash_text)
            .map_err(|e| PyValueError::new_err(format!("leaf {index}: {e}")))?
            .try_into()
            .map_err(|_| {
                PyValueError::new_err(format!("leaf {index}: content_hash must be 32 bytes"))
            })?;
        let tile = ContributingTile {
            family_slot: get_u32("family_slot")?,
            source_id: get_u32("source_id")?,
            tile_x: get_u32("tile_x")?,
            tile_y: get_u32("tile_y")?,
            mip_level: get_u32("mip_level")?,
            content_hash,
        };
        leaves.push(encode_tile_leaf(&tile).to_vec());
    }
    let map_digest = source_map_digest(&source_map);
    leaves.push(encode_source_map_leaf(width, height, &map_digest).to_vec());

    let computed_root = merkle_root(&leaves);
    if computed_root != signed_root {
        return Ok(false);
    }
    // Belt-and-braces: the manifest's own source_map_sha256 must also match.
    if let Some(recorded) = manifest.get("source_map_sha256").and_then(|v| v.as_str()) {
        if !recorded.eq_ignore_ascii_case(&to_hex(&map_digest)) {
            return Ok(false);
        }
    }
    Ok(verify_root(&computed_root, &signature, &public_key))
}
