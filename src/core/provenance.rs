//! VERITAS: per-pixel source provenance commitment.
//!
//! Canonical leaf encodings, a binary SHA256 Merkle tree over the tiles that
//! contributed to a rendered frame, and the Ed25519 seal over the root.
//!
//! Encoding contract (mirrored byte-for-byte by `python/forge3d/provenance.py`
//! and `tools/verify_provenance.py`; changing any rule here is a schema bump):
//! - Tile leaf (56 bytes): `b"VTLF" || family_slot:u32le || source_id:u32le ||
//!   tile_x:u32le || tile_y:u32le || mip_level:u32le || content_hash[32]`.
//! - Source-map leaf (44 bytes): `b"VTSM" || width:u32le || height:u32le ||
//!   sha256(row-major little-endian u32 raster)[32]`.
//! - Leaf hash = SHA256(encoding); leaves sorted ascending by raw encoding
//!   bytes so async tile-arrival order cannot change the root.
//! - Interior node = SHA256(left || right); an odd trailing node is promoted
//!   unchanged to the next level.
//! - Empty leaf set: root = SHA256(b"forge3d.provenance.v1.empty").
//! - Signature message = `b"forge3d.provenance.v1" || root[32]` (Ed25519).
//!
//! Kept free of wgpu/PyO3 so the unit tests run under the curated cargo
//! feature set (which excludes `extension-module`).

use sha2::{Digest, Sha256};

/// Sentinel source id: "no source dataset attributed" (fallback / background).
pub const SOURCE_ID_NONE: u32 = 0;

/// Fixed per-family material capacity used to derive stable source ids.
/// Matches `MATERIAL_LAYER_CAPACITY` / `TERRAIN_VT_MATERIAL_CAPACITY` (4).
pub const SOURCE_ID_MATERIAL_CAPACITY: u32 = 4;

/// VT family names indexed by family slot.
pub const FAMILY_NAMES: [&str; 3] = ["albedo", "normal", "mask"];

/// Domain-separation prefix for the signed Merkle root.
pub const SIGN_CONTEXT: &[u8] = b"forge3d.provenance.v1";

const EMPTY_ROOT_PREIMAGE: &[u8] = b"forge3d.provenance.v1.empty";
const TILE_LEAF_TAG: &[u8; 4] = b"VTLF";
const SOURCE_MAP_LEAF_TAG: &[u8; 4] = b"VTSM";

/// One deduplicated tile that was resident and sampled for a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ContributingTile {
    pub family_slot: u32,
    pub source_id: u32,
    pub tile_x: u32,
    pub tile_y: u32,
    pub mip_level: u32,
    pub content_hash: [u8; 32],
}

/// Stable, device-independent source id for a registered VT source.
/// `0` is reserved for [`SOURCE_ID_NONE`].
pub fn source_id_for(family_slot: u32, material_index: u32) -> u32 {
    family_slot * SOURCE_ID_MATERIAL_CAPACITY + material_index + 1
}

/// SHA256 convenience over a byte slice.
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// Canonical 56-byte tile-leaf preimage.
pub fn encode_tile_leaf(tile: &ContributingTile) -> [u8; 56] {
    let mut out = [0u8; 56];
    out[0..4].copy_from_slice(TILE_LEAF_TAG);
    out[4..8].copy_from_slice(&tile.family_slot.to_le_bytes());
    out[8..12].copy_from_slice(&tile.source_id.to_le_bytes());
    out[12..16].copy_from_slice(&tile.tile_x.to_le_bytes());
    out[16..20].copy_from_slice(&tile.tile_y.to_le_bytes());
    out[20..24].copy_from_slice(&tile.mip_level.to_le_bytes());
    out[24..56].copy_from_slice(&tile.content_hash);
    out
}

/// Canonical 44-byte source-map-leaf preimage. `digest` is the SHA256 of the
/// row-major little-endian u32 raster.
pub fn encode_source_map_leaf(width: u32, height: u32, digest: &[u8; 32]) -> [u8; 44] {
    let mut out = [0u8; 44];
    out[0..4].copy_from_slice(SOURCE_MAP_LEAF_TAG);
    out[4..8].copy_from_slice(&width.to_le_bytes());
    out[8..12].copy_from_slice(&height.to_le_bytes());
    out[12..44].copy_from_slice(digest);
    out
}

/// Binary SHA256 Merkle root over the given leaf preimages. Leaves are sorted
/// by their raw encoding, so the root is independent of arrival order.
pub fn merkle_root(leaf_encodings: &[Vec<u8>]) -> [u8; 32] {
    if leaf_encodings.is_empty() {
        return sha256(EMPTY_ROOT_PREIMAGE);
    }
    let mut sorted: Vec<&Vec<u8>> = leaf_encodings.iter().collect();
    sorted.sort();
    let mut level: Vec<[u8; 32]> = sorted.into_iter().map(|leaf| sha256(leaf)).collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut chunks = level.chunks_exact(2);
        for pair in &mut chunks {
            let mut preimage = [0u8; 64];
            preimage[..32].copy_from_slice(&pair[0]);
            preimage[32..].copy_from_slice(&pair[1]);
            next.push(sha256(&preimage));
        }
        // Odd node count: promote the lone node unchanged.
        if let Some(last) = chunks.remainder().first() {
            next.push(*last);
        }
        level = next;
    }
    level[0]
}

/// Sign a Merkle root with an Ed25519 32-byte seed. Returns
/// `(signature, public_key)`.
pub fn sign_root(root: &[u8; 32], seed: &[u8; 32]) -> ([u8; 64], [u8; 32]) {
    use ed25519_dalek::{Signer, SigningKey};
    let signing_key = SigningKey::from_bytes(seed);
    let mut message = Vec::with_capacity(SIGN_CONTEXT.len() + 32);
    message.extend_from_slice(SIGN_CONTEXT);
    message.extend_from_slice(root);
    let signature = signing_key.sign(&message);
    (signature.to_bytes(), signing_key.verifying_key().to_bytes())
}

/// Verify an Ed25519 seal over a Merkle root.
pub fn verify_root(root: &[u8; 32], signature: &[u8; 64], public_key: &[u8; 32]) -> bool {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};
    let Ok(key) = VerifyingKey::from_bytes(public_key) else {
        return false;
    };
    let signature = Signature::from_bytes(signature);
    let mut message = Vec::with_capacity(SIGN_CONTEXT.len() + 32);
    message.extend_from_slice(SIGN_CONTEXT);
    message.extend_from_slice(root);
    key.verify(&message, &signature).is_ok()
}

/// Hex-encode bytes (lowercase), matching `format!("{:02x}")` semantics used
/// elsewhere in the crate (no `hex` dependency).
pub fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Decode a lowercase/uppercase hex string into bytes.
pub fn from_hex(text: &str) -> Result<Vec<u8>, String> {
    if !text.len().is_multiple_of(2) {
        return Err("hex string has odd length".to_string());
    }
    (0..text.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&text[i..i + 2], 16)
                .map_err(|_| format!("invalid hex byte at offset {i}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(family_slot: u32, material_index: u32, x: u32, y: u32, mip: u32) -> ContributingTile {
        ContributingTile {
            family_slot,
            source_id: source_id_for(family_slot, material_index),
            tile_x: x,
            tile_y: y,
            mip_level: mip,
            content_hash: sha256(&[family_slot as u8, material_index as u8]),
        }
    }

    #[test]
    fn source_ids_are_dense_and_reserve_zero() {
        assert_eq!(source_id_for(0, 0), 1);
        assert_eq!(source_id_for(0, 3), 4);
        assert_eq!(source_id_for(1, 0), 5);
        assert_eq!(source_id_for(2, 3), 12);
        assert_ne!(source_id_for(0, 0), SOURCE_ID_NONE);
    }

    #[test]
    fn leaf_encodings_have_fixed_sizes_and_tags() {
        let t = tile(0, 1, 3, 4, 2);
        let leaf = encode_tile_leaf(&t);
        assert_eq!(leaf.len(), 56);
        assert_eq!(&leaf[0..4], b"VTLF");
        assert_eq!(&leaf[4..8], &0u32.to_le_bytes());
        assert_eq!(&leaf[8..12], &2u32.to_le_bytes());

        let sm = encode_source_map_leaf(640, 480, &sha256(b"map"));
        assert_eq!(sm.len(), 44);
        assert_eq!(&sm[0..4], b"VTSM");
    }

    #[test]
    fn merkle_root_is_order_independent() {
        let leaves: Vec<Vec<u8>> = (0..7u32)
            .map(|i| encode_tile_leaf(&tile(0, i % 4, i, i * 2, 0)).to_vec())
            .collect();
        let mut shuffled = leaves.clone();
        shuffled.reverse();
        shuffled.swap(0, 3);
        assert_eq!(merkle_root(&leaves), merkle_root(&shuffled));
    }

    #[test]
    fn merkle_odd_count_promotes_lone_node() {
        let leaves: Vec<Vec<u8>> = (0..3u32)
            .map(|i| encode_tile_leaf(&tile(0, i, i, 0, 0)).to_vec())
            .collect();
        let mut sorted = leaves.clone();
        sorted.sort();
        let h: Vec<[u8; 32]> = sorted.iter().map(|l| sha256(l)).collect();
        let mut pair = [0u8; 64];
        pair[..32].copy_from_slice(&h[0]);
        pair[32..].copy_from_slice(&h[1]);
        let left = sha256(&pair);
        // Level 1 = [H(h0||h1), h2 promoted]; root = H(level1[0] || h2).
        let mut root_pre = [0u8; 64];
        root_pre[..32].copy_from_slice(&left);
        root_pre[32..].copy_from_slice(&h[2]);
        assert_eq!(merkle_root(&leaves), sha256(&root_pre));
    }

    #[test]
    fn merkle_empty_uses_documented_sentinel() {
        assert_eq!(merkle_root(&[]), sha256(b"forge3d.provenance.v1.empty"));
    }

    #[test]
    fn single_leaf_root_is_its_hash() {
        let leaf = encode_tile_leaf(&tile(0, 0, 0, 0, 0)).to_vec();
        assert_eq!(merkle_root(std::slice::from_ref(&leaf)), sha256(&leaf));
    }

    #[test]
    fn tamper_in_any_leaf_changes_root() {
        let leaves: Vec<Vec<u8>> = (0..4u32)
            .map(|i| encode_tile_leaf(&tile(0, i, i, 0, 0)).to_vec())
            .collect();
        let root = merkle_root(&leaves);
        let mut tampered = leaves.clone();
        tampered[2][30] ^= 0x01; // flip one content-hash byte
        assert_ne!(root, merkle_root(&tampered));
    }

    #[test]
    fn sign_verify_round_trip_and_rejection() {
        let seed = sha256(b"forge3d-veritas-unit-seed");
        let root = merkle_root(&[encode_tile_leaf(&tile(0, 0, 0, 0, 0)).to_vec()]);
        let (signature, public_key) = sign_root(&root, &seed);
        assert!(verify_root(&root, &signature, &public_key));

        let mut wrong_root = root;
        wrong_root[0] ^= 0xFF;
        assert!(!verify_root(&wrong_root, &signature, &public_key));

        let other_seed = sha256(b"a-different-seed");
        let (_, other_public) = sign_root(&root, &other_seed);
        assert!(!verify_root(&root, &signature, &other_public));
    }

    /// Known-answer vector; `tests/test_provenance_veritas.py` asserts the
    /// same root from the pure-Python implementation.
    #[test]
    fn known_answer_root_vector() {
        let t0 = ContributingTile {
            family_slot: 0,
            source_id: 1,
            tile_x: 0,
            tile_y: 0,
            mip_level: 0,
            content_hash: sha256(b"source-a"),
        };
        let t1 = ContributingTile {
            family_slot: 0,
            source_id: 2,
            tile_x: 1,
            tile_y: 0,
            mip_level: 1,
            content_hash: sha256(b"source-b"),
        };
        let sm = encode_source_map_leaf(4, 2, &sha256(b"source-map"));
        let leaves = vec![
            encode_tile_leaf(&t0).to_vec(),
            encode_tile_leaf(&t1).to_vec(),
            sm.to_vec(),
        ];
        let root = merkle_root(&leaves);
        assert_eq!(
            to_hex(&root),
            "67e632b879b8d0f52360148abad03584b213f9065714e2b722db766e03e980c4"
        );
    }

    #[test]
    fn hex_round_trip() {
        let bytes = sha256(b"hex");
        let text = to_hex(&bytes);
        assert_eq!(from_hex(&text).unwrap(), bytes.to_vec());
        assert!(from_hex("zz").is_err());
        assert!(from_hex("abc").is_err());
    }
}
