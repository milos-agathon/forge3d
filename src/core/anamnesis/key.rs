//! Hermetic content keys for ANAMNESIS render passes.
//!
//! The key boundary is deliberately byte-oriented. Callers must pass the exact
//! bytes uploaded to the GPU (including deterministically initialized padding)
//! and a canonical descriptor blob containing every item of dynamic render
//! state. Unknown state is never defaulted here: the caller must either encode
//! it or decline to cache the pass.

use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const DOMAIN: &[u8] = b"forge3d.anamnesis/1";
const LEAF_DOMAIN: &[u8] = b"forge3d.anamnesis/1/leaf";

/// SHA-256 identity of a pass output or leaf resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PassKey(pub [u8; 32]);

impl Serialize for PassKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for PassKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::from_hex(&value).map_err(D::Error::custom)
    }
}

impl PassKey {
    pub fn from_hex(value: &str) -> Result<Self, String> {
        if value.len() != 64 {
            return Err("ANAMNESIS key must contain exactly 64 hexadecimal characters".into());
        }
        let mut out = [0u8; 32];
        for (index, byte) in out.iter_mut().enumerate() {
            let offset = index * 2;
            *byte = u8::from_str_radix(&value[offset..offset + 2], 16)
                .map_err(|_| "ANAMNESIS key contains non-hexadecimal characters")?;
        }
        Ok(Self(out))
    }

    pub fn to_hex(self) -> String {
        crate::core::provenance::to_hex(&self.0)
    }
}

impl std::fmt::Display for PassKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// Engine inputs that can change shader translation or engine behaviour.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineFingerprint {
    pub crate_version: String,
    pub git_sha: String,
    pub naga_version: String,
    pub wgsl_tree_sha256: String,
}

impl EngineFingerprint {
    pub fn current() -> Self {
        Self {
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            git_sha: env!("FORGE3D_GIT_SHA").to_string(),
            // Cargo.lock pins this version. It is intentionally explicit: an
            // engine upgrade must review and update the cache schema surface.
            naga_version: "0.19.2".to_string(),
            wgsl_tree_sha256: env!("FORGE3D_WGSL_TREE_SHA256").to_string(),
        }
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("EngineFingerprint is infallibly serializable")
    }
}

/// Negotiated code-generation inputs supplied by CENSOR.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityFingerprint {
    pub granted_features: Vec<String>,
    pub limits: BTreeMap<String, u64>,
    pub backend: String,
    pub dx12_compiler: String,
    pub naga_capabilities: Vec<String>,
}

impl CapabilityFingerprint {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut canonical = self.clone();
        canonical.granted_features.sort();
        canonical.granted_features.dedup();
        canonical.naga_capabilities.sort();
        canonical.naga_capabilities.dedup();
        serde_json::to_vec(&canonical).expect("CapabilityFingerprint is infallibly serializable")
    }
}

/// Self-describing derivation retained in each store entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassKeyMaterial {
    pub label: String,
    pub pipeline_descriptor_hash: PassKey,
    pub uniform_sha256: PassKey,
    pub uniform_byte_length: u64,
    pub input_keys: Vec<PassKey>,
    pub capability_fingerprint_sha256: PassKey,
    pub engine_fingerprint_sha256: PassKey,
}

fn add_segment(hasher: &mut Sha256, tag: &[u8], bytes: &[u8]) {
    hasher.update((tag.len() as u64).to_le_bytes());
    hasher.update(tag);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

pub fn sha256(bytes: &[u8]) -> PassKey {
    PassKey(Sha256::digest(bytes).into())
}

/// Hash a source resource such as a DEM tile, texture, or compiled label plan.
pub fn leaf_key(content: &[u8]) -> PassKey {
    let mut hasher = Sha256::new();
    add_segment(&mut hasher, b"domain", LEAF_DOMAIN);
    add_segment(&mut hasher, b"content", content);
    PassKey(hasher.finalize().into())
}

/// Compute the pass key and the derivation record stored beside its blob.
pub fn pass_key(
    label: &str,
    pipeline_descriptor_bytes: &[u8],
    uniform_bytes: &[u8],
    input_keys: &[PassKey],
    capability_fingerprint_bytes: &[u8],
    engine_fingerprint_bytes: &[u8],
) -> (PassKey, PassKeyMaterial) {
    let pipeline_descriptor_hash = sha256(pipeline_descriptor_bytes);
    let uniform_sha256 = sha256(uniform_bytes);
    let capability_fingerprint_sha256 = sha256(capability_fingerprint_bytes);
    let engine_fingerprint_sha256 = sha256(engine_fingerprint_bytes);
    let mut sorted_inputs = input_keys.to_vec();
    sorted_inputs.sort_unstable();

    let mut hasher = Sha256::new();
    add_segment(&mut hasher, b"domain", DOMAIN);
    add_segment(&mut hasher, b"label", label.as_bytes());
    add_segment(
        &mut hasher,
        b"pipeline_descriptor_hash",
        &pipeline_descriptor_hash.0,
    );
    // Hash the exact bytes, not a Rust struct or a logical field projection.
    add_segment(&mut hasher, b"uniform_bytes", uniform_bytes);
    for input in &sorted_inputs {
        add_segment(&mut hasher, b"input_key", &input.0);
    }
    add_segment(
        &mut hasher,
        b"capability_fingerprint",
        capability_fingerprint_bytes,
    );
    add_segment(&mut hasher, b"engine_fingerprint", engine_fingerprint_bytes);
    let key = PassKey(hasher.finalize().into());
    (
        key,
        PassKeyMaterial {
            label: label.to_string(),
            pipeline_descriptor_hash,
            uniform_sha256,
            uniform_byte_length: uniform_bytes.len() as u64,
            input_keys: sorted_inputs,
            capability_fingerprint_sha256,
            engine_fingerprint_sha256,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_with(pipeline: &[u8], uniform: &[u8], capability: &[u8]) -> PassKey {
        pass_key(
            "terrain.forward",
            pipeline,
            uniform,
            &[],
            capability,
            &EngineFingerprint::current().canonical_bytes(),
        )
        .0
    }

    #[test]
    fn adversarial_pixel_inputs_never_alias() {
        assert_ne!(
            key_with(b"sampler=nearest", b"u", b"naga=all"),
            key_with(b"sampler=linear", b"u", b"naga=all")
        );
        assert_ne!(
            key_with(b"blend=replace", b"u", b"naga=all"),
            key_with(b"blend=alpha", b"u", b"naga=all")
        );
        assert_ne!(
            key_with(b"depth=less", b"u", b"naga=all"),
            key_with(b"depth=less_equal", b"u", b"naga=all")
        );
        assert_ne!(
            key_with(b"wgsl=same", b"u", b"naga=all"),
            key_with(b"wgsl=same", b"u", b"naga=restricted")
        );
    }

    #[test]
    fn uniform_padding_is_part_of_the_identity() {
        let mut zeroed = vec![0u8; 16];
        zeroed[..4].copy_from_slice(&1u32.to_le_bytes());
        let mut different_padding = zeroed.clone();
        different_padding[12] = 1;
        assert_ne!(
            key_with(b"p", &zeroed, b"c"),
            key_with(b"p", &different_padding, b"c")
        );
    }

    #[test]
    fn input_order_is_canonical() {
        let a = leaf_key(b"a");
        let b = leaf_key(b"b");
        let engine = EngineFingerprint::current().canonical_bytes();
        let left = pass_key("x", b"p", b"u", &[a, b], b"c", &engine).0;
        let right = pass_key("x", b"p", b"u", &[b, a], b"c", &engine).0;
        assert_eq!(left, right);
    }
}
