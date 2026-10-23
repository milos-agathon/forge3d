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
        // Stored metadata is untrusted UTF-8. Check the complete alphabet
        // before byte slicing: a 64-byte Unicode string can split a code point.
        if !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("ANAMNESIS key contains non-hexadecimal characters".into());
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
            git_sha: env!("FORGE3D_GIT_SHA_FULL").to_string(),
            naga_version: env!("FORGE3D_NAGA_VERSION").to_string(),
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
        let value = serde_json::to_value(&canonical)
            .expect("CapabilityFingerprint is infallibly serializable");
        serde_json::to_vec(&value).expect("CapabilityFingerprint is infallibly serializable")
    }
}

/// Keep every negotiated limit in cache identity. The exhaustive pattern
/// intentionally stops compiling when wgpu adds another limit.
pub(crate) fn device_limits(limits: &wgpu::Limits) -> BTreeMap<String, u64> {
    macro_rules! capture {
        ($($name:ident),+ $(,)?) => {{
            let wgpu::Limits { $($name),+ } = *limits;
            BTreeMap::from([$((stringify!($name).to_string(), u64::from($name))),+])
        }};
    }
    capture!(
        max_texture_dimension_1d,
        max_texture_dimension_2d,
        max_texture_dimension_3d,
        max_texture_array_layers,
        max_bind_groups,
        max_bindings_per_bind_group,
        max_dynamic_uniform_buffers_per_pipeline_layout,
        max_dynamic_storage_buffers_per_pipeline_layout,
        max_sampled_textures_per_shader_stage,
        max_samplers_per_shader_stage,
        max_storage_buffers_per_shader_stage,
        max_storage_textures_per_shader_stage,
        max_uniform_buffers_per_shader_stage,
        max_uniform_buffer_binding_size,
        max_storage_buffer_binding_size,
        max_vertex_buffers,
        max_buffer_size,
        max_vertex_attributes,
        max_vertex_buffer_array_stride,
        min_uniform_buffer_offset_alignment,
        min_storage_buffer_offset_alignment,
        max_inter_stage_shader_components,
        max_compute_workgroup_storage_size,
        max_compute_invocations_per_workgroup,
        max_compute_workgroup_size_x,
        max_compute_workgroup_size_y,
        max_compute_workgroup_size_z,
        max_compute_workgroups_per_dimension,
        max_push_constant_size,
        max_non_sampler_bindings,
    )
}

pub(crate) fn naga_policy() -> Vec<String> {
    vec![
        format!(
            "wgpu-validation-default@naga-{}",
            env!("FORGE3D_NAGA_VERSION")
        ),
        format!("deterministic={}", crate::core::gpu::deterministic_mode()),
    ]
}

/// Self-describing derivation retained in each store entry.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct InputKey {
    /// Stable semantic binding identity (for example `heightmap@0`).
    ///
    /// The binding is part of the pass identity. Sorting bare hashes is
    /// unsound because swapping two same-shaped resources would preserve the
    /// multiset while changing the pixels.
    pub binding: String,
    pub key: PassKey,
}

impl InputKey {
    pub fn new(binding: impl Into<String>, key: PassKey) -> Self {
        Self {
            binding: binding.into(),
            key,
        }
    }
}

/// Self-describing derivation retained in each store entry.
///
/// Exact canonical input bytes are stored alongside their hashes so
/// `explain` and `verify` can reconstruct the key rather than merely display
/// hashes that cannot be audited.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassKeyMaterial {
    pub label: String,
    pub pipeline_descriptor_hash: PassKey,
    pub pipeline_descriptor_hex: String,
    pub uniform_sha256: PassKey,
    pub uniform_byte_length: u64,
    pub uniform_hex: String,
    pub input_keys: Vec<InputKey>,
    pub capability_fingerprint_sha256: PassKey,
    pub capability_fingerprint_hex: String,
    pub engine_fingerprint_sha256: PassKey,
    pub engine_fingerprint_hex: String,
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
    input_keys: &[InputKey],
    capability_fingerprint_bytes: &[u8],
    engine_fingerprint_bytes: &[u8],
) -> (PassKey, PassKeyMaterial) {
    let pipeline_descriptor_hash = sha256(pipeline_descriptor_bytes);
    let uniform_sha256 = sha256(uniform_bytes);
    let capability_fingerprint_sha256 = sha256(capability_fingerprint_bytes);
    let engine_fingerprint_sha256 = sha256(engine_fingerprint_bytes);
    let mut sorted_inputs = input_keys.to_vec();
    sorted_inputs.sort_unstable();

    let key = hash_pass_key(
        label,
        pipeline_descriptor_hash,
        uniform_bytes,
        &sorted_inputs,
        capability_fingerprint_bytes,
        engine_fingerprint_bytes,
    );
    (
        key,
        PassKeyMaterial {
            label: label.to_string(),
            pipeline_descriptor_hash,
            pipeline_descriptor_hex: crate::core::provenance::to_hex(pipeline_descriptor_bytes),
            uniform_sha256,
            uniform_byte_length: uniform_bytes.len() as u64,
            uniform_hex: crate::core::provenance::to_hex(uniform_bytes),
            input_keys: sorted_inputs,
            capability_fingerprint_sha256,
            capability_fingerprint_hex: crate::core::provenance::to_hex(
                capability_fingerprint_bytes,
            ),
            engine_fingerprint_sha256,
            engine_fingerprint_hex: crate::core::provenance::to_hex(engine_fingerprint_bytes),
        },
    )
}

// The Python key API needs only the hash, not the large reconstruction record.
#[cfg(any(feature = "extension-module", test))]
pub(crate) fn pass_key_only(
    label: &str,
    pipeline_descriptor_bytes: &[u8],
    uniform_bytes: &[u8],
    input_keys: &[InputKey],
    capability_fingerprint_bytes: &[u8],
    engine_fingerprint_bytes: &[u8],
) -> PassKey {
    let mut sorted_inputs = input_keys.to_vec();
    sorted_inputs.sort_unstable();
    hash_pass_key(
        label,
        sha256(pipeline_descriptor_bytes),
        uniform_bytes,
        &sorted_inputs,
        capability_fingerprint_bytes,
        engine_fingerprint_bytes,
    )
}

fn hash_pass_key(
    label: &str,
    pipeline_descriptor_hash: PassKey,
    uniform_bytes: &[u8],
    sorted_inputs: &[InputKey],
    capability_fingerprint_bytes: &[u8],
    engine_fingerprint_bytes: &[u8],
) -> PassKey {
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
    for input in sorted_inputs {
        add_segment(&mut hasher, b"input_binding", input.binding.as_bytes());
        add_segment(&mut hasher, b"input_key", &input.key.0);
    }
    add_segment(
        &mut hasher,
        b"capability_fingerprint",
        capability_fingerprint_bytes,
    );
    add_segment(&mut hasher, b"engine_fingerprint", engine_fingerprint_bytes);
    PassKey(hasher.finalize().into())
}

/// Recompute a stored derivation. This is the rejecting-power primitive used
/// by store verification and `explain`.
pub fn reconstruct_pass_key(material: &PassKeyMaterial) -> Result<PassKey, String> {
    fn decode(name: &str, value: &str) -> Result<Vec<u8>, String> {
        fn nibble(byte: u8) -> Option<u8> {
            match byte {
                b'0'..=b'9' => Some(byte - b'0'),
                b'a'..=b'f' => Some(byte - b'a' + 10),
                _ => None,
            }
        }
        if !value.len().is_multiple_of(2) {
            return Err(format!("{name} contains an odd number of hex characters"));
        }
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| match (nibble(pair[0]), nibble(pair[1])) {
                (Some(high), Some(low)) => Ok(high * 16 + low),
                _ => Err(format!("{name} is not canonical lowercase hex")),
            })
            .collect()
    }
    let pipeline = decode("pipeline_descriptor_hex", &material.pipeline_descriptor_hex)?;
    let uniforms = decode("uniform_hex", &material.uniform_hex)?;
    let capabilities = decode(
        "capability_fingerprint_hex",
        &material.capability_fingerprint_hex,
    )?;
    let engine = decode("engine_fingerprint_hex", &material.engine_fingerprint_hex)?;
    // Preserve the full reconstruction comparison without allocating a second
    // metadata record or re-encoding every canonical byte as hexadecimal.
    if material.pipeline_descriptor_hash != sha256(&pipeline)
        || material.uniform_sha256 != sha256(&uniforms)
        || material.uniform_byte_length != uniforms.len() as u64
        || material.capability_fingerprint_sha256 != sha256(&capabilities)
        || material.engine_fingerprint_sha256 != sha256(&engine)
        || material.input_keys.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err("ANAMNESIS derivation hashes do not match stored canonical bytes".into());
    }
    Ok(hash_pass_key(
        &material.label,
        material.pipeline_descriptor_hash,
        &uniforms,
        &material.input_keys,
        &capabilities,
        &engine,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_only_path_preserves_exact_keys() {
        let inputs = [
            InputKey::new("z@1", leaf_key(b"z")),
            InputKey::new("a@0", leaf_key(b"a")),
        ];
        for label in ["terrain.render", "frame.output"] {
            assert_eq!(
                pass_key_only(
                    label,
                    b"pipeline",
                    b"uniform\0padding",
                    &inputs,
                    b"caps",
                    b"engine"
                ),
                pass_key(
                    label,
                    b"pipeline",
                    b"uniform\0padding",
                    &inputs,
                    b"caps",
                    b"engine"
                )
                .0,
            );
        }
    }

    #[test]
    fn negotiated_texture_and_storage_limits_change_identity() {
        let baseline = wgpu::Limits::downlevel_defaults();
        let mut changed = baseline.clone();
        changed.max_texture_array_layers += 1;
        assert_ne!(device_limits(&baseline), device_limits(&changed));
        changed = baseline.clone();
        changed.max_storage_buffer_binding_size += 1;
        assert_ne!(device_limits(&baseline), device_limits(&changed));
    }

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
        let left = pass_key(
            "x",
            b"p",
            b"u",
            &[InputKey::new("a", a), InputKey::new("b", b)],
            b"c",
            &engine,
        )
        .0;
        let right = pass_key(
            "x",
            b"p",
            b"u",
            &[InputKey::new("b", b), InputKey::new("a", a)],
            b"c",
            &engine,
        )
        .0;
        assert_eq!(left, right);
    }

    #[test]
    fn semantic_role_swap_never_aliases() {
        let same_shape_a = leaf_key(b"aaaa");
        let same_shape_b = leaf_key(b"bbbb");
        let engine = EngineFingerprint::current().canonical_bytes();
        let left = pass_key(
            "x",
            b"p",
            b"u",
            &[
                InputKey::new("height@0", same_shape_a),
                InputKey::new("water@1", same_shape_b),
            ],
            b"c",
            &engine,
        )
        .0;
        let right = pass_key(
            "x",
            b"p",
            b"u",
            &[
                InputKey::new("height@0", same_shape_b),
                InputKey::new("water@1", same_shape_a),
            ],
            b"c",
            &engine,
        )
        .0;
        assert_ne!(left, right);
    }

    #[test]
    fn stored_material_reconstructs_the_key() {
        let engine = EngineFingerprint::current().canonical_bytes();
        let (key, material) = pass_key(
            "x",
            b"pipeline",
            b"uniform",
            &[InputKey::new("dem@0", leaf_key(b"dem"))],
            b"caps",
            &engine,
        );
        assert_eq!(reconstruct_pass_key(&material).unwrap(), key);
    }

    #[test]
    fn stored_material_preserves_rejection_of_altered_fields() {
        let (_, material) = pass_key("x", b"pipeline", b"uniform", &[], b"caps", b"engine");
        let alterations: [fn(&mut PassKeyMaterial); 5] = [
            |value| value.pipeline_descriptor_hash = sha256(b"other"),
            |value| value.uniform_sha256 = sha256(b"other"),
            |value| value.uniform_byte_length += 1,
            |value| value.capability_fingerprint_sha256 = sha256(b"other"),
            |value| value.engine_fingerprint_sha256 = sha256(b"other"),
        ];
        for alter in alterations {
            let mut invalid = material.clone();
            alter(&mut invalid);
            assert!(reconstruct_pass_key(&invalid).is_err());
        }
    }

    #[test]
    fn stored_material_rejects_noncanonical_hex_and_input_order() {
        let (_, material) = pass_key(
            "x",
            &[0xab],
            &[0xcd],
            &[
                InputKey::new("a", leaf_key(b"a")),
                InputKey::new("b", leaf_key(b"b")),
            ],
            &[0xef],
            &[0xab],
        );
        for field in ["pipeline", "uniform", "capabilities", "engine"] {
            let mut invalid = material.clone();
            let hex = match field {
                "pipeline" => &mut invalid.pipeline_descriptor_hex,
                "uniform" => &mut invalid.uniform_hex,
                "capabilities" => &mut invalid.capability_fingerprint_hex,
                _ => &mut invalid.engine_fingerprint_hex,
            };
            *hex = hex.to_uppercase();
            assert!(reconstruct_pass_key(&invalid).is_err());
        }
        let mut invalid = material.clone();
        invalid.uniform_hex = "é".into();
        assert!(reconstruct_pass_key(&invalid).is_err());
        invalid = material;
        invalid.input_keys.reverse();
        assert!(reconstruct_pass_key(&invalid).is_err());
    }

    #[test]
    fn pass_key_rejects_non_hex_utf8_and_signs_without_panicking() {
        for invalid in [
            "\u{1f4a9}".repeat(16),
            format!("{}0", "\u{20ac}".repeat(21)),
            "+0".repeat(32),
        ] {
            assert_eq!(invalid.len(), 64);
            assert!(PassKey::from_hex(&invalid).is_err());
            let encoded = serde_json::to_string(&invalid).unwrap();
            assert!(serde_json::from_str::<PassKey>(&encoded).is_err());
        }
    }

    #[test]
    fn pass_key_preserves_ascii_hex_case_compatibility() {
        let lower = "ab".repeat(32);
        let upper = lower.to_ascii_uppercase();
        let expected = PassKey([0xab; 32]);
        assert_eq!(PassKey::from_hex(&lower).unwrap(), expected);
        assert_eq!(PassKey::from_hex(&upper).unwrap(), expected);
        assert_eq!(expected.to_hex(), lower);
    }
}
