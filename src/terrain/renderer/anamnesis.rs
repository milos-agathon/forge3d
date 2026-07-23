//! Native terrain ANAMNESIS declarations and self-describing pass blobs.

use crate::core::anamnesis::{CapabilityFingerprint, ContentStore, EngineFingerprint};
use crate::core::gpu;
use anyhow::{anyhow, Context, Result};
use std::path::PathBuf;

pub(super) const PORTABLE_PROFILE: &str = "terra-determinata-native-portable-v1";
const BLOB_MAGIC: &[u8; 8] = b"F3DANM02";

#[derive(Clone)]
pub(crate) struct TerrainCacheOptions {
    pub(super) root: PathBuf,
    pub(super) state_bytes: Vec<u8>,
    pub(super) height_bytes: Vec<u8>,
    pub(super) shadow_bytes: Vec<u8>,
    pub(super) capability_bytes: Vec<u8>,
    pub(super) engine_bytes: Vec<u8>,
}

impl TerrainCacheOptions {
    pub(super) fn new(
        root: PathBuf,
        state_bytes: Vec<u8>,
        height_bytes: Vec<u8>,
        shadow_bytes: Vec<u8>,
    ) -> Result<Self> {
        let context = gpu::try_ctx().map_err(|error| anyhow!("{error}"))?;
        let portable = std::env::var("FORGE3D_ANAMNESIS_COMPATIBILITY_PROFILE")
            .ok()
            .filter(|profile| profile == PORTABLE_PROFILE);
        let capability_bytes = if let Some(profile) = portable {
            if !gpu::deterministic_mode() {
                return Err(anyhow!(
                    "{PORTABLE_PROFILE} requires FORGE3D_DETERMINISTIC=1"
                ));
            }
            format!("forge3d.anamnesis.compatibility-profile/v1\0{profile}").into_bytes()
        } else {
            let limits = context.device.limits();
            CapabilityFingerprint {
                granted_features: context
                    .capabilities
                    .granted_names()
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
                limits: std::collections::BTreeMap::from([
                    (
                        "max_texture_dimension_2d".into(),
                        limits.max_texture_dimension_2d as u64,
                    ),
                    ("max_buffer_size".into(), limits.max_buffer_size),
                    ("max_bind_groups".into(), limits.max_bind_groups as u64),
                    (
                        "max_storage_buffers_per_shader_stage".into(),
                        limits.max_storage_buffers_per_shader_stage as u64,
                    ),
                    (
                        "min_uniform_buffer_offset_alignment".into(),
                        limits.min_uniform_buffer_offset_alignment as u64,
                    ),
                    (
                        "min_storage_buffer_offset_alignment".into(),
                        limits.min_storage_buffer_offset_alignment as u64,
                    ),
                ]),
                backend: format!("{:?}", context.adapter.get_info().backend).to_lowercase(),
                dx12_compiler: context.dx12_compiler.into(),
                naga_capabilities: vec![format!(
                    "wgpu-validation-default@naga-{}",
                    env!("FORGE3D_NAGA_VERSION")
                )],
            }
            .canonical_bytes()
        };
        Ok(Self {
            root,
            state_bytes,
            height_bytes,
            shadow_bytes,
            capability_bytes,
            engine_bytes: EngineFingerprint::current().canonical_bytes(),
        })
    }

    pub(super) fn store(&self) -> std::io::Result<ContentStore> {
        ContentStore::new(&self.root, 512 * 1024 * 1024, true)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(super) enum BlobKind {
    Prepared = 1,
    ShadowDepth32 = 2,
    Rgba8 = 3,
}

pub(super) struct PassBlob<'a> {
    pub(super) kind: BlobKind,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) layers: u32,
    pub(super) payload: &'a [u8],
}

pub(super) fn encode_blob(
    kind: BlobKind,
    width: u32,
    height: u32,
    layers: u32,
    payload: &[u8],
) -> Vec<u8> {
    let mut blob = Vec::with_capacity(36 + payload.len());
    blob.extend_from_slice(BLOB_MAGIC);
    blob.extend_from_slice(&(kind as u32).to_le_bytes());
    blob.extend_from_slice(&width.to_le_bytes());
    blob.extend_from_slice(&height.to_le_bytes());
    blob.extend_from_slice(&layers.to_le_bytes());
    blob.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    blob.extend_from_slice(payload);
    blob
}

pub(super) fn decode_blob(blob: &[u8]) -> Result<PassBlob<'_>> {
    if blob.len() < 32 || &blob[..8] != BLOB_MAGIC {
        return Err(anyhow!("invalid native terrain ANAMNESIS blob header"));
    }
    let word = |offset: usize| {
        u32::from_le_bytes(
            blob[offset..offset + 4]
                .try_into()
                .expect("validated blob header length"),
        )
    };
    let kind = match word(8) {
        1 => BlobKind::Prepared,
        2 => BlobKind::ShadowDepth32,
        3 => BlobKind::Rgba8,
        value => return Err(anyhow!("unknown native terrain blob kind {value}")),
    };
    let payload_len = u64::from_le_bytes(
        blob[24..32]
            .try_into()
            .expect("validated blob header length"),
    );
    let payload_len = usize::try_from(payload_len).context("terrain blob length overflow")?;
    if blob.len() != 32 + payload_len {
        return Err(anyhow!(
            "native terrain blob length mismatch: header={payload_len}, actual={}",
            blob.len().saturating_sub(32)
        ));
    }
    Ok(PassBlob {
        kind,
        width: word(12),
        height: word(16),
        layers: word(20),
        payload: &blob[32..],
    })
}

pub(super) fn require_blob(
    blob: &[u8],
    expected_kind: BlobKind,
    width: u32,
    height: u32,
    layers: u32,
) -> Result<&[u8]> {
    let decoded = decode_blob(blob)?;
    if decoded.kind != expected_kind
        || decoded.width != width
        || decoded.height != height
        || decoded.layers != layers
    {
        return Err(anyhow!(
            "native terrain blob descriptor mismatch: got {:?} {}x{}x{}, expected {:?} {}x{}x{}",
            decoded.kind,
            decoded.width,
            decoded.height,
            decoded.layers,
            expected_kind,
            width,
            height,
            layers
        ));
    }
    Ok(decoded.payload)
}

/// Deterministic lossless RLE for depth32 words. Shadow maps are dominated by
/// the cleared 1.0 depth value, so this keeps a 600-frame native store bounded
/// without weakening restoration semantics.
pub(super) fn encode_depth_rle(bytes: &[u8]) -> Result<Vec<u8>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(anyhow!("depth32 payload is not word aligned"));
    }
    let mut encoded = Vec::new();
    let mut words = bytes.chunks_exact(4);
    let Some(first) = words.next() else {
        return Ok(encoded);
    };
    let mut current = <[u8; 4]>::try_from(first).expect("four-byte chunk");
    let mut count = 1u32;
    for word in words {
        let word = <[u8; 4]>::try_from(word).expect("four-byte chunk");
        if word == current && count != u32::MAX {
            count += 1;
        } else {
            encoded.extend_from_slice(&count.to_le_bytes());
            encoded.extend_from_slice(&current);
            current = word;
            count = 1;
        }
    }
    encoded.extend_from_slice(&count.to_le_bytes());
    encoded.extend_from_slice(&current);
    Ok(encoded)
}

pub(super) fn decode_depth_rle(encoded: &[u8], expected_bytes: usize) -> Result<Vec<u8>> {
    if !encoded.len().is_multiple_of(8) || !expected_bytes.is_multiple_of(4) {
        return Err(anyhow!("invalid depth32 RLE alignment"));
    }
    let mut decoded = Vec::with_capacity(expected_bytes);
    for run in encoded.chunks_exact(8) {
        let count = u32::from_le_bytes(run[..4].try_into().expect("four-byte run count"));
        if count == 0 {
            return Err(anyhow!("depth32 RLE contains a zero-length run"));
        }
        let word = &run[4..8];
        let append = (count as usize)
            .checked_mul(4)
            .ok_or_else(|| anyhow!("depth32 RLE run overflow"))?;
        let new_len = decoded
            .len()
            .checked_add(append)
            .ok_or_else(|| anyhow!("depth32 RLE output overflow"))?;
        if new_len > expected_bytes {
            return Err(anyhow!("depth32 RLE expands past its declared extent"));
        }
        for _ in 0..count {
            decoded.extend_from_slice(word);
        }
    }
    if decoded.len() != expected_bytes {
        return Err(anyhow!(
            "depth32 RLE length mismatch: expected {expected_bytes}, got {}",
            decoded.len()
        ));
    }
    Ok(decoded)
}

impl super::TerrainScene {
    pub(super) fn append_anamnesis_external_state(&self, bytes: &mut Vec<u8>) -> Result<()> {
        bytes.extend_from_slice(b"forge3d.terrain.renderer-external-state/v1\0");
        let lights = self
            .light_override
            .lock()
            .map_err(|_| anyhow!("terrain light override mutex poisoned"))?;
        match lights.as_ref() {
            Some(lights) => {
                bytes.push(1);
                bytes.extend_from_slice(&(lights.len() as u64).to_le_bytes());
                bytes.extend_from_slice(bytemuck::cast_slice(lights));
            }
            None => bytes.push(0),
        }
        #[cfg(feature = "enable-gpu-instancing")]
        if !self.scatter_batches.is_empty() {
            return Err(anyhow!(
                "native terrain ANAMNESIS does not yet accept mutable scatter batches"
            ));
        }
        if self.viewer_heightmap.is_some() || self.height_streaming.is_some() {
            return Err(anyhow!(
                "native terrain ANAMNESIS requires one-shot heightmap ownership"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_rle_is_lossless_and_rejects_bad_runs() {
        let words = [1.0f32, 1.0, 1.0, 0.25, 0.25, 0.0, 1.0];
        let bytes = bytemuck::cast_slice(&words);
        let encoded = encode_depth_rle(bytes).unwrap();
        assert!(encoded.len() < bytes.len() + 24);
        assert_eq!(decode_depth_rle(&encoded, bytes.len()).unwrap(), bytes);

        let mut zero_run = encoded;
        zero_run[..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(decode_depth_rle(&zero_run, bytes.len()).is_err());
    }

    #[test]
    fn pass_blob_descriptor_is_self_validating() {
        let blob = encode_blob(BlobKind::Rgba8, 2, 1, 1, &[1; 8]);
        assert_eq!(
            require_blob(&blob, BlobKind::Rgba8, 2, 1, 1).unwrap(),
            &[1; 8]
        );
        assert!(require_blob(&blob, BlobKind::Rgba8, 1, 2, 1).is_err());
    }
}
