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
    pub(super) material_maps: std::collections::BTreeMap<String, Vec<u8>>,
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
        let limits = context.device.limits();
        let capabilities = CapabilityFingerprint {
            granted_features: context
                .capabilities
                .granted_names()
                .into_iter()
                .map(str::to_string)
                .collect(),
            limits: crate::core::anamnesis::key::device_limits(&limits),
            backend: format!("{:?}", context.adapter.get_info().backend).to_lowercase(),
            dx12_compiler: context.dx12_compiler.into(),
            naga_capabilities: crate::core::anamnesis::key::naga_policy(),
        };
        let capability_bytes = cache_capability_bytes(
            &capabilities,
            portable.as_deref(),
            gpu::deterministic_mode(),
        )?;
        Ok(Self {
            root,
            state_bytes,
            height_bytes,
            shadow_bytes,
            capability_bytes,
            engine_bytes: EngineFingerprint::current().canonical_bytes(),
            material_maps: Default::default(),
        })
    }

    pub(super) fn store(&self) -> std::io::Result<ContentStore> {
        ContentStore::new(&self.root, 512 * 1024 * 1024, true)
    }
}

// A requested profile is an additional identity dimension, never authority
// to discard device identity. Cross-backend reuse is unqualified while the
// independent physical renders differ; old portable entries must miss.
fn cache_capability_bytes(
    capabilities: &CapabilityFingerprint,
    profile: Option<&str>,
    deterministic: bool,
) -> Result<Vec<u8>> {
    let canonical = capabilities.canonical_bytes();
    if let Some(profile) = profile {
        if !deterministic {
            return Err(anyhow!(
                "{PORTABLE_PROFILE} requires FORGE3D_DETERMINISTIC=1"
            ));
        }
        let mut bytes =
            format!("forge3d.anamnesis.compatibility-profile/v1\0{profile}\0").into_bytes();
        bytes.extend_from_slice(&canonical);
        Ok(bytes)
    } else {
        Ok(canonical)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(super) enum BlobKind {
    Prepared = 1,
    ShadowDepth32 = 2,
    Rgba8 = 3,
    Rgba16Float = 4,
}

pub(super) fn color_blob_format(format: wgpu::TextureFormat) -> Result<(BlobKind, u32)> {
    match format {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
            Ok((BlobKind::Rgba8, 4))
        }
        wgpu::TextureFormat::Rgba16Float => Ok((BlobKind::Rgba16Float, 8)),
        other => Err(anyhow!(
            "unsupported native terrain cache color format {other:?}"
        )),
    }
}

pub(super) struct PassBlob<'a> {
    pub(super) kind: BlobKind,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) layers: u32,
    pub(super) payload: &'a [u8],
}

fn blob_u32(blob: &[u8], offset: usize) -> Result<u32> {
    let bytes = blob
        .get(offset..offset + 4)
        .ok_or_else(|| anyhow!("native terrain blob header is truncated"))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn blob_u64(blob: &[u8], offset: usize) -> Result<u64> {
    let bytes = blob
        .get(offset..offset + 8)
        .ok_or_else(|| anyhow!("native terrain blob header is truncated"))?;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
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
    let kind = match blob_u32(blob, 8)? {
        1 => BlobKind::Prepared,
        2 => BlobKind::ShadowDepth32,
        3 => BlobKind::Rgba8,
        4 => BlobKind::Rgba16Float,
        value => return Err(anyhow!("unknown native terrain blob kind {value}")),
    };
    let payload_len = blob_u64(blob, 24)?;
    let payload_len = usize::try_from(payload_len).context("terrain blob length overflow")?;
    if blob.len() != 32 + payload_len {
        return Err(anyhow!(
            "native terrain blob length mismatch: header={payload_len}, actual={}",
            blob.len().saturating_sub(32)
        ));
    }
    Ok(PassBlob {
        kind,
        width: blob_u32(blob, 12)?,
        height: blob_u32(blob, 16)?,
        layers: blob_u32(blob, 20)?,
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
    let color_stride = match expected_kind {
        BlobKind::Rgba8 => Some(4u64),
        BlobKind::Rgba16Float => Some(8u64),
        _ => None,
    };
    if let Some(stride) = color_stride {
        let expected = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|size| size.checked_mul(u64::from(layers)))
            .and_then(|size| size.checked_mul(stride))
            .ok_or_else(|| anyhow!("native terrain color payload size overflow"))?;
        if decoded.payload.len() as u64 != expected {
            return Err(anyhow!("native terrain color payload byte length mismatch"));
        }
    }
    Ok(decoded.payload)
}

const EXTERNAL_STATE_DOMAIN: &[u8] = b"forge3d.terrain.renderer-external-state/v1\0";

pub(super) fn append_default_external_state(bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(EXTERNAL_STATE_DOMAIN);
    bytes.push(0); // A new TerrainScene has no light override or mutable sources.
}

impl super::TerrainScene {
    pub(super) fn supports_anamnesis_cache(&self) -> bool {
        self.anamnesis_cache_bypass_reason().is_none()
    }

    pub(super) fn anamnesis_cache_bypass_reason(&self) -> Option<&'static str> {
        #[cfg(feature = "extension-module")]
        match self.material_vt.lock() {
            Ok(vt) if !vt.sources.is_empty() => return Some("material_virtual_textures"),
            Err(_) => return Some("material_vt_state_unavailable"),
            _ => {}
        }
        #[cfg(feature = "enable-gpu-instancing")]
        if !self.scatter_batches.is_empty() {
            return Some("scatter");
        }
        if self.viewer_heightmap.is_some() {
            return Some("viewer_heightmap");
        }
        if self.height_streaming.is_some() {
            return Some("height_streaming");
        }
        None
    }

    pub(super) fn append_anamnesis_external_state(&self, bytes: &mut Vec<u8>) -> Result<()> {
        bytes.extend_from_slice(EXTERNAL_STATE_DOMAIN);
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
    fn unqualified_portable_profile_retains_device_identity() {
        let base = CapabilityFingerprint {
            granted_features: vec![],
            limits: [("max_texture_dimension_2d".into(), 8192)].into(),
            backend: "vulkan".into(),
            dx12_compiler: "fxc".into(),
            naga_capabilities: vec![],
        };
        let bytes = cache_capability_bytes(&base, Some(PORTABLE_PROFILE), true).unwrap();
        assert!(bytes.ends_with(&base.canonical_bytes()));
        for changed in [
            CapabilityFingerprint {
                backend: "dx12".into(),
                ..base.clone()
            },
            CapabilityFingerprint {
                dx12_compiler: "dxc".into(),
                ..base.clone()
            },
            CapabilityFingerprint {
                limits: [("max_texture_dimension_2d".into(), 4096)].into(),
                ..base.clone()
            },
            CapabilityFingerprint {
                granted_features: vec!["timestamp-query".into()],
                ..base.clone()
            },
            CapabilityFingerprint {
                naga_capabilities: vec!["float64".into()],
                ..base.clone()
            },
        ] {
            assert_ne!(
                bytes,
                cache_capability_bytes(&changed, Some(PORTABLE_PROFILE), true).unwrap()
            );
        }
        assert_ne!(
            bytes,
            format!("forge3d.anamnesis.compatibility-profile/v1\0{PORTABLE_PROFILE}").into_bytes()
        );
        assert_eq!(
            cache_capability_bytes(&base, None, false).unwrap(),
            base.canonical_bytes()
        );
        assert!(cache_capability_bytes(&base, Some(PORTABLE_PROFILE), false).is_err());
    }

    #[test]
    fn pass_blob_descriptor_is_self_validating() {
        let blob = encode_blob(BlobKind::Rgba8, 2, 1, 1, &[1; 8]);
        assert_eq!(
            require_blob(&blob, BlobKind::Rgba8, 2, 1, 1).unwrap(),
            &[1; 8]
        );
        assert!(require_blob(&blob, BlobKind::Rgba8, 1, 2, 1).is_err());
        let hdr = encode_blob(BlobKind::Rgba16Float, 2, 1, 1, &[1; 16]);
        assert_eq!(
            require_blob(&hdr, BlobKind::Rgba16Float, 2, 1, 1).unwrap(),
            &[1; 16]
        );
        assert!(require_blob(&hdr, BlobKind::Rgba8, 2, 1, 1).is_err());
        let truncated = encode_blob(BlobKind::Rgba16Float, 2, 1, 1, &[1; 8]);
        assert!(require_blob(&truncated, BlobKind::Rgba16Float, 2, 1, 1).is_err());
    }
}
