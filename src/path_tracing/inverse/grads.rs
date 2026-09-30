// src/path_tracing/inverse/grads.rs
// DIFFERENTIA: memory-tracked inverse-state storage.
//
// Two consolidated storage buffers carry every inverse-side array so the
// shared group-2 layout stays within the negotiated storage-buffer budget
// alongside the forward bindings it must keep:
//
//   v4_buf  `array<vec4<f32>>`   — [0, px) the per-pixel adjoint
//                                  dL/d(linear radiance) written by the loss
//                                  kernel; [px, px + K*px) the per-(slot,
//                                  pixel) reservoir weight/channel snapshot
//                                  written by main_inv_wsnap. Only K =
//                                  min(tile_size, frames) frame histories are
//                                  resident at once — the checkpointed replay
//                                  in eval_pass refills the slots per
//                                  K-frame tile instead of caching all F.
//                                  [px + K*px, px + K*px + px) the
//                                  replicate-mean linear image accumulated by
//                                  main_inv_accum_mean.
//   u32_buf `array<atomic<u32>>` — [0, 8) the scalar gradient slots;
//                                  [8, 8+px) the per-pixel loss (bitcast f32,
//                                  single atomicStore per pixel);
//                                  [8+px, 8+px+4*texels) the per-texel albedo
//                                  gradient (rgb + spare, f32 CAS adds).
//
// All f32 accumulation goes through bitwise CAS on atomic<u32> slots — no
// fixed-point range, so photo-resolution gradients cannot wrap and no
// contribution is silently dropped for exceeding a representable interval.
// RELEVANT FILES: src/shaders/pt_inverse_loss.wgsl, src/shaders/pt_inverse_shade.wgsl,
//                 src/path_tracing/inverse/params.rs

use crate::core::error::RenderError;
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_texture, TrackedBuffer, TrackedTexture,
};

/// Scalar-slot count at the head of `u32_buf` (matches GRAD_SCALAR_SLOTS).
pub const INV_SCALARS_LEN: usize = 8;

/// All GPU-resident inverse state for one solve. Everything is created
/// through the tracked helpers so the memory ledger accounts it; only the
/// transient staging buffers in `read_*` are host-visible.
pub struct InverseBuffers {
    /// `array<vec4<f32>>`: adjoint[0..px], w_hist[px..px+slots*px], then the
    /// replicate-mean image [px+slots*px..px+slots*px+px].
    pub v4_buf: TrackedBuffer,
    /// `array<atomic<u32>>`: scalars[0..8], loss[8..8+px],
    /// dalbedo[8+px..8+px+4*texels].
    pub u32_buf: TrackedBuffer,
    /// Observed beauty image (sampled rgba8unorm).
    pub target_tex: TrackedTexture,
    pub px_count: u64,
    pub texels: u64,
    /// Live whist slots K = min(tile_size, frames).
    pub slots: u32,
}

impl InverseBuffers {
    /// Byte offset of the per-pixel loss region inside `u32_buf`.
    pub fn loss_offset(&self) -> u64 {
        (INV_SCALARS_LEN as u64) * 4
    }
    /// Byte offset of the per-texel albedo gradient region inside `u32_buf`.
    pub fn dalbedo_offset(&self) -> u64 {
        self.loss_offset() + self.px_count * 4
    }
    /// u32 index where the albedo region begins (for the WGSL side this is
    /// 8 + px; keep the two derivations in lockstep).
    pub fn dalbedo_slot_base(&self) -> u32 {
        (INV_SCALARS_LEN as u32) + self.px_count as u32
    }

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        alb_dims: (u32, u32),
        slots: u32,
        target_rgba: &[u8],
    ) -> Result<Self, RenderError> {
        let px = (width as u64) * (height as u64);
        let texels = (alb_dims.0 as u64) * (alb_dims.1 as u64);
        let slots = slots.max(1);
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let v4_len = px + px * slots as u64 + px;
        let u32_len = INV_SCALARS_LEN as u64 + px + texels * 4;
        let v4_buf = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("inv-v4"),
                size: (v4_len * 16).max(16),
                usage,
                mapped_at_creation: false,
            },
        )?;
        let u32_buf = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("inv-u32"),
                size: (u32_len * 4).max(4),
                usage,
                mapped_at_creation: false,
            },
        )?;
        let target_tex = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("inv-target"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &target_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            target_rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Ok(Self {
            v4_buf,
            u32_buf,
            target_tex,
            px_count: px,
            texels,
            slots,
        })
    }

    /// Re-write the observed beauty image (used by tests that synthesize the
    /// target on-device after construction).
    pub fn upload_target(&self, queue: &wgpu::Queue, width: u32, height: u32, rgba: &[u8]) {
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.target_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Decode the scalar gradient region (f32 bit patterns stored through the
/// CAS accumulators) into floats.
pub fn decode_scalars(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Tracked staging readback of a raw buffer (host-visible only for the map).
pub fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset: u64,
    size: u64,
) -> Result<Vec<u8>, RenderError> {
    let staging = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("inv-readback"),
            size: size.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )?;
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("inv-readback-enc"),
    });
    enc.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let out = slice.get_mapped_range().to_vec();
    staging.unmap();
    Ok(out)
}

/// Tracked staging readback of a texture, row-unpadded.
pub fn read_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    bytes_per_texel: u32,
) -> Result<Vec<u8>, RenderError> {
    let width = texture.width();
    let height = texture.height();
    let unpadded = width * bytes_per_texel;
    let padded =
        unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("inv-tex-readback"),
            size: padded as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )?;
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("inv-tex-readback-enc"),
    });
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let out = {
        let data = slice.get_mapped_range();
        let mut rows = Vec::with_capacity(unpadded as usize * height as usize);
        for y in 0..height as usize {
            let start = y * padded as usize;
            rows.extend_from_slice(&data[start..start + unpadded as usize]);
        }
        rows
    };
    staging.unmap();
    Ok(out)
}

/// Decode an RGBA16F readback into u8 RGBA — the same clamp+round the forward
/// path applies to its Reinhard-space out_tex (no sRGB encode).
pub fn rgba16f_to_rgba8(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2);
    for px in data.chunks_exact(8) {
        for c in 0..4 {
            let bits = u16::from_le_bytes([px[c * 2], px[c * 2 + 1]]);
            let v = half::f16::from_bits(bits).to_f32().clamp(0.0, 1.0);
            out.push((v * 255.0 + 0.5) as u8);
        }
    }
    out
}
