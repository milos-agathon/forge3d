// src/viewer/render/snapshot.rs
// Snapshot/texture readback helpers for the interactive viewer

use std::path::Path;
use wgpu::{Device, Queue, Texture, TextureFormat};

use crate::renderer::readback::read_texture_tight;
use crate::util::image_write;

/// Save a texture to PNG (RGBA8/BGRA8 only)
pub fn snapshot_texture_to_png(
    device: &Device,
    queue: &Queue,
    tex: &Texture,
    path: &str,
) -> anyhow::Result<()> {
    use anyhow::{bail, Context};

    let size = tex.size();
    let w = size.width;
    let h = size.height;
    let fmt = tex.format();

    match fmt {
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => {
            let mut data =
                read_texture_tight(device, queue, tex, (w, h), fmt).context("readback failed")?;
            for px in data.chunks_exact_mut(4) {
                px[3] = 255;
            }
            image_write::write_png_rgba8(Path::new(path), &data, w, h)
                .context("failed to write PNG")?;
            Ok(())
        }
        TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb => {
            let mut data =
                read_texture_tight(device, queue, tex, (w, h), fmt).context("readback failed")?;
            for px in data.chunks_exact_mut(4) {
                px.swap(0, 2);
                px[3] = 255;
            }
            image_write::write_png_rgba8(Path::new(path), &data, w, h)
                .context("failed to write PNG")?;
            Ok(())
        }
        other => {
            bail!(
                "snapshot only supports RGBA8/BGRA8 surfaces (got {:?})",
                other
            )
        }
    }
}
