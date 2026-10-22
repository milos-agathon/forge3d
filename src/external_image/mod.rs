//! External image import functionality for forge3d.
//!
//! Provides native copyExternalImageToTexture-like functionality for importing
//! PNG/JPEG images into GPU textures with proper format handling.
//!
//! ## Supported Formats
//! - **PNG**: RGBA8, RGB8, Grayscale (all converted to RGBA8)
//! - **JPEG**: RGB8 (converted to RGBA8)
//! - **Output**: Always RGBA8UnormSrgb for consistency
//!
//! ## Usage
//! ```rust,ignore
//! use forge3d::external_image::{import_image_to_texture, ImageImportConfig};
//!
//! let config = ImageImportConfig::default();
//! let texture_info = import_image_to_texture(device, queue, "image.png", config)?;
//! ```

mod decode;
pub mod types;
mod upload;

pub use types::{ImageImportConfig, ImageSourceFormat, ImportedTextureInfo};

use crate::core::error::{RenderError, RenderResult};
use crate::core::memory_tracker::global_tracker;
use decode::decode_image_file;
use std::path::Path;
use upload::{create_texture_for_import, upload_rgba_data_to_texture};

/// Import an external image file into a GPU texture.
///
/// Decodes the image file and uploads it directly to a GPU texture with format conversion.
///
/// # Arguments
/// * `device` - WGPU device for texture creation
/// * `queue` - WGPU queue for upload operations
/// * `image_path` - Path to the image file (PNG or JPEG)
/// * `config` - Import configuration options
///
/// # Returns
/// Returns `ImportedTextureInfo` containing the texture and metadata.
pub fn import_image_to_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image_path: impl AsRef<Path>,
    config: ImageImportConfig,
) -> RenderResult<ImportedTextureInfo> {
    let path = image_path.as_ref();

    if !path.exists() {
        return Err(RenderError::io(format!(
            "Image file not found: {}",
            path.display()
        )));
    }

    if config.generate_mipmaps {
        return Err(RenderError::Upload(
            "External image mipmap generation is unsupported".into(),
        ));
    }
    if !matches!(
        config.target_format,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
    ) {
        return Err(RenderError::Upload(
            "External images require RGBA8 texture format".into(),
        ));
    }
    let (rgba_data, width, height, source_format) = decode_image_file(path, &config)?;

    if width > config.max_dimension || height > config.max_dimension {
        return Err(RenderError::Upload(format!(
            "Image dimensions {}x{} exceed maximum allowed {}x{}",
            width, height, config.max_dimension, config.max_dimension
        )));
    }

    let texture_size = (width as u64) * (height as u64) * 4;
    let _metrics = global_tracker().get_metrics();

    let texture = create_texture_for_import(device, width, height, &config)?;
    upload_rgba_data_to_texture(queue, &texture, &rgba_data, width, height)?;

    // `create_texture_for_import` routes through the tracked wrapper, which
    // already records the texture in the global allocation ledger.
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    Ok(ImportedTextureInfo {
        texture,
        view,
        width,
        height,
        source_format,
        texture_format: config.target_format,
        size_bytes: texture_size,
    })
}

/// Decode and validate an image file to report its actual dimensions and format.
pub fn probe_image_info(
    image_path: impl AsRef<Path>,
) -> RenderResult<(u32, u32, ImageSourceFormat)> {
    let path = image_path.as_ref();

    if !path.exists() {
        return Err(RenderError::io(format!(
            "Image file not found: {}",
            path.display()
        )));
    }

    let (_, width, height, format) = decode_image_file(path, &ImageImportConfig::default())?;
    Ok((width, height, format))
}

/// Check if external image import is available.
pub fn is_external_image_available() -> bool {
    true
}

/// Get supported image formats.
pub fn get_supported_formats() -> Vec<&'static str> {
    vec!["png", "jpg", "jpeg"]
}

/// WebGPU parity constraints and limitations.
pub mod constraints {
    //! Documents constraints compared to WebGPU copyExternalImageToTexture.

    /// Maximum texture dimension supported.
    pub const MAX_TEXTURE_DIMENSION: u32 = 8192;

    /// Memory budget for textures (512 MiB).
    pub const MEMORY_BUDGET_BYTES: u64 = 512 * 1024 * 1024;

    /// Supported input formats.
    pub const SUPPORTED_INPUT_FORMATS: &[&str] = &["PNG", "JPEG"];

    /// Output format (always the same for consistency).
    pub const OUTPUT_FORMAT: &str = "RGBA8UnormSrgb";
}
