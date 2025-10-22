//! UV transformation utilities for texture coordinate mapping
//!
//! Handles Y-flip, scale, and offset transformations to align textures
//! from different coordinate systems (e.g., image top-left vs bottom-left origin).

use wgpu::{Device, Buffer, BufferUsages, util::DeviceExt};

/// UV transformation parameters (matches WGSL UVTransform struct)
/// Note: Using repr(C) without align to allow bytemuck::Pod derivation
/// The WGSL std140 layout naturally aligns this to 16 bytes
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UVTransform {
    /// Scale factors for U and V
    pub scale: [f32; 2],
    /// Offset for U and V
    pub offset: [f32; 2],
    /// Y-flip flag (0 = no flip, 1 = flip)
    pub y_flip: u32,
    /// Padding to maintain alignment
    pub _pad: u32,
}

impl Default for UVTransform {
    fn default() -> Self {
        Self {
            scale: [1.0, 1.0],
            offset: [0.0, 0.0],
            y_flip: 0,
            _pad: 0,
        }
    }
}

impl UVTransform {
    /// Create identity transform (no changes)
    pub fn identity() -> Self {
        Self::default()
    }

    /// Create transform with Y-flip enabled
    pub fn with_y_flip() -> Self {
        Self {
            scale: [1.0, 1.0],
            offset: [0.0, 0.0],
            y_flip: 1,
            _pad: 0,
        }
    }

    /// Create transform with custom scale and offset
    pub fn new(scale: [f32; 2], offset: [f32; 2], y_flip: bool) -> Self {
        Self {
            scale,
            offset,
            y_flip: if y_flip { 1 } else { 0 },
            _pad: 0,
        }
    }

    /// Create a uniform buffer containing this transform
    pub fn create_buffer(&self, device: &Device) -> Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uv_transform_uniform"),
            contents: bytemuck::bytes_of(self),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        })
    }

    /// Create transform for aligning two textures with different resolutions
    ///
    /// # Arguments
    /// * `src_width`, `src_height` - Source texture dimensions
    /// * `dst_width`, `dst_height` - Destination texture dimensions
    /// * `y_flip` - Whether to flip Y coordinate
    pub fn for_alignment(
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        y_flip: bool,
    ) -> Self {
        let scale_x = (src_width as f32) / (dst_width as f32);
        let scale_y = (src_height as f32) / (dst_height as f32);

        Self::new([scale_x, scale_y], [0.0, 0.0], y_flip)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uv_transform_size() {
        // Ensure struct is properly sized for WGSL uniform (24 bytes)
        assert_eq!(std::mem::size_of::<UVTransform>(), 24);
        // Natural alignment is 4 bytes (f32), WGSL handles 16-byte uniform alignment
        assert_eq!(std::mem::align_of::<UVTransform>(), 4);
    }

    #[test]
    fn test_identity_transform() {
        let xform = UVTransform::identity();
        assert_eq!(xform.scale, [1.0, 1.0]);
        assert_eq!(xform.offset, [0.0, 0.0]);
        assert_eq!(xform.y_flip, 0);
    }

    #[test]
    fn test_y_flip_transform() {
        let xform = UVTransform::with_y_flip();
        assert_eq!(xform.y_flip, 1);
    }

    #[test]
    fn test_custom_transform() {
        let xform = UVTransform::new([2.0, 3.0], [0.1, 0.2], true);
        assert_eq!(xform.scale, [2.0, 3.0]);
        assert_eq!(xform.offset, [0.1, 0.2]);
        assert_eq!(xform.y_flip, 1);
    }

    #[test]
    fn test_alignment_transform() {
        let xform = UVTransform::for_alignment(2048, 1024, 1024, 512, false);
        assert_eq!(xform.scale, [2.0, 2.0]);
        assert_eq!(xform.y_flip, 0);
    }
}
