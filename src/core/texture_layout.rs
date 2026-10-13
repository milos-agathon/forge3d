//! Internal texture storage and transfer geometry.
//!
//! Resident sizes are accounting estimates, not measured driver allocations.
//! They include every mip, layer/depth slice and sample, but no copy-row padding.
//! Copy payloads describe one aspect, one mip and one sample. Queue writes use
//! tight rows; encoder buffer copies use COPY_BYTES_PER_ROW_ALIGNMENT.

use wgpu::{
    Extent3d, ImageDataLayout, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FormatLayout {
    pub block_width: u32,
    pub block_height: u32,
    pub resident_bytes_per_block: u32,
}

impl FormatLayout {
    pub fn new(format: TextureFormat) -> Self {
        let (block_width, block_height) = format.block_dimensions();
        // wgpu deliberately has no All-aspect copy size for these formats.
        // Preserve the ledger's conservative accounting policy independently
        // of the per-aspect transfer size (which may be unavailable).
        let resident_bytes_per_block = match format {
            TextureFormat::Depth24Plus | TextureFormat::Depth24PlusStencil8 => 4,
            TextureFormat::Depth32FloatStencil8 => 8,
            TextureFormat::NV12 => 4,
            _ => format
                .block_copy_size(None)
                .expect("format has a block size"),
        };
        Self {
            block_width,
            block_height,
            resident_bytes_per_block,
        }
    }

    pub fn block_counts(self, extent: Extent3d) -> (u32, u32) {
        (
            extent.width.div_ceil(self.block_width),
            extent.height.div_ceil(self.block_height),
        )
    }

    pub fn resident_level_size(self, extent: Extent3d) -> u64 {
        let (columns, rows) = self.block_counts(extent);
        u64::from(columns)
            .saturating_mul(u64::from(rows))
            .saturating_mul(u64::from(extent.depth_or_array_layers))
            .saturating_mul(u64::from(self.resident_bytes_per_block))
    }
}

/// Logical mip extent: D2 array layers stay fixed; D3 depth shrinks.
pub(crate) fn mip_extent(size: Extent3d, dimension: TextureDimension, mip: u32) -> Extent3d {
    let shrink = |value: u32| value.checked_shr(mip).unwrap_or(0).max(1);
    Extent3d {
        width: shrink(size.width),
        height: if dimension == TextureDimension::D1 {
            1
        } else {
            shrink(size.height)
        },
        depth_or_array_layers: match dimension {
            TextureDimension::D1 => 1,
            TextureDimension::D2 => size.depth_or_array_layers,
            TextureDimension::D3 => shrink(size.depth_or_array_layers),
        },
    }
}

pub(crate) fn resident_size(
    format: TextureFormat,
    size: Extent3d,
    dimension: TextureDimension,
    mip_level_count: u32,
    sample_count: u32,
) -> u64 {
    let layout = FormatLayout::new(format);
    (0..mip_level_count)
        .map(|mip| layout.resident_level_size(mip_extent(size, dimension, mip)))
        .fold(0u64, u64::saturating_add)
        .saturating_mul(u64::from(sample_count))
}

pub(crate) fn resident_descriptor_size(desc: &TextureDescriptor<'_>) -> u64 {
    resident_size(
        desc.format,
        desc.size,
        desc.dimension,
        desc.mip_level_count,
        desc.sample_count,
    )
}

pub(crate) fn resident_2d_size(width: u32, height: u32, format: TextureFormat) -> u64 {
    FormatLayout::new(format).resident_level_size(Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    })
}

/// Compatibility calculation for callers supplying their own square blocks.
pub(crate) fn square_block_size(
    width: u32,
    height: u32,
    bytes_per_block: u64,
    block_size: u32,
) -> u64 {
    u64::from(width.div_ceil(block_size))
        .saturating_mul(u64::from(height.div_ceil(block_size)))
        .saturating_mul(bytes_per_block)
}

/// Preserve sizing of public, caller-constructed TextureFormatInfo records.
/// Registry records derive these fields from FormatLayout; custom records may
/// intentionally describe a different square-block layout.
pub(crate) fn record_2d_size(
    width: u32,
    height: u32,
    is_compressed: bool,
    bytes_per_pixel: u32,
    block_size: u32,
) -> u64 {
    square_block_size(
        width,
        height,
        u64::from(bytes_per_pixel),
        if is_compressed { block_size } else { 1 },
    )
}

pub(crate) fn align_copy_row(bytes_per_row: u32) -> Result<u32, &'static str> {
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    bytes_per_row
        .div_ceil(alignment)
        .checked_mul(alignment)
        .ok_or("Padded texture row exceeds u32")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TransferLayout {
    pub row_bytes: u32,
    pub bytes_per_row: u32,
    pub rows_per_image: u32,
    /// Tight aspect payload, excluding padding, mips and multisampling.
    pub payload_bytes: u64,
    /// Full rows allocated for a transfer, including trailing row padding.
    pub transfer_bytes: u64,
}

impl TransferLayout {
    /// Extents use the selected aspect's coordinates (including NV12 planes).
    pub fn queue_write(
        format: TextureFormat,
        aspect: TextureAspect,
        extent: Extent3d,
    ) -> Result<Self, &'static str> {
        if extent.width == 0 || extent.height == 0 || extent.depth_or_array_layers == 0 {
            return Err("Texture transfer dimensions must be > 0");
        }
        if format.aspect_specific_format(aspect).is_none() {
            return Err("Texture aspect does not belong to this format");
        }
        let bytes_per_block = format
            .block_copy_size(Some(aspect))
            .ok_or("Texture aspect has no defined copy size")?;
        let (columns, rows) = FormatLayout::new(format).block_counts(extent);
        let row_bytes = columns
            .checked_mul(bytes_per_block)
            .ok_or("Texture row exceeds u32")?;
        let payload_bytes = u64::from(row_bytes)
            .checked_mul(u64::from(rows))
            .and_then(|bytes| bytes.checked_mul(u64::from(extent.depth_or_array_layers)))
            .ok_or("Texture payload exceeds u64")?;
        Ok(Self {
            row_bytes,
            bytes_per_row: row_bytes,
            rows_per_image: rows,
            payload_bytes,
            transfer_bytes: payload_bytes,
        })
    }

    pub fn buffer_copy(
        format: TextureFormat,
        aspect: TextureAspect,
        extent: Extent3d,
    ) -> Result<Self, &'static str> {
        let mut layout = Self::queue_write(format, aspect, extent)?;
        layout.bytes_per_row = align_copy_row(layout.row_bytes)?;
        layout.transfer_bytes = u64::from(layout.bytes_per_row)
            .checked_mul(u64::from(layout.rows_per_image))
            .and_then(|bytes| bytes.checked_mul(u64::from(extent.depth_or_array_layers)))
            .ok_or("Texture transfer exceeds u64")?;
        Ok(layout)
    }

    pub fn image_data_layout(self) -> ImageDataLayout {
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(self.bytes_per_row),
            rows_per_image: Some(self.rows_per_image),
        }
    }
}

// These contract tests exercise native registry, image and copy-helper callers.
#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "texture_layout_tests.rs"]
mod tests;
