# Texture Memory Accounting

This document describes the comprehensive texture memory accounting system in forge3d, which tracks memory usage for all WebGPU texture formats.

## Overview

The texture memory accounting system provides accurate memory usage calculations for:
- **Uncompressed formats** - All standard WebGPU texture formats
- **Compressed formats** - Block-based compression (BC, ETC2, ASTC)
- **Depth/stencil formats** - Depth buffers and combined depth-stencil
- **HDR formats** - High dynamic range textures

## Supported Texture Formats

### 8-bit Formats (1 byte per pixel)
- `R8Unorm`, `R8Snorm`, `R8Uint`, `R8Sint`

### 16-bit Formats (2 bytes per pixel)
- **Two 8-bit channels**: `Rg8Unorm`, `Rg8Snorm`, `Rg8Uint`, `Rg8Sint`
- **Single 16-bit channel**: `R16Uint`, `R16Sint`, `R16Float`
- **Depth**: `Depth16Unorm`

### 32-bit Formats (4 bytes per pixel)
- **Standard RGBA**: `Rgba8Unorm`, `Rgba8UnormSrgb`, `Rgba8Snorm`, `Rgba8Uint`, `Rgba8Sint`
- **BGRA variants**: `Bgra8Unorm`, `Bgra8UnormSrgb`
- **Packed formats**: `Rgb10a2Unorm`, `Rgb10a2Uint`, `Rg11b10Float`
- **Two 16-bit channels**: `Rg16Uint`, `Rg16Sint`, `Rg16Float`
- **Single 32-bit channel**: `R32Uint`, `R32Sint`, `R32Float`
- **Depth formats**: `Depth32Float`, `Depth24Plus`, `Depth24PlusStencil8`

### 64-bit Formats (8 bytes per pixel)
- **Four 16-bit channels**: `Rgba16Uint`, `Rgba16Sint`, `Rgba16Float`
- **Two 32-bit channels**: `Rg32Uint`, `Rg32Sint`, `Rg32Float`
- **Depth+Stencil**: `Depth32FloatStencil8`

### 128-bit Formats (16 bytes per pixel)
- **Four 32-bit channels**: `Rgba32Uint`, `Rgba32Sint`, `Rgba32Float`

## Compressed Texture Formats

### BC (Block Compression) Formats

| Format | Block Size | Bytes per Block | Compression | Usage |
|--------|------------|-----------------|-------------|-------|
| **BC1** (DXT1) | 4×4 | 8 | 4:1 | RGB + 1-bit alpha |
| **BC2** (DXT3) | 4×4 | 16 | 2:1 | RGBA explicit alpha |
| **BC3** (DXT5) | 4×4 | 16 | 2:1 | RGBA interpolated alpha |
| **BC4** | 4×4 | 8 | 4:1 | Single channel |
| **BC5** | 4×4 | 16 | 2:1 | Two channels (normal maps) |
| **BC6H** | 4×4 | 16 | 2:1 | HDR RGB |
| **BC7** | 4×4 | 16 | 2:1 | High-quality RGBA |

### ETC2 Formats

| Format | Block Size | Bytes per Block | Usage |
|--------|------------|-----------------|-------|
| `Etc2Rgb8Unorm/Srgb` | 4×4 | 8 | RGB compression |
| `Etc2Rgb8A1Unorm/Srgb` | 4×4 | 8 | RGB + 1-bit alpha |
| `Etc2Rgba8Unorm/Srgb` | 4×4 | 16 | RGBA compression |
| `EacR11Unorm/Snorm` | 4×4 | 8 | Single channel |
| `EacRg11Unorm/Snorm` | 4×4 | 16 | Two channels |

### ASTC Formats

ASTC supports variable block sizes. The accounting system assumes 4×4 blocks with 16 bytes per block for memory estimation.

## Memory Calculation Examples

### Uncompressed Textures

```rust
// Example: 1024×1024 RGBA8 texture
let width = 1024;
let height = 1024; 
let format = TextureFormat::Rgba8Unorm; // 4 bytes per pixel
let memory_usage = width * height * 4; // 4,194,304 bytes (4 MiB)
```

### Compressed Textures

```rust
// Example: 1024×1024 BC3 (DXT5) texture
let width = 1024;
let height = 1024;
let format = TextureFormat::Bc3RgbaUnorm;

// BC3 uses 4×4 blocks with 16 bytes per block
let blocks_x = (1024 + 3) / 4; // 256 blocks
let blocks_y = (1024 + 3) / 4; // 256 blocks  
let memory_usage = 256 * 256 * 16; // 1,048,576 bytes (1 MiB)
```

## API Usage

### Tracking Texture Allocation

```rust
use forge3d::core::memory_tracker::global_tracker;
use wgpu::TextureFormat;

// Track texture allocation
let width = 512;
let height = 512;
let format = TextureFormat::Rgba16Float;

global_tracker().track_texture_allocation(width, height, format);

// Get current metrics
let metrics = global_tracker().get_metrics();
println!("Texture memory: {} bytes", metrics.texture_bytes);
println!("Texture count: {}", metrics.texture_count);
```

### Resource Handle Integration

```rust
use forge3d::core::resource_tracker::{ResourceHandle, register_texture};

// Automatic tracking with RAII cleanup
let _handle = register_texture(width, height, format);

// Memory is automatically freed when handle drops
```

## Memory Budget Management

### Budget Constraints

The system enforces a **512 MiB** memory budget for host-visible allocations:

```rust
let metrics = global_tracker().get_metrics();
if !metrics.within_budget {
    println!("Warning: Exceeding memory budget!");
    println!("Host-visible: {} MiB / 512 MiB", metrics.host_visible_bytes / (1024 * 1024));
}
```

### Memory Optimization Tips

1. **Use compressed textures** where possible:
   - BC1/BC3 for color textures (4:1 or 2:1 compression)
   - BC5 for normal maps (2:1 compression)
   - BC6H for HDR textures (2:1 compression)

2. **Choose appropriate formats**:
   - `R8Unorm` for single-channel data (1 byte vs 4 bytes for RGBA)
   - `Rg8Unorm` for two-channel data (2 bytes vs 4 bytes)
   - `R16Float` vs `R32Float` for precision requirements

3. **Optimize texture sizes**:
   - Use power-of-2 dimensions for better GPU performance
   - Consider mip-mapping for distant objects

## Implementation Details

### Calculation Functions

```rust
/// Calculate uncompressed texture size
fn calculate_texture_size(width: u32, height: u32, format: TextureFormat) -> u64 {
    let bytes_per_pixel = get_format_bytes_per_pixel(format);
    (width as u64) * (height as u64) * bytes_per_pixel
}

/// Calculate compressed texture size  
fn calculate_compressed_texture_size(
    width: u32, 
    height: u32, 
    bytes_per_block: u64, 
    block_size: u32
) -> u64 {
    let blocks_x = (width + block_size - 1) / block_size;
    let blocks_y = (height + block_size - 1) / block_size;
    (blocks_x as u64) * (blocks_y as u64) * bytes_per_block
}
```

### Error Handling

Unknown or future texture formats use a conservative 4-byte estimate to prevent underestimating memory usage:

```rust
_ => {
    // Conservative estimate for unknown formats
    4  // bytes per pixel
}
```

## Testing and Validation

### Test Coverage

The system includes comprehensive tests for:
- All supported uncompressed formats
- Common compressed formats (BC, ETC2)
- Block alignment calculations
- Memory accounting accuracy
- Edge cases (non-power-of-2 dimensions)

### Example Test

```rust
#[test]
fn test_bc3_compressed_size() {
    // 16×16 BC3 texture = 4×4 blocks, 16 bytes per block
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Bc3RgbaUnorm),
        4 * 4 * 16  // 256 bytes
    );
    
    // Non-aligned size rounds up to block boundaries
    assert_eq!(
        calculate_texture_size(17, 17, TextureFormat::Bc3RgbaUnorm), 
        5 * 5 * 16  // 400 bytes (20×20 blocks)
    );
}
```

## Performance Characteristics

### Memory Overhead

- **Tracking overhead**: ~16 bytes per tracked texture
- **Calculation cost**: O(1) for all formats
- **Thread safety**: Atomic operations for counters

### Platform Considerations

Different GPU vendors may have slight variations in actual memory usage:
- **NVIDIA**: Close to calculated values
- **AMD**: May use additional padding
- **Intel**: Conservative estimates work well
- **Mobile GPUs**: Tiled rendering may affect actual usage

## Future Extensions

### Planned Improvements

1. **3D Texture Support**: Extend calculations for volume textures
2. **Array Texture Support**: Account for layer count in arrays
3. **Mipmap Calculations**: Include full mipmap chain sizes
4. **Platform-specific Adjustments**: GPU vendor-specific optimizations

### Integration Points

The texture accounting integrates with:
- **Resource Tracker**: Automatic lifecycle management
- **Shadow System**: CSM memory constraint validation  
- **Terrain System**: Height texture memory tracking
- **PBR Pipeline**: Material texture memory accounting

## References

- [WebGPU Texture Format Specifications](https://gpuweb.github.io/gpuweb/#texture-formats)
- [DirectX Texture Compression](https://docs.microsoft.com/en-us/windows/win32/direct3d11/texture-block-compression)
- [OpenGL Compressed Texture Formats](https://www.khronos.org/opengl/wiki/Compressed_Texture)
- [Vulkan Texture Compression Guide](https://github.com/KhronosGroup/Vulkan-Guide/blob/master/chapters/extensions/compression.md)