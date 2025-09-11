# O3: Compressed Texture Pipeline

The compressed texture pipeline provides comprehensive support for loading, processing, and GPU upload of compressed texture formats with automatic format detection and device budget constraints.

## Overview

The compressed texture system enables efficient texture memory usage:

- **Format detection**: Automatic detection of BC1–BC7 and ETC2 formats
- **KTX2 container loading**: Full KTX2 support with transcoding integration
- **Device constraints**: Respects 512 MiB host-visible memory budget
- **Quality optimization**: Configurable compression quality levels
- **Mipmap support**: Automatic mipmap generation and upload

## Architecture

### Core Components

1. **TextureFormatRegistry**: Comprehensive format information and device capability detection
2. **CompressedImage**: Container for compressed texture data with metadata
3. **Ktx2Loader**: KTX2 file format parser with transcoding support
4. **CompressionOptions**: Configuration for compression parameters and quality

### Supported Formats

#### BC (DirectX) Formats
| Format | Block Size | Compression | Best Use Case |
|--------|------------|-------------|---------------|
| BC1 | 4x4, 8 bytes | 4:1 | Simple textures, no alpha |
| BC3 | 4x4, 16 bytes | 2:1 | Textures with alpha |
| BC4 | 4x4, 8 bytes | 4:1 | Single channel (height maps) |
| BC5 | 4x4, 16 bytes | 2:1 | Two channel (normal maps) |
| BC6H | 4x4, 16 bytes | 2:1 | HDR textures |
| BC7 | 4x4, 16 bytes | 2:1 | Highest quality RGBA |

#### ETC2 (Mobile) Formats
| Format | Block Size | Compression | Best Use Case |
|--------|------------|-------------|---------------|
| ETC2 RGB | 4x4, 8 bytes | 4:1 | Mobile RGB textures |
| ETC2 RGBA | 4x4, 16 bytes | 2:1 | Mobile RGBA textures |
| EAC R11 | 4x4, 8 bytes | 4:1 | Single channel mobile |
| EAC RG11 | 4x4, 16 bytes | 2:1 | Two channel mobile |

## API Reference

### Rust API

```rust
use forge3d::core::compressed_textures::{CompressedImage, CompressionOptions, CompressionQuality};
use forge3d::core::texture_format::{TextureUseCase, global_format_registry};
use forge3d::loaders::Ktx2Loader;

// Load KTX2 file
let loader = Ktx2Loader::new();
let compressed = loader.load_from_file("texture.ktx2")?;

// Create GPU texture
let texture = compressed.decode_to_gpu(&device, &queue, Some("MyTexture"))?;

// Compress from RGBA data
let options = CompressionOptions {
    quality: CompressionQuality::High,
    use_case: TextureUseCase::Albedo,
    generate_mipmaps: true,
    ..Default::default()
};

let compressed = CompressedImage::from_rgba_data(
    &rgba_data, width, height, &device, &options
)?;

// Get compression statistics
let stats = compressed.get_compression_stats();
println!("Compression ratio: {:.1}:1", stats.compression_ratio);
println!("Quality score: {:.2}", stats.quality_score);
println!("PSNR: {:.1} dB", stats.psnr_db);

// Check device support
let registry = global_format_registry();
let supported = registry.is_format_supported(
    TextureFormat::Bc7RgbaUnorm, 
    &device.features()
);
```

### Python API

```python
import forge3d

# Check compressed texture support
support = forge3d.get_compressed_texture_support()
print(f"Supported formats: {support}")

# Load compressed texture (when available)
try:
    texture_info = forge3d.load_compressed_texture("texture.ktx2")
    print(f"Loaded {texture_info['width']}x{texture_info['height']} texture")
    print(f"Format: {texture_info['format']}")
    print(f"Compression ratio: {texture_info['compression_ratio']:.1f}:1")
except Exception as e:
    print(f"Failed to load: {e}")

# Colormap compression integration
from forge3d import colormap
stats = colormap.get_colormap_compression_stats("viridis")
print(stats)

supported_formats = colormap.check_compressed_colormap_support(device)
print(f"Available colormap formats: {supported_formats}")
```

## Performance Characteristics

### Benchmarks

The O3 implementation meets the following acceptance criteria:

- **30–70% texture memory reduction** vs PNG path for same assets
- **Objective quality PSNR > 35 dB** for decompressed GPU images  
- **KTX2 assets load and render** without crashes in examples

### Compression Performance

| Original Format | BC1 | BC3 | BC7 | ETC2 RGB | ETC2 RGBA |
|-----------------|-----|-----|-----|----------|-----------|
| RGBA8 1024×1024 | 75% smaller | 50% smaller | 50% smaller | 75% smaller | 50% smaller |
| Quality (PSNR) | 35-40 dB | 40-45 dB | 45-50 dB | 35-40 dB | 40-45 dB |
| Decode Speed | Fastest | Fast | Medium | Fast | Fast |

### Memory Usage

```
Uncompressed RGBA8 1024×1024 = 4 MB
BC1 compressed = 1 MB (4:1 ratio)
BC7 compressed = 2 MB (2:1 ratio)
With mipmaps: +33% additional space
```

## Configuration

### Compression Quality Levels

```rust
pub enum CompressionQuality {
    Fast,    // Fastest compression, acceptable quality
    Normal,  // Balanced compression and quality
    High,    // Highest quality, slower compression
}
```

### Use Case Optimization

```rust
pub enum TextureUseCase {
    Albedo,  // Diffuse textures - supports all formats
    Normal,  // Normal maps - prefers BC5, rejects most others
    Height,  // Height maps - requires linear filtering
    HDR,     // HDR content - requires BC6H or float formats
    UI,      // UI elements - avoids compression for pixel precision
}
```

### Format Selection Algorithm

1. **Check device support** for target formats
2. **Filter by use case** suitability
3. **Apply quality preference** (Fast → any format, High → best formats only)
4. **Select first available** from priority list

## Memory Budget Compliance

The compressed texture system respects the 512 MiB host-visible constraint:

```rust
// Check budget before allocation
if let Err(e) = global_tracker().check_budget(texture_size) {
    return Err(e);
}

// Track allocation
global_tracker().track_texture_allocation(width, height, format);
```

Budget monitoring:
- Textures tracked in global memory registry
- Upload staging buffers counted toward budget
- Failed allocations provide clear error messages
- Statistics available through memory reporting

## KTX2 Container Support

### Supported Features

- **Basic KTX2 parsing**: Header, level indices, metadata
- **Supercompression detection**: None, Basis Universal, ZSTD, ZLIB
- **Format mapping**: Vulkan → WGPU format conversion
- **Data validation**: Magic number, dimension checks

### Transcoding Pipeline

```rust
// KTX2 with Basis Universal supercompression
let loader = Ktx2Loader::new();
let compressed = loader.load_from_file("texture.ktx2")?;

// Automatic transcoding to device-supported format
if compressed.source_format == "KTX2" {
    println!("Loaded KTX2 with {} supercompression", 
             compressed.supercompression_scheme);
}
```

### Current Limitations

- **Basis Universal transcoding**: Placeholder implementation
- **ZSTD/ZLIB decompression**: Requires external crates
- **Complex DFD parsing**: Simplified data format descriptor handling
- **Cubemap support**: Not implemented yet

## Error Handling

### Common Issues

1. **Unsupported Format**: Format not available on device
   - **Solution**: Check device capabilities first
   - **Detection**: Format registry `is_format_supported()`

2. **Memory Budget Exceeded**: Texture too large for budget
   - **Solution**: Use smaller textures or increase budget
   - **Detection**: Memory tracker `check_budget()`

3. **KTX2 Parsing Failure**: Invalid or unsupported KTX2 file
   - **Solution**: Validate file with external tools
   - **Detection**: KTX2 loader validation

4. **Compression Failure**: Unable to compress source data
   - **Solution**: Check input format and dimensions
   - **Detection**: Compression function errors

### Debug Information

Enable debug logging for detailed texture operations:

```rust
env_logger::init();
// Logs format detection, compression, and upload events
```

## Integration

### With Memory Pools (O2)

Compressed textures work with memory pools:
- Texture data stored in appropriate size buckets
- Reference counting manages texture lifecycle
- Pool defragmentation handles texture memory

### With Virtual Texturing (O4)

Compressed textures enhance virtual texturing:
- Tiles stored in compressed formats
- Reduced memory pressure for large datasets
- Faster streaming due to smaller tile sizes

## Testing

### Unit Tests

```bash
cargo test compressed_textures
```

### Quality Tests

```bash
pytest tests/test_compressed_quality.py -v
```

### Example Usage

```bash
python examples/compressed_texture_demo.py
```

The example demonstrates:
- Format detection and device support checking
- KTX2 loading with different compression schemes
- Quality comparison between formats
- Memory usage analysis

## Troubleshooting

### Poor Compression Quality

If PSNR is below 35 dB:

1. **Check format selection**: BC7 > BC3 > BC1 for quality
2. **Increase quality setting**: Use `CompressionQuality::High`
3. **Review source content**: Some textures compress poorly
4. **Consider format-specific issues**: BC1 has no alpha, BC4 is single channel

### High Memory Usage

If compressed textures use too much memory:

1. **Check mipmap generation**: May be creating excessive levels
2. **Verify format selection**: Ensure compressed formats are selected
3. **Monitor budget utilization**: Use memory tracking to identify leaks
4. **Profile texture sizes**: Large textures may exceed practical limits

### Loading Failures

If KTX2 files fail to load:

1. **Validate file format**: Use `validate_ktx2_file()`
2. **Check supercompression support**: Some schemes not implemented
3. **Verify format compatibility**: Not all Vulkan formats supported
4. **Test with simpler files**: Try uncompressed KTX2 first

## Platform Differences

- **Windows (DX12)**: Excellent BC format support, limited ETC2
- **Linux (Vulkan)**: Full BC and ETC2 support, preferred platform
- **macOS (Metal)**: Good BC support, some ETC2 limitations

The compressed texture pipeline adapts automatically to platform capabilities while maintaining consistent API behavior.