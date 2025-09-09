# forge3d Public API Policy

## Overview

The forge3d Python API is designed with stability, clarity, and ease-of-use in mind. This document defines the public API structure and policies for module organization.

## Public API Structure

### Core Module (`forge3d`)
The main module exports only stable, commonly-used functionality:

```python
import forge3d as f3d

# Core rendering
renderer = f3d.Renderer(800, 600)
scene = f3d.Scene(800, 600)

# Terrain
terrain = f3d.make_terrain(128, 128, 64)
stats = f3d.dem_stats(height_data)

# Vector graphics
f3d.add_points_py(points, colors=colors)
f3d.add_lines_py(lines, colors=colors)

# Image utilities
image = f3d.png_to_numpy("texture.png")
f3d.numpy_to_png("output.png", image)

# Device utilities
has_gpu = f3d.has_gpu()
adapters = f3d.enumerate_adapters()
```

### Specialized Submodules
Advanced functionality is available via explicit submodule imports:

```python
# PBR Materials (recommended)
import forge3d.pbr as pbr
material = pbr.PbrMaterial(base_color=(0.8, 0.2, 0.2, 1.0), metallic=0.0, roughness=0.7)

# Shadow Mapping (if available)
import forge3d.shadows as shadows
config = shadows.get_preset_config('high_quality')

# Legacy compatibility
import forge3d.materials as mat  # Re-exports forge3d.pbr
```

## API Stability Levels

### Stable (Public API)
- Exported in `forge3d.__all__`
- Semantic versioning guarantees
- Deprecation warnings before removal
- Examples: `Renderer`, `Scene`, `png_to_numpy`, `PbrMaterial`

### Experimental (Submodule)
- Available via explicit submodule import
- May change between minor versions
- Clear documentation of stability level
- Examples: Advanced rendering features, new algorithms

### Internal (Not Exported)
- Not in `__all__` exports
- No stability guarantees
- Subject to change without notice
- Examples: Test functions, internal utilities

## Materials Module Policy

### Primary Implementation: `forge3d.pbr`
- Contains all PBR materials functionality
- Stable public API
- Comprehensive documentation
- Recommended for all new code

### Compatibility Shim: `forge3d.materials`
- Re-exports everything from `forge3d.pbr`
- Maintained for backward compatibility
- Legacy code continues to work
- Not recommended for new projects

### Migration Path
1. **Existing Code**: No changes required, `forge3d.materials` continues to work
2. **New Code**: Use `forge3d.pbr` directly for better clarity
3. **Future**: `forge3d.materials` will be maintained indefinitely for compatibility

## Submodule Import Policy

### Lightweight Core
The main `forge3d` module imports only essential functionality to minimize startup time and dependencies.

### Explicit Imports
Advanced features require explicit imports:
- Keeps the core API clean and focused
- Clear separation of concerns
- Optional features can be easily identified
- Better error handling when features are unavailable

### Feature Detection
```python
import forge3d as f3d

# Check for GPU support
if f3d.has_gpu():
    # Use GPU-accelerated features
    pass

# Check for shadows support
try:
    import forge3d.shadows as shadows
    if shadows.has_shadows_support():
        # Use shadow mapping
        pass
except ImportError:
    # Shadows not available
    pass

# Check for PBR support
try:
    import forge3d.pbr as pbr
    if pbr.has_pbr_support():
        # Use PBR materials
        pass
except ImportError:
    # PBR not available
    pass
```

## API Organization Principles

1. **Discoverability**: Common operations are in the main module
2. **Clarity**: Function names clearly indicate their purpose
3. **Consistency**: Similar operations follow similar patterns
4. **Extensibility**: New features can be added without breaking existing code
5. **Performance**: Heavy imports are deferred to submodules
6. **Compatibility**: Legacy APIs are maintained with clear migration paths

## Version Policy

- **Major versions** (X.0.0): Breaking changes allowed
- **Minor versions** (1.X.0): New features, backward compatible
- **Patch versions** (1.1.X): Bug fixes only

Deprecated features will be marked with warnings for at least one major version before removal.

## Testing and Examples

All public API functions must have:
- Comprehensive test coverage
- Clear documentation with examples  
- Error handling for invalid inputs
- Graceful fallbacks when features are unavailable

This policy ensures forge3d provides a stable, predictable API that grows gracefully over time while maintaining backward compatibility.