// T33-BEGIN:colormap-registry
//! Central colormap registry.
//! - Single source for supported names
//! - Embedded 256×1 PNG bytes via `include_bytes!`
//! - Small helpers (enum mapping + PyO3 error)

/// Built-in colormap names (case-sensitive).
pub const SUPPORTED: &[&str] = &["viridis", "magma", "terrain"];

/// Resolve embedded 256×1 PNG bytes for the given name.
pub fn resolve_bytes(name: &str) -> Result<&'static [u8], &'static str> {
    match name {
        "viridis" => Ok(include_bytes!("../assets/colormaps/viridis_256x1.png")),
        "magma"   => Ok(include_bytes!("../assets/colormaps/magma_256x1.png")),
        "terrain" => Ok(include_bytes!("../assets/colormaps/terrain_256x1.png")),
        _ => Err("unknown"),
    }
}

/// Optional typed mapping if you keep a ColormapType in your pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColormapType { Viridis, Magma, Terrain }

pub fn map_name_to_type(name: &str) -> Result<ColormapType, String> {
    match name {
        "viridis" => Ok(ColormapType::Viridis),
        "magma"   => Ok(ColormapType::Magma),
        "terrain" => Ok(ColormapType::Terrain),
        _ => Err(format!("Unknown colormap '{}'. Supported: {}", name, SUPPORTED.join(", "))),
    }
}

/// PyO3-friendly error helper (always compiled; crate already depends on pyo3).
pub fn py_err_unknown(name: &str) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(
        format!("Unknown colormap '{}'. Supported: {}", name, SUPPORTED.join(", "))
    )
}
// T33-END:colormap-registry
