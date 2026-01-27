// src/import/building_materials.rs
// P4.2: Building material presets for PBR rendering
//
// Maps OSM building material tags to physically-based rendering parameters.
// Provides realistic defaults for common building materials.

use std::collections::HashMap;

/// PBR material properties for building surfaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuildingMaterial {
    /// Base color/albedo in linear RGB [0,1]
    pub albedo: [f32; 3],
    /// Surface roughness [0=mirror, 1=fully diffuse]
    pub roughness: f32,
    /// Metallic factor [0=dielectric, 1=metal]
    pub metallic: f32,
    /// Index of refraction for Fresnel calculations (default ~1.5 for glass/plastic)
    pub ior: f32,
    /// Emissive intensity (for lit windows, signs)
    pub emissive: f32,
}

impl Default for BuildingMaterial {
    fn default() -> Self {
        Self {
            albedo: [0.7, 0.7, 0.7], // Light gray concrete
            roughness: 0.6,
            metallic: 0.0,
            ior: 1.5,
            emissive: 0.0,
        }
    }
}

impl BuildingMaterial {
    /// Create a new building material with specified properties.
    pub fn new(albedo: [f32; 3], roughness: f32, metallic: f32) -> Self {
        Self {
            albedo,
            roughness,
            metallic,
            ior: 1.5,
            emissive: 0.0,
        }
    }

    /// Create material with emissive component (for windows at night).
    pub fn with_emissive(mut self, emissive: f32) -> Self {
        self.emissive = emissive;
        self
    }

    /// Create material with custom IOR.
    pub fn with_ior(mut self, ior: f32) -> Self {
        self.ior = ior;
        self
    }
}

// ============================================================================
// Material Presets
// ============================================================================

/// Common brick material (reddish-brown)
pub const MATERIAL_BRICK: BuildingMaterial = BuildingMaterial {
    albedo: [0.55, 0.25, 0.18],
    roughness: 0.75,
    metallic: 0.0,
    ior: 1.5,
    emissive: 0.0,
};

/// Old/weathered brick
pub const MATERIAL_BRICK_OLD: BuildingMaterial = BuildingMaterial {
    albedo: [0.45, 0.22, 0.15],
    roughness: 0.85,
    metallic: 0.0,
    ior: 1.5,
    emissive: 0.0,
};

/// Plain concrete
pub const MATERIAL_CONCRETE: BuildingMaterial = BuildingMaterial {
    albedo: [0.6, 0.58, 0.55],
    roughness: 0.7,
    metallic: 0.0,
    ior: 1.5,
    emissive: 0.0,
};

/// Weathered/stained concrete
pub const MATERIAL_CONCRETE_WEATHERED: BuildingMaterial = BuildingMaterial {
    albedo: [0.45, 0.42, 0.4],
    roughness: 0.8,
    metallic: 0.0,
    ior: 1.5,
    emissive: 0.0,
};

/// Glass facade (highly reflective, low roughness)
pub const MATERIAL_GLASS: BuildingMaterial = BuildingMaterial {
    albedo: [0.04, 0.04, 0.05],
    roughness: 0.1,
    metallic: 0.0,
    ior: 1.52,
    emissive: 0.0,
};

/// Steel/metal cladding
pub const MATERIAL_STEEL: BuildingMaterial = BuildingMaterial {
    albedo: [0.56, 0.57, 0.58],
    roughness: 0.35,
    metallic: 0.9,
    ior: 2.5,
    emissive: 0.0,
};

/// Aluminum siding
pub const MATERIAL_ALUMINUM: BuildingMaterial = BuildingMaterial {
    albedo: [0.91, 0.92, 0.92],
    roughness: 0.3,
    metallic: 0.95,
    ior: 1.44,
    emissive: 0.0,
};

/// Wood siding (natural)
pub const MATERIAL_WOOD: BuildingMaterial = BuildingMaterial {
    albedo: [0.5, 0.35, 0.2],
    roughness: 0.7,
    metallic: 0.0,
    ior: 1.5,
    emissive: 0.0,
};

/// Painted wood (white)
pub const MATERIAL_WOOD_PAINTED: BuildingMaterial = BuildingMaterial {
    albedo: [0.9, 0.9, 0.88],
    roughness: 0.5,
    metallic: 0.0,
    ior: 1.5,
    emissive: 0.0,
};

/// White plaster/stucco
pub const MATERIAL_PLASTER: BuildingMaterial = BuildingMaterial {
    albedo: [0.88, 0.86, 0.82],
    roughness: 0.65,
    metallic: 0.0,
    ior: 1.5,
    emissive: 0.0,
};

/// Stone (limestone)
pub const MATERIAL_STONE: BuildingMaterial = BuildingMaterial {
    albedo: [0.65, 0.6, 0.5],
    roughness: 0.6,
    metallic: 0.0,
    ior: 1.5,
    emissive: 0.0,
};

/// Sandstone
pub const MATERIAL_SANDSTONE: BuildingMaterial = BuildingMaterial {
    albedo: [0.76, 0.65, 0.45],
    roughness: 0.7,
    metallic: 0.0,
    ior: 1.5,
    emissive: 0.0,
};

/// Granite
pub const MATERIAL_GRANITE: BuildingMaterial = BuildingMaterial {
    albedo: [0.35, 0.33, 0.32],
    roughness: 0.4,
    metallic: 0.0,
    ior: 1.5,
    emissive: 0.0,
};

/// Marble
pub const MATERIAL_MARBLE: BuildingMaterial = BuildingMaterial {
    albedo: [0.92, 0.9, 0.88],
    roughness: 0.25,
    metallic: 0.0,
    ior: 1.5,
    emissive: 0.0,
};

/// Roof tiles (terracotta)
pub const MATERIAL_ROOF_TILES: BuildingMaterial = BuildingMaterial {
    albedo: [0.6, 0.3, 0.15],
    roughness: 0.7,
    metallic: 0.0,
    ior: 1.5,
    emissive: 0.0,
};

/// Metal roof (galvanized)
pub const MATERIAL_ROOF_METAL: BuildingMaterial = BuildingMaterial {
    albedo: [0.6, 0.6, 0.62],
    roughness: 0.4,
    metallic: 0.7,
    ior: 2.0,
    emissive: 0.0,
};

/// Asphalt shingles
pub const MATERIAL_ROOF_SHINGLES: BuildingMaterial = BuildingMaterial {
    albedo: [0.25, 0.25, 0.27],
    roughness: 0.9,
    metallic: 0.0,
    ior: 1.5,
    emissive: 0.0,
};

/// Slate roof
pub const MATERIAL_ROOF_SLATE: BuildingMaterial = BuildingMaterial {
    albedo: [0.3, 0.32, 0.35],
    roughness: 0.5,
    metallic: 0.0,
    ior: 1.5,
    emissive: 0.0,
};

// ============================================================================
// Material Inference from OSM Tags
// ============================================================================

/// Infer building material from OSM tags.
///
/// Checks common OSM material tags in priority order:
/// 1. `building:material` (explicit facade material)
/// 2. `building:facade:material` (alternative tag)
/// 3. `material` (general material tag)
/// 4. `building` type (heuristic inference)
///
/// # Example
/// ```
/// use forge3d::import::building_materials::{material_from_tags, MATERIAL_BRICK};
/// use std::collections::HashMap;
///
/// let mut tags = HashMap::new();
/// tags.insert("building:material".to_string(), "brick".to_string());
/// let mat = material_from_tags(&tags);
/// assert_eq!(mat.albedo, MATERIAL_BRICK.albedo);
/// ```
pub fn material_from_tags(tags: &HashMap<String, String>) -> BuildingMaterial {
    // Priority 1: Explicit material tags
    if let Some(mat_str) = tags.get("building:material")
        .or_else(|| tags.get("building:facade:material"))
        .or_else(|| tags.get("material"))
    {
        return material_from_name(mat_str);
    }

    // Priority 2: Color-based override
    if let Some(color) = tags.get("building:colour")
        .or_else(|| tags.get("building:color"))
    {
        if let Some(rgb) = parse_css_color(color) {
            // Return default material with custom color
            return BuildingMaterial {
                albedo: rgb,
                ..BuildingMaterial::default()
            };
        }
    }

    // Priority 3: Infer from building type
    if let Some(building_type) = tags.get("building") {
        return match building_type.to_lowercase().as_str() {
            // Glass facades common in commercial
            "commercial" | "office" | "retail" | "skyscraper" => MATERIAL_GLASS,

            // Industrial typically concrete/metal
            "industrial" | "warehouse" | "hangar" => MATERIAL_CONCRETE,

            // Residential varies by region, default to brick
            "house" | "detached" | "semidetached_house" | "terrace"
            | "residential" => MATERIAL_BRICK,

            // Apartments often concrete
            "apartments" => MATERIAL_CONCRETE,

            // Historic buildings often stone
            "church" | "cathedral" | "castle" | "palace" | "monument" => MATERIAL_STONE,

            // Farm buildings often wood
            "barn" | "farm" | "cabin" | "shed" => MATERIAL_WOOD,

            // Public buildings vary
            "public" | "civic" | "government" => MATERIAL_STONE,

            "hotel" => MATERIAL_GLASS,

            _ => BuildingMaterial::default(),
        };
    }

    BuildingMaterial::default()
}

/// Convert a material name string to a BuildingMaterial.
pub fn material_from_name(name: &str) -> BuildingMaterial {
    match name.to_lowercase().trim() {
        "brick" | "bricks" => MATERIAL_BRICK,
        "brick_old" | "old_brick" | "weathered_brick" => MATERIAL_BRICK_OLD,
        "concrete" | "cement" => MATERIAL_CONCRETE,
        "concrete_weathered" | "weathered_concrete" => MATERIAL_CONCRETE_WEATHERED,
        "glass" => MATERIAL_GLASS,
        "steel" | "metal" => MATERIAL_STEEL,
        "aluminium" | "aluminum" => MATERIAL_ALUMINUM,
        "wood" | "timber" => MATERIAL_WOOD,
        "wood_painted" | "painted_wood" => MATERIAL_WOOD_PAINTED,
        "plaster" | "stucco" | "render" => MATERIAL_PLASTER,
        "stone" | "limestone" => MATERIAL_STONE,
        "sandstone" => MATERIAL_SANDSTONE,
        "granite" => MATERIAL_GRANITE,
        "marble" => MATERIAL_MARBLE,
        "tiles" | "roof_tiles" | "terracotta" => MATERIAL_ROOF_TILES,
        "roof_metal" | "metal_roof" => MATERIAL_ROOF_METAL,
        "shingles" | "asphalt" => MATERIAL_ROOF_SHINGLES,
        "slate" => MATERIAL_ROOF_SLATE,
        _ => BuildingMaterial::default(),
    }
}

/// Infer roof material from OSM tags.
pub fn roof_material_from_tags(tags: &HashMap<String, String>) -> BuildingMaterial {
    if let Some(mat_str) = tags.get("roof:material")
        .or_else(|| tags.get("building:roof:material"))
    {
        return material_from_name(mat_str);
    }

    // Infer from roof color
    if let Some(color) = tags.get("roof:colour")
        .or_else(|| tags.get("roof:color"))
    {
        if let Some(rgb) = parse_css_color(color) {
            return BuildingMaterial {
                albedo: rgb,
                roughness: 0.6,
                ..BuildingMaterial::default()
            };
        }
    }

    // Default based on building type
    if let Some(building_type) = tags.get("building") {
        return match building_type.to_lowercase().as_str() {
            "house" | "detached" | "residential" => MATERIAL_ROOF_TILES,
            "industrial" | "warehouse" | "commercial" => MATERIAL_ROOF_METAL,
            "church" | "cathedral" => MATERIAL_ROOF_SLATE,
            _ => MATERIAL_ROOF_SHINGLES,
        };
    }

    MATERIAL_ROOF_SHINGLES
}

// ============================================================================
// Color Parsing Utilities
// ============================================================================

/// Parse a CSS color string to linear RGB [0,1].
/// Supports hex (#RGB, #RRGGBB) and named colors.
pub fn parse_css_color(color: &str) -> Option<[f32; 3]> {
    let c = color.trim().to_lowercase();

    // Named colors
    match c.as_str() {
        "white" => return Some([1.0, 1.0, 1.0]),
        "black" => return Some([0.0, 0.0, 0.0]),
        "red" => return Some([1.0, 0.0, 0.0]),
        "green" => return Some([0.0, 0.5, 0.0]),
        "blue" => return Some([0.0, 0.0, 1.0]),
        "yellow" => return Some([1.0, 1.0, 0.0]),
        "gray" | "grey" => return Some([0.5, 0.5, 0.5]),
        "brown" => return Some([0.6, 0.3, 0.1]),
        "beige" => return Some([0.96, 0.96, 0.86]),
        "tan" => return Some([0.82, 0.71, 0.55]),
        "cream" => return Some([1.0, 0.99, 0.82]),
        "orange" => return Some([1.0, 0.65, 0.0]),
        "pink" => return Some([1.0, 0.75, 0.8]),
        _ => {}
    }

    // Hex colors
    if c.starts_with('#') {
        let hex = &c[1..];
        return parse_hex_rgb(hex);
    }

    None
}

fn parse_hex_rgb(hex: &str) -> Option<[f32; 3]> {
    let (r, g, b) = match hex.len() {
        3 => {
            let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
            let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
            let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
            (r, g, b)
        }
        6 => {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            (r, g, b)
        }
        _ => return None,
    };

    // Convert sRGB to linear (approximate gamma=2.2)
    Some([
        srgb_to_linear(r),
        srgb_to_linear(g),
        srgb_to_linear(b),
    ])
}

fn srgb_to_linear(srgb: u8) -> f32 {
    let s = srgb as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

// ============================================================================
// Python Bindings
// ============================================================================

#[cfg(feature = "extension-module")]
use pyo3::{prelude::*, exceptions::PyValueError};

/// P4.2: Python binding for material inference from OSM tags
#[cfg(feature = "extension-module")]
#[pyfunction(signature = (tags_json))]
pub fn material_from_tags_py(tags_json: &str) -> PyResult<PyObject> {
    let tags: HashMap<String, String> = serde_json::from_str(tags_json)
        .map_err(|e| PyValueError::new_err(format!("invalid JSON: {e}")))?;

    let mat = material_from_tags(&tags);

    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("albedo", (mat.albedo[0], mat.albedo[1], mat.albedo[2]))?;
        dict.set_item("roughness", mat.roughness)?;
        dict.set_item("metallic", mat.metallic)?;
        dict.set_item("ior", mat.ior)?;
        dict.set_item("emissive", mat.emissive)?;
        Ok(dict.into())
    })
}

/// P4.2: Python binding for material by name
#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn material_from_name_py(name: &str) -> PyObject {
    let mat = material_from_name(name);

    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new_bound(py);
        let _ = dict.set_item("albedo", (mat.albedo[0], mat.albedo[1], mat.albedo[2]));
        let _ = dict.set_item("roughness", mat.roughness);
        let _ = dict.set_item("metallic", mat.metallic);
        let _ = dict.set_item("ior", mat.ior);
        let _ = dict.set_item("emissive", mat.emissive);
        dict.into()
    })
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_material_from_name() {
        let mat = material_from_name("brick");
        assert_eq!(mat.albedo, MATERIAL_BRICK.albedo);
        assert_eq!(mat.roughness, MATERIAL_BRICK.roughness);
    }

    #[test]
    fn test_material_from_tags_explicit() {
        let mut tags = HashMap::new();
        tags.insert("building:material".to_string(), "glass".to_string());
        let mat = material_from_tags(&tags);
        assert_eq!(mat.albedo, MATERIAL_GLASS.albedo);
    }

    #[test]
    fn test_material_from_building_type() {
        let mut tags = HashMap::new();
        tags.insert("building".to_string(), "warehouse".to_string());
        let mat = material_from_tags(&tags);
        assert_eq!(mat.albedo, MATERIAL_CONCRETE.albedo);
    }

    #[test]
    fn test_parse_hex_color() {
        let rgb = parse_css_color("#ff0000").unwrap();
        assert!(rgb[0] > 0.9);
        assert!(rgb[1] < 0.1);
        assert!(rgb[2] < 0.1);
    }

    #[test]
    fn test_parse_named_color() {
        let rgb = parse_css_color("white").unwrap();
        assert_eq!(rgb, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_color_override() {
        let mut tags = HashMap::new();
        tags.insert("building:colour".to_string(), "#8B4513".to_string()); // saddle brown
        let mat = material_from_tags(&tags);
        assert!(mat.albedo[0] > mat.albedo[1]);
        assert!(mat.albedo[1] > mat.albedo[2]);
    }
}
