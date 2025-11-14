// src/lighting/types.rs
// P0 lighting type definitions with GPU-aligned layouts
// All types are repr(C) and bytemuck-compatible for GPU upload

use bytemuck::{Pod, Zeroable};

/// Light type enumeration (P0/P1)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    Directional = 0,
    Point = 1,
    Spot = 2,
    Environment = 3,
    AreaRect = 4,   // P1: Rectangular area light
    AreaDisk = 5,   // P1: Disk area light
    AreaSphere = 6, // P1: Spherical area light
}

impl LightType {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// BRDF model enumeration (P0/P2)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrdfModel {
    Lambert = 0,
    Phong = 1,
    BlinnPhong = 2,
    OrenNayar = 3,
    CookTorranceGgx = 4,
    CookTorranceBeckmann = 5,
    DisneyPrincipled = 6,
    AshikhminShirley = 7,
    Ward = 8,
    Toon = 9,
    Minnaert = 10,
    Subsurface = 11,
    Hair = 12,
}

impl BrdfModel {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Shadow technique enumeration (P0/P3)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowTechnique {
    Hard = 0,
    PCF = 1,
    PCSS = 2, // P3: Percentage Closer Soft Shadows
    VSM = 3,  // P3: Variance Shadow Maps
    EVSM = 4, // P3: Exponential Variance Shadow Maps
    MSM = 5,  // P3: Moment Shadow Maps
    CSM = 6,  // P3: Cascaded Shadow Maps (can combine with others)
}

impl ShadowTechnique {
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn name(&self) -> &'static str {
        match self {
            ShadowTechnique::Hard => "hard",
            ShadowTechnique::PCF => "pcf",
            ShadowTechnique::PCSS => "pcss",
            ShadowTechnique::VSM => "vsm",
            ShadowTechnique::EVSM => "evsm",
            ShadowTechnique::MSM => "msm",
            ShadowTechnique::CSM => "csm",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "hard" => Some(ShadowTechnique::Hard),
            "pcf" => Some(ShadowTechnique::PCF),
            "pcss" => Some(ShadowTechnique::PCSS),
            "vsm" => Some(ShadowTechnique::VSM),
            "evsm" => Some(ShadowTechnique::EVSM),
            "msm" => Some(ShadowTechnique::MSM),
            "csm" => Some(ShadowTechnique::CSM),
            _ => None,
        }
    }

    /// Returns true if technique requires moment textures (VSM/EVSM/MSM)
    pub fn requires_moments(&self) -> bool {
        matches!(
            self,
            ShadowTechnique::VSM | ShadowTechnique::EVSM | ShadowTechnique::MSM
        )
    }

    /// Returns number of channels needed (1 for depth, 2 for VSM, 4 for EVSM/MSM)
    pub fn channels(&self) -> u32 {
        match self {
            ShadowTechnique::VSM => 2,
            ShadowTechnique::EVSM | ShadowTechnique::MSM => 4,
            _ => 1,
        }
    }
}

/// Global illumination technique enumeration (P0)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GiTechnique {
    None = 0,
    Ibl = 1,
}

impl GiTechnique {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Light configuration (P1 extended)
/// GPU-aligned struct for SSBO upload (std430 layout)
///
/// # Layout Parity (P1-01)
///
/// This struct MUST maintain exact binary layout parity with WGSL `LightGPU` in
/// `src/shaders/lights.wgsl` for correct GPU buffer uploads.
///
/// **Size**: 80 bytes (5 vec4s)  
/// **Alignment**: 16 bytes (vec4 boundary)  
/// **Traits**: `Pod` + `Zeroable` (bytemuck) for safe byte casting
///
/// ## Memory Layout
///
/// ```text
/// Offset | Field                | Type      | Size  | Notes
/// -------|----------------------|-----------|-------|------------------------
///   0-15 | Vec4 #1              |           |  16   |
///      0 |   kind               | u32       |   4   | LightType enum
///      4 |   intensity          | f32       |   4   |
///      8 |   range              | f32       |   4   | Unused for directional
///     12 |   env_texture_index  | u32       |   4   | Only for Environment
///  16-31 | Vec4 #2              |           |  16   |
///     16 |   color              | [f32; 3]  |  12   | RGB
///     28 |   _pad1              | f32       |   4   |
///  32-47 | Vec4 #3              |           |  16   |
///     32 |   pos_ws             | [f32; 3]  |  12   | Unused for directional
///     44 |   _pad2              | f32       |   4   |
///  48-63 | Vec4 #4              |           |  16   |
///     48 |   dir_ws             | [f32; 3]  |  12   | Unused for point
///     60 |   _pad3              | f32       |   4   |
///  64-79 | Vec4 #5              |           |  16   |
///     64 |   cone_cos           | [f32; 2]  |   8   | Spot: [inner, outer]
///     72 |   area_half          | [f32; 2]  |   8   | Area: extents or radius
/// ```
///
/// ## LightType Values (must match WGSL constants)
///
/// - Directional = 0 (LIGHT_DIRECTIONAL)
/// - Point = 1 (LIGHT_POINT)
/// - Spot = 2 (LIGHT_SPOT)
/// - Environment = 3 (LIGHT_ENVIRONMENT)
/// - AreaRect = 4 (LIGHT_AREA_RECT)
/// - AreaDisk = 5 (LIGHT_AREA_DISK)
/// - AreaSphere = 6 (LIGHT_AREA_SPHERE)
///
/// ## Field Usage by Light Type
///
/// | Field            | Directional | Point | Spot | Environment | Area (Rect/Disk/Sphere) |
/// |------------------|-------------|-------|------|-------------|-------------------------|
/// | kind             | ✓           | ✓     | ✓    | ✓           | ✓                       |
/// | intensity        | ✓           | ✓     | ✓    | ✓           | ✓                       |
/// | range            | ✗           | ✓     | ✓    | ✗           | ✓                       |
/// | env_texture_index| ✗           | ✗     | ✗    | ✓           | ✗                       |
/// | color            | ✓           | ✓     | ✓    | ✓           | ✓                       |
/// | pos_ws           | ✗           | ✓     | ✓    | ✗           | ✓                       |
/// | dir_ws           | ✓           | ✗     | ✓    | ✗           | ✓ (rect/disk normal)    |
/// | cone_cos         | ✗           | ✗     | ✓    | ✗           | ✗                       |
/// | area_half        | ✗           | ✗     | ✗    | ✗           | ✓                       |
///
/// ## Verification
///
/// Unit tests in `#[cfg(test)] mod tests` verify:
/// - Struct size is exactly 80 bytes
/// - Field offsets match expected std430 layout
/// - Enum values match WGSL constants
/// - Pod/Zeroable traits work correctly
/// - Constructor functions initialize unused fields consistently
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Light {
    // Vec4 #1: type, intensity, range, env_texture_index
    /// Light type (0=Directional, 1=Point, 2=Spot, 3=Environment, 4=AreaRect, 5=AreaDisk, 6=AreaSphere)
    pub kind: u32,
    /// Light intensity multiplier
    pub intensity: f32,
    /// Range for point/spot/area lights (unused for directional/environment)
    pub range: f32,
    /// Environment texture index (for LightType::Environment, unused otherwise)
    pub env_texture_index: u32,

    // Vec4 #2: color (RGB + padding)
    /// Light color (RGB)
    pub color: [f32; 3],
    pub _pad1: f32,

    // Vec4 #3: position in world space (for point/spot/area lights)
    /// Position in world space (unused for directional)
    pub pos_ws: [f32; 3],
    pub _pad2: f32,

    // Vec4 #4: direction in world space (for directional/spot lights)
    /// Direction in world space (normalized, unused for point lights)
    pub dir_ws: [f32; 3],
    pub _pad3: f32,

    // Vec4 #5: cone cosines + area half-extents
    /// Spot cone cosines: [cos(inner_angle), cos(outer_angle)]
    pub cone_cos: [f32; 2],
    /// Area light half-extents: [half_width, half_height] for rect, [radius, 0] for disk/sphere
    pub area_half: [f32; 2],
}

impl Default for Light {
    fn default() -> Self {
        Self {
            kind: LightType::Directional.as_u32(),
            intensity: 3.0,
            range: 1000.0,
            env_texture_index: 0,
            color: [1.0, 1.0, 1.0],
            _pad1: 0.0,
            pos_ws: [0.0, 0.0, 0.0],
            _pad2: 0.0,
            dir_ws: [0.0, -1.0, 0.0], // Downward by default
            _pad3: 0.0,
            cone_cos: [0.939, 0.819], // ~20° inner, ~35° outer
            area_half: [1.0, 1.0],
        }
    }
}

impl Light {
    /// Create a directional light from azimuth and elevation angles
    pub fn directional(
        azimuth_deg: f32,
        elevation_deg: f32,
        intensity: f32,
        color: [f32; 3],
    ) -> Self {
        let az_rad = azimuth_deg.to_radians();
        let el_rad = elevation_deg.to_radians();

        // Convert spherical to Cartesian (pointing down for directional light)
        let x = el_rad.cos() * az_rad.sin();
        let y = -el_rad.sin();
        let z = -el_rad.cos() * az_rad.cos();

        Self {
            kind: LightType::Directional.as_u32(),
            intensity,
            range: 0.0,
            env_texture_index: 0,
            color,
            _pad1: 0.0,
            pos_ws: [0.0; 3], // Unused for directional
            _pad2: 0.0,
            dir_ws: [x, y, z],
            _pad3: 0.0,
            cone_cos: [1.0, 1.0], // Unused for directional
            area_half: [0.0, 0.0],
        }
    }

    /// Create a point light at a position
    pub fn point(position: [f32; 3], range: f32, intensity: f32, color: [f32; 3]) -> Self {
        Self {
            kind: LightType::Point.as_u32(),
            intensity,
            range,
            env_texture_index: 0,
            color,
            _pad1: 0.0,
            pos_ws: position,
            _pad2: 0.0,
            dir_ws: [0.0; 3], // Unused for point
            _pad3: 0.0,
            cone_cos: [1.0, 1.0],
            area_half: [0.0, 0.0],
        }
    }

    /// Create a spot light
    pub fn spot(
        position: [f32; 3],
        direction: [f32; 3],
        range: f32,
        inner_deg: f32,
        outer_deg: f32,
        intensity: f32,
        color: [f32; 3],
    ) -> Self {
        // Normalize direction
        let len = (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
        let dir_norm = if len > 0.0001 {
            [direction[0] / len, direction[1] / len, direction[2] / len]
        } else {
            [0.0, -1.0, 0.0] // Default downward
        };

        // Precompute cosines for GPU
        let inner_cos = inner_deg.to_radians().cos();
        let outer_cos = outer_deg.to_radians().cos();

        Self {
            kind: LightType::Spot.as_u32(),
            intensity,
            range,
            env_texture_index: 0,
            color,
            _pad1: 0.0,
            pos_ws: position,
            _pad2: 0.0,
            dir_ws: dir_norm,
            _pad3: 0.0,
            cone_cos: [inner_cos, outer_cos],
            area_half: [0.0, 0.0],
        }
    }

    /// Create an environment/IBL light
    pub fn environment(intensity: f32, texture_index: u32) -> Self {
        Self {
            kind: LightType::Environment.as_u32(),
            intensity,
            range: 0.0,
            env_texture_index: texture_index,
            color: [1.0, 1.0, 1.0],
            _pad1: 0.0,
            pos_ws: [0.0; 3],
            _pad2: 0.0,
            dir_ws: [0.0; 3],
            _pad3: 0.0,
            cone_cos: [1.0, 1.0],
            area_half: [0.0, 0.0],
        }
    }

    /// Create a rectangular area light (P1)
    pub fn area_rect(
        position: [f32; 3],
        direction: [f32; 3], // Normal direction
        half_width: f32,
        half_height: f32,
        intensity: f32,
        color: [f32; 3],
    ) -> Self {
        // Normalize direction
        let len = (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
        let dir_norm = if len > 0.0001 {
            [direction[0] / len, direction[1] / len, direction[2] / len]
        } else {
            [0.0, -1.0, 0.0]
        };

        Self {
            kind: LightType::AreaRect.as_u32(),
            intensity,
            range: 0.0, // Area lights typically have infinite range or separate falloff
            env_texture_index: 0,
            color,
            _pad1: 0.0,
            pos_ws: position,
            _pad2: 0.0,
            dir_ws: dir_norm,
            _pad3: 0.0,
            cone_cos: [1.0, 1.0],
            area_half: [half_width, half_height],
        }
    }

    /// Create a disk area light (P1)
    pub fn area_disk(
        position: [f32; 3],
        direction: [f32; 3], // Normal direction
        radius: f32,
        intensity: f32,
        color: [f32; 3],
    ) -> Self {
        let len = (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
        let dir_norm = if len > 0.0001 {
            [direction[0] / len, direction[1] / len, direction[2] / len]
        } else {
            [0.0, -1.0, 0.0]
        };

        Self {
            kind: LightType::AreaDisk.as_u32(),
            intensity,
            range: 0.0,
            env_texture_index: 0,
            color,
            _pad1: 0.0,
            pos_ws: position,
            _pad2: 0.0,
            dir_ws: dir_norm,
            _pad3: 0.0,
            cone_cos: [1.0, 1.0],
            area_half: [radius, 0.0], // Store radius in x component
        }
    }

    /// Create a spherical area light (P1)
    pub fn area_sphere(position: [f32; 3], radius: f32, intensity: f32, color: [f32; 3]) -> Self {
        Self {
            kind: LightType::AreaSphere.as_u32(),
            intensity,
            range: 0.0,
            env_texture_index: 0,
            color,
            _pad1: 0.0,
            pos_ws: position,
            _pad2: 0.0,
            dir_ws: [0.0; 3], // Unused for sphere
            _pad3: 0.0,
            cone_cos: [1.0, 1.0],
            area_half: [radius, 0.0], // Store radius in x component
        }
    }
}

/// Material shading parameters (P0/P2)
/// GPU-aligned struct for uniform buffer upload
/// Material shading parameters for BRDF dispatch (P2-06)
///
/// This struct is GPU-aligned and matches WGSL `ShadingParamsGPU` exactly.
/// Can be uploaded directly to GPU via uniform buffer at @group(0) @binding(2).
///
/// **Layout Parity**: Must match `ShadingParamsGPU` in `src/shaders/lighting.wgsl`
///
/// **Size**: 32 bytes (2 vec4s)  
/// **Alignment**: 16 bytes (vec4 boundary)  
/// **Traits**: `Pod` + `Zeroable` (bytemuck) for safe GPU upload
///
/// ## Memory Layout
///
/// ```text
/// Offset | Field       | Type | Size | Notes
/// -------|-------------|------|------|------------------
///   0-15 | Vec4 #1     |      |  16  |
///      0 |   brdf      | u32  |   4  | BrdfModel enum value
///      4 |   metallic  | f32  |   4  | [0, 1]
///      8 |   roughness | f32  |   4  | [0, 1]
///     12 |   sheen     | f32  |   4  | [0, 1]
///  16-31 | Vec4 #2     |      |  16  |
///     16 |   clearcoat | f32  |   4  | [0, 1]
///     20 |   subsurface| f32  |   4  | [0, 1]
///     24 |   anisotropy| f32  |   4  | [-1, 1]
///     28 |   _pad      | f32  |   4  | Padding
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaterialShading {
    // Vec4 #1
    /// BRDF model (see BrdfModel enum)
    pub brdf: u32,
    /// Metallic factor [0, 1]
    pub metallic: f32,
    /// Surface roughness [0, 1]
    pub roughness: f32,
    /// Sheen intensity [0, 1] (for cloth-like materials)
    pub sheen: f32,

    // Vec4 #2
    /// Clearcoat layer intensity [0, 1]
    pub clearcoat: f32,
    /// Subsurface scattering [0, 1]
    pub subsurface: f32,
    /// Anisotropy [-1, 1] (directional roughness)
    pub anisotropy: f32,
    /// Padding for 16-byte alignment
    pub _pad: f32,
}

impl Default for MaterialShading {
    fn default() -> Self {
        Self {
            brdf: BrdfModel::CookTorranceGgx.as_u32(),
            metallic: 0.0,
            roughness: 0.5,
            sheen: 0.0,
            clearcoat: 0.0,
            subsurface: 0.0,
            anisotropy: 0.0,
            _pad: 0.0,
        }
    }
}

/// Type alias clarifying that MaterialShading is the CPU-side representation
/// of WGSL `ShadingParamsGPU` (P2-06)
///
/// **Usage**: Create `MaterialShading` on CPU, upload to GPU uniform buffer
/// at @group(0) @binding(2), where it becomes `ShadingParamsGPU` in shaders.
///
/// **Example**:
/// ```rust
/// use forge3d::lighting::MaterialShading;
///
/// let shading = MaterialShading {
///     brdf: BrdfModel::DisneyPrincipled.as_u32(),
///     metallic: 1.0,
///     roughness: 0.3,
///     sheen: 0.1,
///     ..Default::default()
/// };
///
/// // Upload to GPU (in pipeline code):
/// queue.write_buffer(&shading_uniform, 0, bytemuck::bytes_of(&shading));
/// ```
pub type ShadingParamsGPU = MaterialShading;

impl MaterialShading {
    /// Validate material parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.roughness < 0.0 || self.roughness > 1.0 {
            return Err(format!(
                "roughness must be in [0,1], got {}",
                self.roughness
            ));
        }
        if self.metallic < 0.0 || self.metallic > 1.0 {
            return Err(format!("metallic must be in [0,1], got {}", self.metallic));
        }
        if self.sheen < 0.0 || self.sheen > 1.0 {
            return Err(format!("sheen must be in [0,1], got {}", self.sheen));
        }
        if self.clearcoat < 0.0 || self.clearcoat > 1.0 {
            return Err(format!(
                "clearcoat must be in [0,1], got {}",
                self.clearcoat
            ));
        }
        if self.subsurface < 0.0 || self.subsurface > 1.0 {
            return Err(format!(
                "subsurface must be in [0,1], got {}",
                self.subsurface
            ));
        }
        if self.anisotropy < -1.0 || self.anisotropy > 1.0 {
            return Err(format!(
                "anisotropy must be in [-1,1], got {}",
                self.anisotropy
            ));
        }
        Ok(())
    }

    /// Create a Lambert (pure diffuse) material
    pub fn lambert(roughness: f32) -> Self {
        Self {
            brdf: BrdfModel::Lambert.as_u32(),
            roughness,
            ..Default::default()
        }
    }

    /// Create a Phong material
    pub fn phong(roughness: f32, metallic: f32) -> Self {
        Self {
            brdf: BrdfModel::Phong.as_u32(),
            roughness,
            metallic,
            ..Default::default()
        }
    }

    /// Create a Disney Principled material with full parameter control
    pub fn disney(
        roughness: f32,
        metallic: f32,
        sheen: f32,
        clearcoat: f32,
        subsurface: f32,
    ) -> Self {
        Self {
            brdf: BrdfModel::DisneyPrincipled.as_u32(),
            metallic,
            roughness,
            sheen,
            clearcoat,
            subsurface,
            anisotropy: 0.0,
            _pad: 0.0,
        }
    }

    /// Create an anisotropic material (Ward or Ashikhmin-Shirley)
    pub fn anisotropic(brdf_model: BrdfModel, roughness: f32, anisotropy: f32) -> Self {
        Self {
            brdf: brdf_model.as_u32(),
            roughness,
            anisotropy,
            ..Default::default()
        }
    }
}

/// Shadow settings (P0)
/// GPU-aligned struct for uniform buffer upload
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ShadowSettings {
    /// Shadow technique (0=Hard, 1=PCF)
    pub tech: u32,
    /// Shadow map resolution (default 2048)
    pub map_res: u32,
    /// Shadow bias to prevent acne
    pub bias: f32,
    /// Normal-based bias
    pub normal_bias: f32,

    /// PCF/PCSS softness
    pub softness: f32,
    /// PCSS blocker radius in world units
    pub pcss_blocker_radius: f32,
    /// PCSS filter radius in world units
    pub pcss_filter_radius: f32,
    /// Effective light size for penumbra estimation
    pub light_size: f32,
    /// Moment bias for VSM/EVSM/MSM techniques
    pub moment_bias: f32,
    /// Padding for 16-byte alignment
    pub _pad: [f32; 3],
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            tech: ShadowTechnique::PCF.as_u32(),
            map_res: 2048,
            bias: 0.002,
            normal_bias: 0.5,
            softness: 1.25,
            pcss_blocker_radius: 0.03,
            pcss_filter_radius: 0.06,
            light_size: 0.25,
            moment_bias: 0.0005,
            _pad: [0.0; 3],
        }
    }
}

impl ShadowSettings {
    fn technique_enum(&self) -> Option<ShadowTechnique> {
        match self.tech {
            x if x == ShadowTechnique::Hard.as_u32() => Some(ShadowTechnique::Hard),
            x if x == ShadowTechnique::PCF.as_u32() => Some(ShadowTechnique::PCF),
            x if x == ShadowTechnique::PCSS.as_u32() => Some(ShadowTechnique::PCSS),
            x if x == ShadowTechnique::VSM.as_u32() => Some(ShadowTechnique::VSM),
            x if x == ShadowTechnique::EVSM.as_u32() => Some(ShadowTechnique::EVSM),
            x if x == ShadowTechnique::MSM.as_u32() => Some(ShadowTechnique::MSM),
            x if x == ShadowTechnique::CSM.as_u32() => Some(ShadowTechnique::CSM),
            _ => None,
        }
    }

    /// Validate shadow settings
    pub fn validate(&self) -> Result<(), String> {
        if self.map_res == 0 {
            return Err("map_res must be greater than zero".to_string());
        }
        if self.map_res > 4096 {
            return Err(format!("map_res must be <= 4096, got {}", self.map_res));
        }
        if !self.map_res.is_power_of_two() {
            return Err(format!(
                "map_res must be power of two, got {}",
                self.map_res
            ));
        }
        let technique = self
            .technique_enum()
            .ok_or_else(|| format!("unknown shadow technique id {}", self.tech))?;
        if technique == ShadowTechnique::PCSS {
            if self.pcss_blocker_radius < 0.0 {
                return Err("pcss_blocker_radius must be non-negative".to_string());
            }
            if self.pcss_filter_radius < 0.0 {
                return Err("pcss_filter_radius must be non-negative".to_string());
            }
            if self.light_size <= 0.0 {
                return Err("light_size must be positive for PCSS".to_string());
            }
        }
        if technique.requires_moments() && self.moment_bias <= 0.0 {
            return Err("moment_bias must be positive for moment-based techniques".to_string());
        }
        Ok(())
    }

    /// Calculate memory budget for shadow atlas (in bytes)
    pub fn memory_budget(&self) -> u64 {
        let pixels = (self.map_res as u64) * (self.map_res as u64);
        let depth_bytes = pixels * 4;
        let moment_bytes = match self.technique_enum().unwrap_or(ShadowTechnique::Hard) {
            ShadowTechnique::VSM => pixels * 8,
            ShadowTechnique::EVSM | ShadowTechnique::MSM => pixels * 16,
            _ => 0,
        };
        depth_bytes + moment_bytes
    }
}

/// Global illumination settings (P0)
/// GPU-aligned struct for uniform buffer upload
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GiSettings {
    /// GI technique (0=None, 1=IBL)
    pub tech: u32,
    /// IBL intensity multiplier
    pub ibl_intensity: f32,
    /// IBL environment rotation (degrees)
    pub ibl_rotation_deg: f32,
    /// Padding for 16-byte alignment
    pub _pad: f32,
}

impl Default for GiSettings {
    fn default() -> Self {
        Self {
            tech: GiTechnique::Ibl.as_u32(),
            ibl_intensity: 1.0,
            ibl_rotation_deg: 0.0,
            _pad: 0.0,
        }
    }
}

/// Atmospheric settings (P0)
/// GPU-aligned struct for uniform buffer upload
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Atmosphere {
    /// Fog density
    pub fog_density: f32,
    /// Exposure multiplier
    pub exposure: f32,
    /// Sky model (0=Off, 1=Preetham)
    pub sky_model: u32,
    /// Padding for 16-byte alignment
    pub _pad: f32,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self {
            fog_density: 0.0,
            exposure: 1.0,
            sky_model: 0,
            _pad: 0.0,
        }
    }
}

// ============================================================================
// P5: Screen-space effects (SSAO/GTAO, SSGI, SSR)
// ============================================================================

/// Screen-space effect technique enumeration (P5)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScreenSpaceEffect {
    None = 0,
    SSAO = 1, // Screen-space ambient occlusion
    GTAO = 2, // Ground-truth ambient occlusion
    SSGI = 3, // Screen-space global illumination
    SSR = 4,  // Screen-space reflections
}

impl ScreenSpaceEffect {
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn name(&self) -> &'static str {
        match self {
            ScreenSpaceEffect::None => "none",
            ScreenSpaceEffect::SSAO => "ssao",
            ScreenSpaceEffect::GTAO => "gtao",
            ScreenSpaceEffect::SSGI => "ssgi",
            ScreenSpaceEffect::SSR => "ssr",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "none" => Some(ScreenSpaceEffect::None),
            "ssao" => Some(ScreenSpaceEffect::SSAO),
            "gtao" => Some(ScreenSpaceEffect::GTAO),
            "ssgi" => Some(ScreenSpaceEffect::SSGI),
            "ssr" => Some(ScreenSpaceEffect::SSR),
            _ => None,
        }
    }
}

/// SSAO/GTAO settings (P5)
/// GPU-aligned struct for uniform buffer upload
/// Size: 32 bytes (2 vec4s)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SSAOSettings {
    // Vec4 #1
    /// World-space occlusion radius
    pub radius: f32,
    /// AO intensity multiplier [0-2]
    pub intensity: f32,
    /// Depth bias to prevent self-occlusion
    pub bias: f32,
    /// Number of samples per pixel
    pub sample_count: u32,

    // Vec4 #2
    /// Spiral pattern parameter
    pub spiral_turns: f32,
    /// Technique (0=SSAO, 1=GTAO)
    pub technique: u32,
    /// Bilateral blur radius
    pub blur_radius: u32,
    /// Temporal accumulation factor [0=off, 0.95=strong]
    pub temporal_alpha: f32,
}

impl Default for SSAOSettings {
    fn default() -> Self {
        Self {
            radius: 0.5,
            intensity: 1.0,
            bias: 0.025,
            sample_count: 16,
            spiral_turns: 7.0,
            technique: 0, // SSAO
            blur_radius: 2,
            temporal_alpha: 0.0,
        }
    }
}

impl SSAOSettings {
    /// Create SSAO configuration
    pub fn ssao(radius: f32, intensity: f32) -> Self {
        Self {
            radius,
            intensity,
            technique: 0,
            ..Default::default()
        }
    }

    /// Create GTAO configuration (more accurate)
    pub fn gtao(radius: f32, intensity: f32) -> Self {
        Self {
            radius,
            intensity,
            technique: 1,
            ..Default::default()
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.radius <= 0.0 || self.radius > 100.0 {
            return Err("radius must be in (0, 100]");
        }
        if self.intensity < 0.0 || self.intensity > 5.0 {
            return Err("intensity must be in [0, 5]");
        }
        if self.sample_count == 0 || self.sample_count > 64 {
            return Err("sample_count must be in [1, 64]");
        }
        if self.technique > 1 {
            return Err("technique must be 0 (SSAO) or 1 (GTAO)");
        }
        Ok(())
    }
}

/// SSGI settings (P5)
/// GPU-aligned struct for uniform buffer upload
/// Size: 32 bytes (2 vec4s)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SSGISettings {
    // Vec4 #1
    /// Max ray marching steps (16-32)
    pub ray_steps: u32,
    /// Max ray distance in world space
    pub ray_radius: f32,
    /// Ray hit thickness tolerance
    pub ray_thickness: f32,
    /// GI intensity multiplier
    pub intensity: f32,

    // Vec4 #2
    /// Temporal accumulation [0=off, 0.9=strong]
    pub temporal_alpha: f32,
    /// 1=half resolution for performance
    pub use_half_res: u32,
    /// IBL contribution when ray misses [0-1]
    pub ibl_fallback: f32,
    /// Padding
    pub _pad: f32,
}

impl Default for SSGISettings {
    fn default() -> Self {
        Self {
            ray_steps: 24,
            ray_radius: 5.0,
            ray_thickness: 0.5,
            intensity: 1.0,
            temporal_alpha: 0.0,
            use_half_res: 1,
            ibl_fallback: 0.3,
            _pad: 0.0,
        }
    }
}

impl SSGISettings {
    pub fn new(ray_radius: f32, intensity: f32) -> Self {
        Self {
            ray_radius,
            intensity,
            ..Default::default()
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.ray_steps == 0 || self.ray_steps > 128 {
            return Err("ray_steps must be in [1, 128]");
        }
        if self.ray_radius <= 0.0 || self.ray_radius > 1000.0 {
            return Err("ray_radius must be in (0, 1000]");
        }
        if self.intensity < 0.0 || self.intensity > 10.0 {
            return Err("intensity must be in [0, 10]");
        }
        Ok(())
    }
}

/// SSR settings (P5)
/// GPU-aligned struct for uniform buffer upload
/// Size: 32 bytes (2 vec4s)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SSRSettings {
    // Vec4 #1
    /// Max ray marching steps (32-64)
    pub max_steps: u32,
    /// Max reflection distance in view space
    pub max_distance: f32,
    /// Ray hit thickness tolerance
    pub thickness: f32,
    /// Initial step size multiplier
    pub stride: f32,

    // Vec4 #2
    /// Reflection intensity
    pub intensity: f32,
    /// Fade reflections with roughness [0-1]
    pub roughness_fade: f32,
    /// Screen edge fade distance [0-0.2]
    pub edge_fade: f32,
    /// Temporal accumulation [0=off, 0.85=strong]
    pub temporal_alpha: f32,
}

impl Default for SSRSettings {
    fn default() -> Self {
        Self {
            max_steps: 48,
            max_distance: 50.0,
            thickness: 0.5,
            stride: 1.0,
            intensity: 1.0,
            roughness_fade: 0.8,
            edge_fade: 0.1,
            temporal_alpha: 0.0,
        }
    }
}

impl SSRSettings {
    pub fn new(max_distance: f32, intensity: f32) -> Self {
        Self {
            max_distance,
            intensity,
            ..Default::default()
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_steps == 0 || self.max_steps > 256 {
            return Err("max_steps must be in [1, 256]");
        }
        if self.max_distance <= 0.0 || self.max_distance > 1000.0 {
            return Err("max_distance must be in (0, 1000]");
        }
        if self.intensity < 0.0 || self.intensity > 10.0 {
            return Err("intensity must be in [0, 10]");
        }
        Ok(())
    }
}

/// Combined screen-space settings (P5)
/// Convenience struct for enabling multiple screen-space effects
#[derive(Debug, Clone, Copy)]
pub struct ScreenSpaceSettings {
    pub ssao: Option<SSAOSettings>,
    pub ssgi: Option<SSGISettings>,
    pub ssr: Option<SSRSettings>,
}

impl Default for ScreenSpaceSettings {
    fn default() -> Self {
        Self {
            ssao: None,
            ssgi: None,
            ssr: None,
        }
    }
}

impl ScreenSpaceSettings {
    pub fn with_ssao(radius: f32, intensity: f32) -> Self {
        Self {
            ssao: Some(SSAOSettings::ssao(radius, intensity)),
            ..Default::default()
        }
    }

    pub fn with_gtao(radius: f32, intensity: f32) -> Self {
        Self {
            ssao: Some(SSAOSettings::gtao(radius, intensity)),
            ..Default::default()
        }
    }

    pub fn with_ssgi(ray_radius: f32, intensity: f32) -> Self {
        Self {
            ssgi: Some(SSGISettings::new(ray_radius, intensity)),
            ..Default::default()
        }
    }

    pub fn with_ssr(max_distance: f32, intensity: f32) -> Self {
        Self {
            ssr: Some(SSRSettings::new(max_distance, intensity)),
            ..Default::default()
        }
    }
}

// ============================================================================
// P6: Atmospherics & sky (Hosek-Wilkie/Preetham, volumetric fog/god-rays)
// ============================================================================

/// Sky model enumeration (P6)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkyModel {
    Off = 0,
    Preetham = 1,
    HosekWilkie = 2,
}

impl SkyModel {
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn name(&self) -> &'static str {
        match self {
            SkyModel::Off => "off",
            SkyModel::Preetham => "preetham",
            SkyModel::HosekWilkie => "hosek-wilkie",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "off" => Some(SkyModel::Off),
            "preetham" => Some(SkyModel::Preetham),
            "hosek-wilkie" | "hosek_wilkie" | "hosekwilkie" => Some(SkyModel::HosekWilkie),
            _ => None,
        }
    }
}

/// Sky settings (P6)
/// GPU-aligned struct for uniform buffer upload
/// Size: 48 bytes (3 vec4s)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SkySettings {
    // Vec4 #1
    /// Sun direction (normalized, typically from directional light)
    pub sun_direction: [f32; 3],
    /// Atmospheric turbidity [1.0-10.0, 2.0=clear, 6.0=hazy]
    pub turbidity: f32,

    // Vec4 #2
    /// Ground reflectance/albedo [0-1]
    pub ground_albedo: f32,
    /// Sky model (0=Off, 1=Preetham, 2=Hosek-Wilkie)
    pub model: u32,
    /// Sun intensity multiplier
    pub sun_intensity: f32,
    /// Exposure adjustment
    pub exposure: f32,

    // Vec4 #3 (padding to 48 bytes)
    pub _pad: [f32; 4],
}

impl Default for SkySettings {
    fn default() -> Self {
        Self {
            sun_direction: [0.3, 0.8, 0.5], // Default sun angle
            turbidity: 2.5,
            ground_albedo: 0.2,
            model: SkyModel::HosekWilkie.as_u32(),
            sun_intensity: 20.0,
            exposure: 1.0,
            _pad: [0.0; 4],
        }
    }
}

impl SkySettings {
    /// Create Preetham sky model
    pub fn preetham(turbidity: f32, ground_albedo: f32) -> Self {
        Self {
            turbidity,
            ground_albedo,
            model: SkyModel::Preetham.as_u32(),
            ..Default::default()
        }
    }

    /// Create Hosek-Wilkie sky model (more accurate)
    pub fn hosek_wilkie(turbidity: f32, ground_albedo: f32) -> Self {
        Self {
            turbidity,
            ground_albedo,
            model: SkyModel::HosekWilkie.as_u32(),
            ..Default::default()
        }
    }

    /// Set sun direction from azimuth and elevation angles (degrees)
    pub fn with_sun_angles(mut self, azimuth_deg: f32, elevation_deg: f32) -> Self {
        let az_rad = azimuth_deg.to_radians();
        let el_rad = elevation_deg.to_radians();

        self.sun_direction = [
            el_rad.cos() * az_rad.sin(),
            el_rad.sin(),
            el_rad.cos() * az_rad.cos(),
        ];

        self
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.turbidity < 1.0 || self.turbidity > 10.0 {
            return Err("turbidity must be in [1.0, 10.0]");
        }
        if self.ground_albedo < 0.0 || self.ground_albedo > 1.0 {
            return Err("ground_albedo must be in [0, 1]");
        }
        if self.sun_intensity < 0.0 || self.sun_intensity > 1000.0 {
            return Err("sun_intensity must be in [0, 1000]");
        }
        Ok(())
    }
}

/// Volumetric phase function enumeration (P6)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumetricPhase {
    Isotropic = 0,
    HenyeyGreenstein = 1,
}

impl VolumetricPhase {
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn name(&self) -> &'static str {
        match self {
            VolumetricPhase::Isotropic => "isotropic",
            VolumetricPhase::HenyeyGreenstein => "hg",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "isotropic" | "iso" => Some(VolumetricPhase::Isotropic),
            "hg" | "henyey-greenstein" | "henyey_greenstein" => {
                Some(VolumetricPhase::HenyeyGreenstein)
            }
            _ => None,
        }
    }
}

/// Volumetric fog settings (P6)
/// GPU-aligned struct for uniform buffer upload
/// Size: 80 bytes (5 vec4s)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct VolumetricSettings {
    // Vec4 #1
    /// Base fog density [0.001-1.0]
    pub density: f32,
    /// Exponential height falloff [0-1]
    pub height_falloff: f32,
    /// Henyey-Greenstein asymmetry [-1 to 1, 0=isotropic, 0.7=forward]
    pub phase_g: f32,
    /// Ray marching steps [16-128]
    pub max_steps: u32,

    // Vec4 #2
    /// Near plane for fog [0.1-10]
    pub start_distance: f32,
    /// Far plane for fog [10-1000]
    pub max_distance: f32,
    /// Absorption coefficient [0-1]
    pub absorption: f32,
    /// Sun intensity for in-scattering
    pub sun_intensity: f32,

    // Vec4 #3
    /// Fog tint color (RGB)
    pub scattering_color: [f32; 3],
    /// Temporal reprojection blend [0-0.9]
    pub temporal_alpha: f32,

    // Vec4 #4
    /// Ambient sky contribution (RGB)
    pub ambient_color: [f32; 3],
    /// 1=sample shadow map for god-rays
    pub use_shadows: u32,

    // Vec4 #5
    /// Ray march jitter [0-1]
    pub jitter_strength: f32,
    /// Phase function type
    pub phase_function: u32,
    /// Padding
    pub _pad: [f32; 2],
}

impl Default for VolumetricSettings {
    fn default() -> Self {
        Self {
            density: 0.015,
            height_falloff: 0.1,
            phase_g: 0.7,
            max_steps: 48,
            start_distance: 0.1,
            max_distance: 100.0,
            absorption: 0.5,
            sun_intensity: 1.0,
            scattering_color: [1.0, 1.0, 1.0],
            temporal_alpha: 0.0,
            ambient_color: [0.3, 0.4, 0.5],
            use_shadows: 1,
            jitter_strength: 0.5,
            phase_function: VolumetricPhase::HenyeyGreenstein.as_u32(),
            _pad: [0.0; 2],
        }
    }
}

impl VolumetricSettings {
    /// Create volumetric fog with god-rays
    pub fn with_god_rays(density: f32, phase_g: f32) -> Self {
        Self {
            density,
            phase_g,
            use_shadows: 1,
            ..Default::default()
        }
    }

    /// Create simple uniform fog without god-rays
    pub fn uniform_fog(density: f32) -> Self {
        Self {
            density,
            phase_g: 0.0, // Isotropic
            use_shadows: 0,
            height_falloff: 0.0,
            ..Default::default()
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.density < 0.0 || self.density > 10.0 {
            return Err("density must be in [0, 10]");
        }
        if self.height_falloff < 0.0 || self.height_falloff > 10.0 {
            return Err("height_falloff must be in [0, 10]");
        }
        if self.phase_g < -1.0 || self.phase_g > 1.0 {
            return Err("phase_g must be in [-1, 1]");
        }
        if self.max_steps == 0 || self.max_steps > 256 {
            return Err("max_steps must be in [1, 256]");
        }
        if self.start_distance < 0.0 || self.start_distance >= self.max_distance {
            return Err("start_distance must be in [0, max_distance)");
        }
        Ok(())
    }

    /// Calculate approximate memory budget for froxelized volumetrics
    pub fn froxel_memory_budget(&self) -> usize {
        // Typical froxel grid: 16x8x64 = 8192 froxels
        // Each froxel: 4 channels (RGB scattering + extinction) × 2 bytes (f16) = 8 bytes
        8192 * 8
    }
}

/// Combined atmospherics settings (P6)
#[derive(Debug, Clone, Copy)]
pub struct AtmosphericsSettings {
    pub sky: Option<SkySettings>,
    pub volumetric: Option<VolumetricSettings>,
}

impl Default for AtmosphericsSettings {
    fn default() -> Self {
        Self {
            sky: None,
            volumetric: None,
        }
    }
}

impl AtmosphericsSettings {
    pub fn with_sky(turbidity: f32, ground_albedo: f32) -> Self {
        Self {
            sky: Some(SkySettings::hosek_wilkie(turbidity, ground_albedo)),
            ..Default::default()
        }
    }

    pub fn with_volumetric(density: f32, phase_g: f32) -> Self {
        Self {
            volumetric: Some(VolumetricSettings::with_god_rays(density, phase_g)),
            ..Default::default()
        }
    }

    pub fn full_atmospherics(
        turbidity: f32,
        ground_albedo: f32,
        fog_density: f32,
        phase_g: f32,
    ) -> Self {
        Self {
            sky: Some(SkySettings::hosek_wilkie(turbidity, ground_albedo)),
            volumetric: Some(VolumetricSettings::with_god_rays(fog_density, phase_g)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_light_sizes() {
        // Verify sizes are multiples of 16 for UBO alignment
        assert_eq!(std::mem::size_of::<Light>() % 16, 0);
        assert_eq!(std::mem::size_of::<MaterialShading>() % 16, 0);
        assert_eq!(std::mem::size_of::<ShadowSettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<GiSettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<Atmosphere>() % 16, 0);
        // P5: Screen-space effects
        assert_eq!(std::mem::size_of::<SSAOSettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<SSGISettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<SSRSettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<SSAOSettings>(), 32);
        assert_eq!(std::mem::size_of::<SSGISettings>(), 32);
        assert_eq!(std::mem::size_of::<SSRSettings>(), 32);
        // P6: Atmospherics
        assert_eq!(std::mem::size_of::<SkySettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<VolumetricSettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<SkySettings>(), 48);
        assert_eq!(std::mem::size_of::<VolumetricSettings>(), 80);
    }

    // P1-01: Host/device layout parity tests
    #[test]
    fn test_light_struct_size_and_alignment() {
        // Light must be exactly 80 bytes (5 * vec4) for std430 SSBO layout
        assert_eq!(
            std::mem::size_of::<Light>(),
            80,
            "Light struct must be 80 bytes to match WGSL LightGPU layout"
        );

        // Verify 16-byte alignment for vec4 boundary
        assert_eq!(
            std::mem::align_of::<Light>(),
            16,
            "Light struct must be 16-byte aligned"
        );
    }

    #[test]
    fn test_light_field_offsets() {
        use std::mem::offset_of;

        // Vec4 #1: type, intensity, range, env_texture_index (bytes 0-15)
        assert_eq!(offset_of!(Light, kind), 0);
        assert_eq!(offset_of!(Light, intensity), 4);
        assert_eq!(offset_of!(Light, range), 8);
        assert_eq!(offset_of!(Light, env_texture_index), 12);

        // Vec4 #2: color + padding (bytes 16-31)
        assert_eq!(offset_of!(Light, color), 16);
        assert_eq!(offset_of!(Light, _pad1), 28);

        // Vec4 #3: pos_ws + padding (bytes 32-47)
        assert_eq!(offset_of!(Light, pos_ws), 32);
        assert_eq!(offset_of!(Light, _pad2), 44);

        // Vec4 #4: dir_ws + padding (bytes 48-63)
        assert_eq!(offset_of!(Light, dir_ws), 48);
        assert_eq!(offset_of!(Light, _pad3), 60);

        // Vec4 #5: cone_cos + area_half (bytes 64-79)
        assert_eq!(offset_of!(Light, cone_cos), 64);
        assert_eq!(offset_of!(Light, area_half), 72);
    }

    #[test]
    fn test_light_type_enum_values() {
        // Verify enum values match WGSL constants
        assert_eq!(LightType::Directional.as_u32(), 0);
        assert_eq!(LightType::Point.as_u32(), 1);
        assert_eq!(LightType::Spot.as_u32(), 2);
        assert_eq!(LightType::Environment.as_u32(), 3);
        assert_eq!(LightType::AreaRect.as_u32(), 4);
        assert_eq!(LightType::AreaDisk.as_u32(), 5);
        assert_eq!(LightType::AreaSphere.as_u32(), 6);
    }

    #[test]
    fn test_light_pod_safety() {
        // Verify Light can be safely cast to bytes (bytemuck Pod trait)
        let light = Light::default();
        let _bytes: &[u8] = bytemuck::bytes_of(&light);
        assert_eq!(_bytes.len(), 80);
    }

    #[test]
    fn test_light_directional_unused_fields() {
        // Directional lights should have well-defined unused fields
        let light = Light::directional(45.0, 30.0, 2.0, [1.0, 0.9, 0.8]);
        assert_eq!(light.kind, LightType::Directional.as_u32());
        assert_eq!(light.pos_ws, [0.0; 3]); // Unused for directional
        assert_eq!(light.range, 0.0); // Unused for directional
    }

    #[test]
    fn test_light_environment_fields() {
        // Environment lights should carry env_texture_index
        let light = Light::environment(1.5, 42);
        assert_eq!(light.kind, LightType::Environment.as_u32());
        assert_eq!(light.env_texture_index, 42);
        assert_eq!(light.intensity, 1.5);
    }

    #[test]
    fn test_light_point_fields() {
        let pos = [10.0, 20.0, 30.0];
        let light = Light::point(pos, 50.0, 3.0, [1.0, 0.5, 0.2]);
        assert_eq!(light.kind, LightType::Point.as_u32());
        assert_eq!(light.pos_ws, pos);
        assert_eq!(light.range, 50.0);
        assert_eq!(light.dir_ws, [0.0; 3]); // Unused for point
    }

    #[test]
    fn test_light_spot_cone_precompute() {
        let light = Light::spot(
            [0.0, 5.0, 0.0],
            [0.0, -1.0, 0.0],
            100.0,
            20.0, // inner_deg
            35.0, // outer_deg
            5.0,
            [1.0, 1.0, 1.0],
        );
        assert_eq!(light.kind, LightType::Spot.as_u32());
        // Verify cosines are precomputed (not raw degrees)
        assert!(light.cone_cos[0] > 0.9 && light.cone_cos[0] < 1.0); // cos(20°) ≈ 0.94
        assert!(light.cone_cos[1] > 0.8 && light.cone_cos[1] < 0.9); // cos(35°) ≈ 0.82
    }

    #[test]
    fn test_light_area_rect_fields() {
        let light = Light::area_rect(
            [0.0, 10.0, 0.0],
            [0.0, -1.0, 0.0],
            2.5, // half_width
            1.5, // half_height
            10.0,
            [1.0, 1.0, 1.0],
        );
        assert_eq!(light.kind, LightType::AreaRect.as_u32());
        assert_eq!(light.area_half, [2.5, 1.5]);
    }

    #[test]
    fn test_light_area_disk_fields() {
        let light = Light::area_disk(
            [5.0, 5.0, 5.0],
            [1.0, 0.0, 0.0],
            3.0, // radius
            8.0,
            [1.0, 1.0, 0.8],
        );
        assert_eq!(light.kind, LightType::AreaDisk.as_u32());
        assert_eq!(light.area_half[0], 3.0); // Radius stored in x
        assert_eq!(light.area_half[1], 0.0); // y unused
    }

    #[test]
    fn test_light_area_sphere_fields() {
        let light = Light::area_sphere(
            [0.0, 15.0, 0.0],
            2.0, // radius
            12.0,
            [1.0, 0.9, 0.7],
        );
        assert_eq!(light.kind, LightType::AreaSphere.as_u32());
        assert_eq!(light.area_half[0], 2.0); // Radius stored in x
        assert_eq!(light.dir_ws, [0.0; 3]); // Unused for sphere
    }

    #[test]
    fn test_material_validation() {
        let mut mat = MaterialShading::default();
        assert!(mat.validate().is_ok());

        mat.roughness = 1.5;
        assert!(mat.validate().is_err());

        mat.roughness = 0.5;
        mat.metallic = -0.1;
        assert!(mat.validate().is_err());
    }

    #[test]
    fn test_shadow_validation() {
        let mut shadow = ShadowSettings::default();
        assert!(shadow.validate().is_ok());

        shadow.map_res = 5000;
        assert!(shadow.validate().is_err());

        shadow.map_res = 2047;
        assert!(shadow.validate().is_err());
    }

    #[test]
    fn test_shadow_memory_budget() {
        let shadow = ShadowSettings {
            map_res: 2048,
            ..Default::default()
        };
        // 2048 * 2048 * 4 bytes = 16 MiB
        assert_eq!(shadow.memory_budget(), 16 * 1024 * 1024);
    }
}
