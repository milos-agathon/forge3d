use std::path::PathBuf;

/// Blend mode for overlay compositing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Standard alpha blend: mix(base, overlay, alpha)
    #[default]
    Normal,
    /// Multiply: base * overlay
    Multiply,
    /// Overlay: Photoshop-style overlay blend
    Overlay,
}

impl BlendMode {
    /// Convert to f32 for shader uniform
    pub fn to_shader_value(&self) -> f32 {
        match self {
            BlendMode::Normal => 0.0,
            BlendMode::Multiply => 1.0,
            BlendMode::Overlay => 2.0,
        }
    }
}

/// Source data for an overlay layer
#[derive(Debug, Clone)]
pub enum OverlayData {
    /// Raw RGBA pixels
    Raster {
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    },
    /// Path to an image file (PNG, JPEG, etc.)
    Image { path: PathBuf },
}

/// Single overlay layer configuration
#[derive(Debug, Clone)]
pub struct OverlayLayer {
    /// Unique name for this layer
    pub name: String,
    /// Source data (raster pixels or image path)
    pub data: OverlayData,
    /// Extent in terrain UV space: [u_min, v_min, u_max, v_max]
    /// None means full terrain coverage [0, 0, 1, 1]
    pub extent: Option<[f32; 4]>,
    /// Opacity: 0.0 (transparent) to 1.0 (opaque)
    pub opacity: f32,
    /// Blend mode for compositing
    pub blend_mode: BlendMode,
    /// Whether this layer is visible
    pub visible: bool,
    /// Z-order for stacking (lower = behind, higher = in front)
    pub z_order: i32,
}

impl Default for OverlayLayer {
    fn default() -> Self {
        Self {
            name: String::new(),
            data: OverlayData::Raster {
                rgba: Vec::new(),
                width: 0,
                height: 0,
            },
            extent: None,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            visible: true,
            z_order: 0,
        }
    }
}

/// GPU resources for a single overlay layer
pub struct OverlayLayerGpu {
    /// Layer ID
    pub id: u32,
    /// Layer configuration (CPU side)
    pub config: OverlayLayer,
    /// GPU texture
    pub texture: wgpu::Texture,
    /// Texture view
    pub view: wgpu::TextureView,
    /// Texture dimensions
    pub dimensions: (u32, u32),
}

/// Overlay configuration for IPC/API
#[derive(Debug, Clone, Default)]
pub struct OverlayConfig {
    /// Enable overlay system (default: false)
    pub enabled: bool,
    /// Global overlay opacity multiplier (0.0 - 1.0)
    pub global_opacity: f32,
    /// Preserve source overlay colors by compositing after terrain lighting
    pub preserve_colors: bool,
    /// Show solid base surface (default: true)
    /// When false, fragments with overlay alpha=0 are discarded (like rayshader solid=FALSE)
    pub solid: bool,
}

impl OverlayConfig {
    pub fn new() -> Self {
        Self {
            enabled: false,
            global_opacity: 1.0,
            preserve_colors: false,
            solid: true,
        }
    }
}
