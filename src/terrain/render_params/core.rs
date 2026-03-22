use super::*;

#[derive(Clone)]
pub struct DecodedTerrainSettings {
    pub light: LightSettingsNative,
    pub triplanar: TriplanarSettingsNative,
    pub pom: PomSettingsNative,
    pub lod: LodSettingsNative,
    pub clamp: ClampSettingsNative,
    pub sampling: SamplingSettingsNative,
    pub shadow: ShadowSettingsNative,
    pub fog: FogSettingsNative,
    pub reflection: ReflectionSettingsNative,
    pub detail: DetailSettingsNative,
    pub height_ao: HeightAoSettingsNative,
    pub sun_visibility: SunVisibilitySettingsNative,
    pub bloom: BloomSettingsNative,
    pub materials: MaterialLayerSettingsNative,
    pub vector_overlay: VectorOverlaySettingsNative,
    pub tonemap: TonemapSettingsNative,
    pub aov: AovSettingsNative,
    pub dof: DofSettingsNative,
    pub motion_blur: MotionBlurSettingsNative,
    pub lens_effects: LensEffectsSettingsNative,
    pub denoise: DenoiseSettingsNative,
    pub volumetrics: VolumetricsSettingsNative,
    pub sky: SkySettingsNative,
    pub probes: ProbeSettingsNative,
    pub reflection_probes: ReflectionProbeSettingsNative,
}

/// Terrain render parameter wrapper used by the native renderer.
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "TerrainRenderParams")]
#[derive(Clone)]
pub struct TerrainRenderParams {
    pub size_px: (u32, u32),
    pub render_scale: f32,
    pub terrain_span: f32,
    pub msaa_samples: u32,
    pub z_scale: f32,
    pub cam_target: [f32; 3],
    pub cam_radius: f32,
    pub cam_phi_deg: f32,
    pub cam_theta_deg: f32,
    pub cam_gamma_deg: f32,
    pub fov_y_deg: f32,
    pub clip: (f32, f32),
    pub exposure: f32,
    pub gamma: f32,
    pub albedo_mode: String,
    pub colormap_strength: f32,
    /// P5: AO weight/multiplier (0.0 = no AO effect, 1.0 = full AO). Default 0.0 for P4 compatibility.
    pub ao_weight: f32,
    pub height_curve_mode: String,
    pub height_curve_strength: f32,
    pub height_curve_power: f32,
    /// P5-L: Lambert contrast parameter [0,1] for gradient enhancement
    pub lambert_contrast: f32,
    /// P6.1: Use Rgba8UnormSrgb for colormap texture (correct color space sampling)
    pub colormap_srgb: bool,
    /// P6.1: Use exact linear_to_srgb() instead of pow-gamma for output encoding
    pub output_srgb_eotf: bool,
    /// P7: Camera projection mode ("screen" = fullscreen triangle, "mesh" = perspective grid)
    pub camera_mode: String,
    /// P7: Debug mode for projection probes (0=normal, 40=view-depth, 41=NDC depth, 42=view-pos XYZ)
    pub debug_mode: u32,
    /// M1: Accumulation AA sample count (1 = no AA, 16/64/256 typical for offline)
    pub aa_samples: u32,
    /// M1: Accumulation AA seed for deterministic jitter (None = default sequence)
    pub aa_seed: Option<u64>,
    pub height_curve_lut: Option<Arc<Vec<f32>>>,
    pub overlays: Vec<Py<crate::core::overlay_layer::OverlayLayer>>,
    pub(crate) light: Py<PyAny>,
    pub(crate) ibl: Py<PyAny>,
    pub(crate) shadows: Py<PyAny>,
    pub(crate) triplanar: Py<PyAny>,
    pub(crate) pom: Py<PyAny>,
    pub(crate) lod: Py<PyAny>,
    pub(crate) sampling: Py<PyAny>,
    pub(crate) clamp: Py<PyAny>,
    pub(crate) python_object: Py<PyAny>,
    pub(crate) decoded: DecodedTerrainSettings,
}
