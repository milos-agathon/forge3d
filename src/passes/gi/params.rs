//! GI composite parameter structs and GPU layout conversions.

/// GPU-side std140 representation of GI composite controls.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GiCompositeParamsStd140 {
    pub ao_enable: u32,
    pub ssgi_enable: u32,
    pub ssr_enable: u32,
    pub _pad0: u32,
    pub ao_weight: f32,
    pub ssgi_weight: f32,
    pub ssr_weight: f32,
    pub energy_cap: f32,
}

/// CPU-side representation of GI composite controls.
#[derive(Clone, Copy)]
pub struct GiCompositeParams {
    /// Enable AO multiplier on diffuse.
    pub ao_enable: bool,
    /// Enable SSGI diffuse GI term.
    pub ssgi_enable: bool,
    /// Enable SSR specular replacement/lerp.
    pub ssr_enable: bool,
    /// AO weight in [0,1]: 0 = no AO effect, 1 = full AO.
    pub ao_weight: f32,
    /// Scales diffuse GI radiance before energy capping.
    pub ssgi_weight: f32,
    /// Scales SSR blend factor (in addition to Fresnel/roughness).
    pub ssr_weight: f32,
    /// Global energy cap relative to baseline+IBL (typically 1.05).
    pub energy_cap: f32,
}

impl Default for GiCompositeParams {
    fn default() -> Self {
        Self {
            ao_enable: true,
            ssgi_enable: true,
            ssr_enable: true,
            ao_weight: 1.0,
            ssgi_weight: 1.0,
            ssr_weight: 1.0,
            energy_cap: 1.05,
        }
    }
}

impl From<GiCompositeParams> for GiCompositeParamsStd140 {
    fn from(p: GiCompositeParams) -> Self {
        Self {
            ao_enable: u32::from(p.ao_enable),
            ssgi_enable: u32::from(p.ssgi_enable),
            ssr_enable: u32::from(p.ssr_enable),
            _pad0: 0,
            ao_weight: p.ao_weight,
            ssgi_weight: p.ssgi_weight,
            ssr_weight: p.ssr_weight,
            energy_cap: p.energy_cap,
        }
    }
}
