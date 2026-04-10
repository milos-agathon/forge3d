use super::*;

#[derive(Debug)]
pub(super) struct PbrIblResources {
    pub(super) _irradiance_texture: Texture,
    pub(super) irradiance_view: TextureView,
    pub(super) irradiance_sampler: Sampler,
    pub(super) _prefilter_texture: Texture,
    pub(super) prefilter_view: TextureView,
    pub(super) _prefilter_sampler: Sampler,
    pub(super) _brdf_lut_texture: Texture,
    pub(super) brdf_lut_view: TextureView,
    pub(super) _brdf_lut_sampler: Sampler,
}

pub(super) fn create_fallback_ibl_resources(device: &Device, queue: &Queue) -> PbrIblResources {
    let irradiance_texture = create_default_texture(
        device,
        queue,
        "pbr_fallback_irradiance",
        [255, 255, 255, 255],
    );
    let prefilter_texture = create_default_texture(
        device,
        queue,
        "pbr_fallback_prefilter",
        [255, 255, 255, 255],
    );
    let brdf_lut_texture =
        create_default_texture(device, queue, "pbr_fallback_brdf_lut", [255, 255, 255, 255]);

    let irradiance_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("pbr_fallback_ibl_irradiance_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    });
    let prefilter_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("pbr_fallback_ibl_prefilter_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    });
    let brdf_lut_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("pbr_fallback_ibl_lut_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    });

    PbrIblResources {
        irradiance_view: irradiance_texture.create_view(&TextureViewDescriptor::default()),
        irradiance_sampler,
        _irradiance_texture: irradiance_texture,
        prefilter_view: prefilter_texture.create_view(&TextureViewDescriptor::default()),
        _prefilter_sampler: prefilter_sampler,
        _prefilter_texture: prefilter_texture,
        brdf_lut_view: brdf_lut_texture.create_view(&TextureViewDescriptor::default()),
        _brdf_lut_sampler: brdf_lut_sampler,
        _brdf_lut_texture: brdf_lut_texture,
    }
}
