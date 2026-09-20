use super::*;

pub(super) struct PbrIblResources {
    pub(super) _renderer: crate::core::ibl::IBLRenderer,
    pub(super) irradiance_view: TextureView,
    pub(super) irradiance_sampler: Sampler,
    pub(super) prefilter_view: TextureView,
    pub(super) brdf_lut_view: TextureView,
}

pub(super) fn create_fallback_ibl_resources(
    device: &Device,
    queue: &Queue,
) -> RenderResult<PbrIblResources> {
    create_ibl_resources(device, queue, &[1.0, 1.0, 1.0], 1, 1)
}

pub(super) fn create_ibl_resources(
    device: &Device,
    queue: &Queue,
    pixels: &[f32],
    width: u32,
    height: u32,
) -> RenderResult<PbrIblResources> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3));
    if width == 0
        || height == 0
        || expected != Some(pixels.len())
        || !pixels.iter().all(|v| v.is_finite() && *v >= 0.0)
    {
        return Err(crate::core::error::RenderError::Render(
            "PBR environment must be a finite nonnegative RGB image".into(),
        ));
    }
    let mut renderer = crate::core::ibl::IBLRenderer::new(device, Default::default())?;
    renderer
        .load_environment_map(device, queue, pixels, width, height)
        .map_err(crate::core::error::RenderError::Render)?;
    renderer
        .initialize(device, queue)
        .map_err(crate::core::error::RenderError::Render)?;
    let (Some(irradiance), Some(specular), Some(brdf)) = renderer.textures() else {
        return Err(crate::core::error::RenderError::Render(
            "IBL precomputation produced incomplete maps".into(),
        ));
    };
    let cube_view = TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    };
    let irradiance_view = irradiance.create_view(&cube_view);
    let prefilter_view = specular.create_view(&cube_view);
    let brdf_lut_view = brdf.create_view(&TextureViewDescriptor::default());
    let irradiance_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("pbr_ibl_sampler"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        ..Default::default()
    });
    Ok(PbrIblResources {
        _renderer: renderer,
        irradiance_view,
        irradiance_sampler,
        prefilter_view,
        brdf_lut_view,
    })
}

impl PbrPipelineWithShadows {
    pub fn set_environment_map(
        &mut self,
        device: &Device,
        queue: &Queue,
        pixels: &[f32],
        width: u32,
        height: u32,
    ) -> RenderResult<()> {
        self.ibl_resources = create_ibl_resources(device, queue, pixels, width, height)?;
        self.ibl_bind_group = None;
        Ok(())
    }
}
