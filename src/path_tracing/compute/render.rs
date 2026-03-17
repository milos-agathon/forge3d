use super::*;

impl PathTracerGPU {
    pub fn render(
        width: u32,
        height: u32,
        spheres: &[Sphere],
        uniforms: Uniforms,
    ) -> Result<Vec<u8>, RenderError> {
        let resources = setup::create_dispatch_resources(width, height, spheres, uniforms);
        dispatch::dispatch(&resources, width, height);
        readback::read_output_rgba8(&resources.out_tex, width, height)
    }

    pub fn render_aovs(
        width: u32,
        height: u32,
        spheres: &[Sphere],
        mut uniforms: Uniforms,
        aov_mask: u32,
    ) -> Result<std::collections::HashMap<AovKind, Vec<u8>>, RenderError> {
        uniforms.aov_flags = aov_mask;
        let resources = setup::create_dispatch_resources(width, height, spheres, uniforms);
        dispatch::dispatch(&resources, width, height);

        let mut out = std::collections::HashMap::new();
        for &kind in &[
            AovKind::Albedo,
            AovKind::Normal,
            AovKind::Depth,
            AovKind::Direct,
            AovKind::Indirect,
            AovKind::Emission,
            AovKind::Visibility,
        ] {
            if (aov_mask & (1u32 << kind.flag_bit())) == 0 {
                continue;
            }
            let texture = resources
                .aov_frames
                .get_texture(kind)
                .ok_or_else(|| RenderError::Readback("Missing AOV texture".into()))?;
            out.insert(kind, readback::read_aov(kind, texture, width, height)?);
        }

        Ok(out)
    }
}
