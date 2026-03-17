use std::path::Path;

use crate::p5::meta as p5_meta;
use crate::viewer::Viewer;
use anyhow::Context;

impl Viewer {
    pub(crate) fn write_p5_meta<
        F: FnOnce(&mut std::collections::BTreeMap<String, serde_json::Value>),
    >(
        &self,
        patch: F,
    ) -> anyhow::Result<()> {
        p5_meta::write_p5_meta(Path::new("reports/p5"), patch)
    }

    pub(crate) fn sync_ssr_params_to_gi(&mut self) {
        if let Some(ref mut gi) = self.gi {
            gi.set_ssr_params(&self.queue, &self.ssr_params);
        }
    }

    pub(crate) fn with_comp_pipeline<T>(
        &self,
        f: impl FnOnce(&wgpu::RenderPipeline, &wgpu::BindGroupLayout) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let comp_pl = self.comp_pipeline.as_ref().context("comp pipeline")?;
        let comp_bgl = self.comp_bind_group_layout.as_ref().context("comp bgl")?;
        f(comp_pl, comp_bgl)
    }

    pub(crate) fn update_lit_uniform(&mut self) {
        let sun_dir = [0.3f32, 0.6, -1.0];
        let params: [f32; 12] = [
            sun_dir[0],
            sun_dir[1],
            sun_dir[2],
            self.lit_sun_intensity,
            self.lit_ibl_intensity,
            if self.lit_use_ibl { 1.0 } else { 0.0 },
            self.lit_brdf as f32,
            0.0,
            self.lit_roughness.clamp(0.0, 1.0),
            self.lit_debug_mode as f32,
            0.0,
            0.0,
        ];
        self.queue
            .write_buffer(&self.lit_uniform, 0, bytemuck::cast_slice(&params));
    }
}
