use std::path::Path;

use crate::p5::meta as p5_meta;
use crate::viewer::Viewer;
use anyhow::Context;

impl Viewer {
    /// Apply an intensity-only edit without leaving the current lighting mode.
    pub(crate) fn apply_lit_sun_intensity(&mut self, intensity: f32) {
        self.sky_lighting.set_intensity(intensity);
        self.update_lit_uniform();
        self.sync_terrain_sun_to_lit();
    }

    /// A successfully loaded terrain adopts the complete live lighting state.
    pub(crate) fn attach_terrain_lighting(&mut self) {
        self.sky_lighting.reset_terrain_source();
        self.update_lit_uniform();
        self.sync_terrain_sun_to_lit();
    }

    /// Apply one manual sun edit to every renderer-owned representation.
    /// Manual edits always leave observation replay and remove stale night
    /// instances so a later terrain command cannot silently diverge from the
    /// sky or lit-object state.
    pub(crate) fn apply_manual_sun(
        &mut self,
        azimuth_deg: f32,
        elevation_deg: f32,
        intensity: Option<f32>,
    ) {
        self.apply_manual_sun_with_source(azimuth_deg, elevation_deg, intensity, None);
    }

    pub(crate) fn apply_manual_sun_with_source(
        &mut self,
        azimuth_deg: f32,
        elevation_deg: f32,
        intensity: Option<f32>,
        source: Option<&str>,
    ) {
        self.sky_lighting
            .set_manual_sun(azimuth_deg, elevation_deg, intensity, source);
        self.update_lit_uniform();
        self.sync_terrain_sun_to_lit();
    }

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
        let sun_direction_vs = (self.current_frame_camera().view()
            * glam::Vec4::from((
                glam::Vec3::from_array(self.sky_lighting.light_direction()),
                0.0,
            )))
        .truncate()
        .normalize();
        let params: [f32; 12] = [
            sun_direction_vs.x,
            sun_direction_vs.y,
            sun_direction_vs.z,
            self.sky_lighting.effective_intensity(),
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

    pub(crate) fn sync_terrain_sun_to_lit(&mut self) {
        let (azimuth, elevation, intensity, source) = self.sky_lighting.terrain_sun();
        if let Some(terrain_viewer) = self.terrain_viewer.as_mut() {
            terrain_viewer.set_sun(azimuth, elevation, intensity, source);
        }
    }

    /// The GPU adapter publishes only complete state decisions.
    pub(crate) fn apply_sky_observation(
        &mut self,
        observation: &crate::astro::observation::ObservationSnapshot,
    ) {
        if self.sky_lighting.observe(observation) {
            self.queue.write_buffer(
                &self.night_instances,
                0,
                bytemuck::cast_slice(self.sky_lighting.night_instances()),
            );
            self.update_lit_uniform();
            self.sync_terrain_sun_to_lit();
        }
    }

    pub(crate) fn sync_astro_observation(&mut self) {
        if self.sky_lighting.is_observed() {
            if let Some(observation) = crate::astro::observation::current_observation() {
                self.apply_sky_observation(&observation);
            }
        }
    }
}
