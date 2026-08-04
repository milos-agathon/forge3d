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
        let sun_direction_vs = (self.current_frame_camera().view()
            * glam::Vec4::from((glam::Vec3::from_array(self.lit_sun_direction_ws), 0.0)))
        .truncate()
        .normalize();
        let params: [f32; 12] = [
            sun_direction_vs.x,
            sun_direction_vs.y,
            sun_direction_vs.z,
            if self.lit_sun_direction_ws[1] > 0.0 {
                self.lit_sun_intensity * self.lit_directional_scale
            } else {
                0.0
            },
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
        let direction = glam::Vec3::from_array(self.lit_sun_direction_ws).normalize();
        let azimuth = direction
            .x
            .atan2(direction.z)
            .to_degrees()
            .rem_euclid(360.0);
        let elevation = direction.y.clamp(-1.0, 1.0).asin().to_degrees();
        if let Some(terrain_viewer) = self.terrain_viewer.as_mut() {
            terrain_viewer.set_sun(
                azimuth,
                elevation,
                if elevation > 0.0 {
                    self.lit_sun_intensity * self.lit_directional_scale
                } else {
                    0.0
                },
            );
        }
    }

    pub(crate) fn sync_astro_observation(&mut self) {
        if !self.astro_observation_active {
            return;
        }
        let Some(observation) = crate::astro::observation::current_observation() else {
            return;
        };
        let sun = observation.bodies[0].1;
        let azimuth = sun.azimuth.value() as f32;
        let elevation = sun.altitude.value() as f32;
        if observation.revision != self.astro_night_revision {
            let instances = crate::astro::night::instances(&observation);
            self.queue.write_buffer(
                &self.night_instances,
                0,
                bytemuck::cast_slice(instances.as_slice()),
            );
            self.night_instance_count = instances.len() as u32;
            self.astro_night_revision = observation.revision;
        }
        if observation.revision != self.astro_observation_revision {
            let azimuth_rad = azimuth.to_radians();
            let elevation_rad = elevation.to_radians();
            self.sky_sun_direction_ws = [
                elevation_rad.cos() * azimuth_rad.sin(),
                elevation_rad.sin(),
                elevation_rad.cos() * azimuth_rad.cos(),
            ];
            let moon = observation.bodies[1].1;
            if elevation > 0.0 {
                self.lit_sun_direction_ws = self.sky_sun_direction_ws;
                self.lit_directional_scale = 1.0;
            } else {
                self.lit_sun_direction_ws = crate::astro::night::horizontal_direction(
                    moon.azimuth.value(),
                    moon.altitude.value(),
                )
                .as_vec3()
                .to_array();
                self.lit_directional_scale = crate::astro::night::normalized_moonlight(
                    observation.moon_phase.phase_angle.value(),
                    moon.altitude.value(),
                );
            }
            self.update_lit_uniform();
            self.astro_observation_revision = observation.revision;
        }
        if observation.revision != self.astro_terrain_revision {
            if self.terrain_viewer.is_some() {
                self.sync_terrain_sun_to_lit();
                self.astro_terrain_revision = observation.revision;
            }
        }
    }
}
