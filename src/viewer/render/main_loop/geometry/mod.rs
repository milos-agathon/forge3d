mod fog;
mod fog_dispatch;
mod pass;

use glam::Mat4;

use super::RenderAvailability;
use crate::viewer::Viewer;

pub(super) fn mat4_to_array(m: Mat4) -> [[f32; 4]; 4] {
    let c = m.to_cols_array();
    [
        [c[0], c[1], c[2], c[3]],
        [c[4], c[5], c[6], c[7]],
        [c[8], c[9], c[10], c[11]],
        [c[12], c[13], c[14], c[15]],
    ]
}

impl Viewer {
    pub(super) fn render_geometry_stage(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderAvailability {
        let availability = self.geometry_availability();
        self.log_snapshot_geometry_gate(availability);
        self.log_missing_geometry_gate(availability);

        if self.geom_bind_group.is_none() {
            if let Err(err) = self.ensure_geom_bind_group() {
                eprintln!("[viewer] failed to build geometry bind group: {err}");
            }
        }

        if let (Some(mut gi), Some(zv), Some(_bgl)) = (
            self.gi.take(),
            self.z_view.take(),
            self.geom_bind_group_layout.as_ref(),
        ) {
            self.render_geometry_pass(encoder, &mut gi, &zv);
            self.render_geometry_fog(encoder, &mut gi);
            self.render_postfx_stage(&mut gi, encoder, &zv);
            self.gi = Some(gi);
            self.z_view = Some(zv);
        }

        availability
    }

    fn geometry_availability(&self) -> RenderAvailability {
        RenderAvailability {
            have_gi: self.gi.is_some(),
            have_pipe: self.geom_pipeline.is_some(),
            have_cam: self.geom_camera_buffer.is_some(),
            have_vb: self.geom_vb.is_some(),
            have_z: self.z_view.is_some(),
            have_bgl: self.geom_bind_group_layout.is_some(),
        }
    }

    fn log_snapshot_geometry_gate(&self, availability: RenderAvailability) {
        if self.snapshot_request.is_some() {
            let msg = format!(
                "[D1-GATE] frame={} gi={} pipe={} cam={} vb={} z={} bgl={} idx_cnt={} transform_identity={}\n",
                self.frame_count,
                availability.have_gi,
                availability.have_pipe,
                availability.have_cam,
                availability.have_vb,
                availability.have_z,
                availability.have_bgl,
                self.geom_index_count,
                self.object_transform == glam::Mat4::IDENTITY
            );
            let _ = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("examples/out/d1_debug.log")
                .and_then(|mut f| {
                    use std::io::Write;
                    f.write_all(msg.as_bytes())
                });
        }
    }

    fn log_missing_geometry_gate(&mut self, availability: RenderAvailability) {
        if !availability.ready() && !self.debug_logged_render_gate {
            eprintln!(
                "[viewer-debug] render gate: gi={} pipe={} cam={} vb={} z={} bgl={}",
                availability.have_gi,
                availability.have_pipe,
                availability.have_cam,
                availability.have_vb,
                availability.have_z,
                availability.have_bgl
            );
            self.debug_logged_render_gate = true;
        }
    }
}
