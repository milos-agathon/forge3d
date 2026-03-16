use super::*;
use pyo3::types::PyDict;
use pyo3::{PyObject, PyResult, Python};

impl Scene {
    pub(super) fn get_stats_impl(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        let bytes_per_pixel_color = 4;
        let bytes_per_pixel_normal = 8;
        let bytes_per_pixel_depth = 4;

        let color_bytes = (self.width * self.height * bytes_per_pixel_color) as usize;
        let normal_bytes = (self.width * self.height * bytes_per_pixel_normal) as usize;
        let depth_bytes = if self.depth.is_some() {
            (self.width * self.height * self.sample_count * bytes_per_pixel_depth) as usize
        } else {
            0
        };
        let msaa_color_bytes = if self.sample_count > 1 {
            (self.width * self.height * self.sample_count * bytes_per_pixel_color) as usize
        } else {
            0
        };
        let msaa_normal_bytes = if self.sample_count > 1 {
            (self.width * self.height * self.sample_count * bytes_per_pixel_normal) as usize
        } else {
            0
        };
        let ssao_bytes = (self.ssao.width * self.ssao.height * 4) as usize * 2;
        let grid_verts = (self.grid * self.grid * 4 * 4) as usize;
        let grid_indices = ((self.grid - 1) * (self.grid - 1) * 6 * 4) as usize;

        let total_bytes = color_bytes
            + normal_bytes
            + depth_bytes
            + msaa_color_bytes
            + msaa_normal_bytes
            + ssao_bytes
            + grid_verts
            + grid_indices;
        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        let budget_mb = 512.0;

        dict.set_item("gpu_memory_mb", total_mb)?;
        dict.set_item("gpu_memory_peak_mb", total_mb)?;
        dict.set_item("gpu_memory_budget_mb", budget_mb)?;
        dict.set_item("gpu_utilization", total_mb / budget_mb)?;
        dict.set_item("frame_time_ms", 0.0)?;

        let mut passes: Vec<&str> = Vec::new();
        if self.terrain_enabled {
            passes.push("terrain");
        }
        if self.ssao_enabled {
            passes.push("ssao");
        }
        if self.reflections_enabled {
            passes.push("reflections");
        }
        if self.dof_enabled {
            passes.push("dof");
        }
        if self.cloud_shadows_enabled {
            passes.push("cloud_shadows");
        }
        if self.clouds_enabled {
            passes.push("clouds");
        }
        if self.ground_plane_enabled {
            passes.push("ground_plane");
        }
        if self.water_surface_enabled {
            passes.push("water_surface");
        }
        if self.soft_light_radius_enabled {
            passes.push("soft_light_radius");
        }
        if self.point_spot_lights_enabled {
            passes.push("point_spot_lights");
        }
        if self.ltc_area_lights_enabled {
            passes.push("ltc_area_lights");
        }
        if self.ibl_enabled {
            passes.push("ibl");
        }
        if self.dual_source_oit_enabled {
            passes.push("dual_source_oit");
        }
        if self.overlay_enabled {
            passes.push("overlay");
        }
        if self.text_overlay_enabled {
            passes.push("text_overlay");
        }
        if self.text3d_enabled {
            passes.push("text3d");
        }

        dict.set_item("passes_enabled", passes)?;
        Ok(dict.into())
    }
}
