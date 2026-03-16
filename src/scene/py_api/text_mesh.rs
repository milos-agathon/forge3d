use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // -----------------------------
    // D11: 3D Text Meshes API
    // -----------------------------
    #[pyo3(text_signature = "($self)")]
    pub fn enable_text_meshes(&mut self) -> PyResult<()> {
        self.text3d_enabled = true;
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_text_meshes(&mut self) -> PyResult<()> {
        self.text3d_enabled = false;
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn clear_text_meshes(&mut self) -> PyResult<()> {
        self.text3d_instances.clear();
        Ok(())
    }

    /// Add a 3D text mesh instance from font bytes
    ///
    /// font can be either bytes (PyBytes) or a 1D numpy uint8 array.
    #[pyo3(
        text_signature = "($self, text, font, size_px=32.0, depth=0.2, position=(0,0,0), color=(1,1,1,1), rotation_deg=(0,0,0), scale=1.0, scale_xyz=(1,1,1), light_dir=(0.5,1.0,0.3), light_intensity=1.0, bevel_strength=0.0, bevel_segments=3)"
    )]
    pub fn add_text_mesh(
        &mut self,
        _py: pyo3::Python<'_>,
        text: String,
        font: &pyo3::types::PyAny,
        size_px: Option<f32>,
        depth: Option<f32>,
        position: Option<(f32, f32, f32)>,
        color: Option<(f32, f32, f32, f32)>,
        rotation_deg: Option<(f32, f32, f32)>,
        scale: Option<f32>,
        scale_xyz: Option<(f32, f32, f32)>,
        light_dir: Option<(f32, f32, f32)>,
        light_intensity: Option<f32>,
        bevel_strength: Option<f32>,
        bevel_segments: Option<u32>,
    ) -> PyResult<()> {
        // Extract font bytes
        let font_bytes: Vec<u8> = if let Ok(b) = font.extract::<&PyBytes>() {
            b.as_bytes().to_vec()
        } else if let Ok(arr) = font.extract::<PyReadonlyArray1<u8>>() {
            arr.as_slice()
                .map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err("font array must be C-contiguous uint8")
                })?
                .to_vec()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "font must be bytes or numpy uint8 array",
            ));
        };

        let sz = size_px.unwrap_or(32.0).max(1.0);
        let dp = depth.unwrap_or(0.2).max(0.0);
        let pos = position.unwrap_or((0.0, 0.0, 0.0));
        let col = color.unwrap_or((1.0, 1.0, 1.0, 1.0));
        let rot = rotation_deg.unwrap_or((0.0, 0.0, 0.0));
        let scl = scale.unwrap_or(1.0).max(1e-6);
        let sxyz = scale_xyz.unwrap_or((1.0, 1.0, 1.0));
        let svec = glam::Vec3::new(sxyz.0 * scl, sxyz.1 * scl, sxyz.2 * scl);
        let ldir = light_dir.unwrap_or((0.5, 1.0, 0.3));
        let lint = light_intensity.unwrap_or(1.0).max(0.0);
        let bevel = bevel_strength.unwrap_or(0.0);
        let bev_segs = bevel_segments.unwrap_or(3).max(1);

        // Build mesh on CPU
        let (verts, inds) =
            crate::core::text_mesh::build_text_mesh(&text, &font_bytes, sz, dp, bevel, bev_segs)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "text mesh build failed: {}",
                        e
                    ))
                })?;

        // Upload to GPU
        let g = crate::core::gpu::ctx();
        let vsize = (verts.len() * std::mem::size_of::<crate::core::text_mesh::VertexPN>()) as u64;
        let isize = (inds.len() * std::mem::size_of::<u32>()) as u64;
        let vbuf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("text3d_vbuf"),
            size: vsize,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ibuf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("text3d_ibuf"),
            size: isize,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        g.queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(&verts));
        g.queue.write_buffer(&ibuf, 0, bytemuck::cast_slice(&inds));

        // Create instance with model transform: T * Rz * Ry * Rx * S
        let rx = rot.0.to_radians();
        let ry = rot.1.to_radians();
        let rz = rot.2.to_radians();
        let t = glam::Mat4::from_translation(glam::Vec3::new(pos.0, pos.1, pos.2));
        let sx = glam::Mat4::from_scale(svec);
        let rr = glam::Mat4::from_rotation_z(rz)
            * glam::Mat4::from_rotation_y(ry)
            * glam::Mat4::from_rotation_x(rx);
        let model = t * rr * sx;
        let inst = Text3DInstance {
            vbuf,
            ibuf,
            index_count: inds.len() as u32,
            vertex_count: verts.len() as u32,
            model,
            color: [col.0, col.1, col.2, col.3],
            light_dir: [ldir.0, ldir.1, ldir.2],
            light_intensity: lint,
            metallic: 0.0,
            roughness: 1.0,
        };
        self.text3d_instances.push(inst);
        self.text3d_enabled = true;
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_text_mesh_stats(&self) -> PyResult<(usize, u64, u64)> {
        let instances = self.text3d_instances.len();
        let mut v: u64 = 0;
        let mut i: u64 = 0;
        for inst in &self.text3d_instances {
            v += inst.vertex_count as u64;
            i += inst.index_count as u64;
        }
        Ok((instances, v, i))
    }

    #[pyo3(text_signature = "($self, index, position, rotation_deg, scale=None, scale_xyz=None)")]
    pub fn update_text_mesh_transform(
        &mut self,
        index: usize,
        position: (f32, f32, f32),
        rotation_deg: (f32, f32, f32),
        scale: Option<f32>,
        scale_xyz: Option<(f32, f32, f32)>,
    ) -> PyResult<()> {
        if index >= self.text3d_instances.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "text mesh index out of range",
            ));
        }
        let rx = rotation_deg.0.to_radians();
        let ry = rotation_deg.1.to_radians();
        let rz = rotation_deg.2.to_radians();
        let t = glam::Mat4::from_translation(glam::Vec3::new(position.0, position.1, position.2));
        let s = scale.unwrap_or(1.0).max(1e-6);
        let sxyz = scale_xyz.unwrap_or((1.0, 1.0, 1.0));
        let svec = glam::Vec3::new(sxyz.0 * s, sxyz.1 * s, sxyz.2 * s);
        let sx = glam::Mat4::from_scale(svec);
        let rr = glam::Mat4::from_rotation_z(rz)
            * glam::Mat4::from_rotation_y(ry)
            * glam::Mat4::from_rotation_x(rx);
        let model = t * rr * sx;
        if let Some(inst) = self.text3d_instances.get_mut(index) {
            inst.model = model;
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, index, r, g, b, a)")]
    pub fn update_text_mesh_color(
        &mut self,
        index: usize,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    ) -> PyResult<()> {
        if index >= self.text3d_instances.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "text mesh index out of range",
            ));
        }
        if let Some(inst) = self.text3d_instances.get_mut(index) {
            inst.color = [r, g, b, a];
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, index, dx, dy, dz, intensity)")]
    pub fn update_text_mesh_light(
        &mut self,
        index: usize,
        dx: f32,
        dy: f32,
        dz: f32,
        intensity: f32,
    ) -> PyResult<()> {
        if index >= self.text3d_instances.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "text mesh index out of range",
            ));
        }
        if let Some(inst) = self.text3d_instances.get_mut(index) {
            inst.light_dir = [dx, dy, dz];
            inst.light_intensity = intensity.max(0.0);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, index, metallic, roughness)")]
    pub fn set_text_mesh_material(
        &mut self,
        index: usize,
        metallic: f32,
        roughness: f32,
    ) -> PyResult<()> {
        if index >= self.text3d_instances.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "text mesh index out of range",
            ));
        }
        if let Some(inst) = self.text3d_instances.get_mut(index) {
            inst.metallic = metallic.clamp(0.0, 1.0);
            inst.roughness = roughness.clamp(0.04, 1.0);
        }
        Ok(())
    }
}
