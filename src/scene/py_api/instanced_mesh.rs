use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // -----------------------------
    // F16: GPU Instanced Meshes API
    // -----------------------------
    #[cfg(feature = "enable-gpu-instancing")]
    #[pyo3(
        text_signature = "($self, positions, indices, transforms, normals=None, color=(0.85,0.85,0.9,1.0), light_dir=(0.3,0.7,0.2), light_intensity=1.2)"
    )]
    pub fn add_instanced_mesh(
        &mut self,
        positions: PyReadonlyArray2<'_, f32>,       // (Nv,3)
        indices: PyReadonlyArray2<'_, u32>,         // (Nt,3)
        transforms: PyReadonlyArray2<'_, f32>,      // (Ni,16) row-major
        normals: Option<PyReadonlyArray2<'_, f32>>, // (Nv,3) optional
        color: Option<(f32, f32, f32, f32)>,
        light_dir: Option<(f32, f32, f32)>,
        light_intensity: Option<f32>,
    ) -> PyResult<usize> {
        let pos = positions.as_array();
        if pos.ndim() != 2 || pos.shape()[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "positions must have shape (N,3)",
            ));
        }
        let idx = indices.as_array();
        if idx.ndim() != 2 || idx.shape()[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "indices must have shape (M,3)",
            ));
        }
        let trs = transforms.as_array();
        if trs.ndim() != 2 || trs.shape()[1] != 16 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "transforms must have shape (K,16) row-major 4x4",
            ));
        }

        let g = crate::core::gpu::ctx();

        // Build vertices (position, normal)
        #[cfg(feature = "enable-gpu-instancing")]
        use crate::render::mesh_instanced::VertexPN as Vpn;
        let nv = pos.shape()[0];
        let mut verts: Vec<Vpn> = Vec::with_capacity(nv);
        let n_opt = normals.as_ref().map(|n| n.as_array());
        for i in 0..nv {
            let p = [pos[[i, 0]], pos[[i, 1]], pos[[i, 2]]];
            let n = if let Some(nrm) = n_opt.as_ref() {
                if nrm.ndim() == 2 && nrm.shape()[1] == 3 {
                    [nrm[[i, 0]], nrm[[i, 1]], nrm[[i, 2]]]
                } else {
                    [0.0, 0.0, 1.0]
                }
            } else {
                [0.0, 0.0, 1.0]
            };
            verts.push(Vpn {
                position: p,
                normal: n,
            });
        }

        // Upload vertex/index buffers
        let vsize = (verts.len() * std::mem::size_of::<Vpn>()) as u64;
        let isize = (idx.len() * std::mem::size_of::<u32>()) as u64;
        let vbuf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-instanced-vbuf"),
            size: vsize,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ibuf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-instanced-ibuf"),
            size: isize,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        g.queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(&verts));
        // Flatten indices to u32
        let mut inds: Vec<u32> = Vec::with_capacity(idx.len());
        for t in idx.rows() {
            inds.push(t[0]);
            inds.push(t[1]);
            inds.push(t[2]);
        }
        g.queue.write_buffer(&ibuf, 0, bytemuck::cast_slice(&inds));

        // Instance buffer: pack row-major 4x4 to column-major (vec4 columns)
        let ni = trs.shape()[0];
        let mut packed: Vec<f32> = Vec::with_capacity(ni * 16);
        for i in 0..ni {
            let r = trs.row(i);
            let cm = [
                r[0], r[4], r[8], r[12], r[1], r[5], r[9], r[13], r[2], r[6], r[10], r[14], r[3],
                r[7], r[11], r[15],
            ];
            packed.extend_from_slice(&cm);
        }
        let instbuf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-instanced-instbuf"),
            size: (packed.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        g.queue
            .write_buffer(&instbuf, 0, bytemuck::cast_slice(&packed));

        // Ensure renderer exists (defensive)
        if self.mesh_instanced_renderer.is_none() {
            let depth_format = if self.sample_count > 1 {
                Some(wgpu::TextureFormat::Depth32Float)
            } else {
                None
            };
            self.mesh_instanced_renderer =
                Some(crate::render::mesh_instanced::MeshInstancedRenderer::new(
                    &g.device,
                    TEXTURE_FORMAT,
                    depth_format,
                ));
        }

        let batch = InstancedBatch {
            vbuf,
            ibuf,
            instbuf,
            index_count: inds.len() as u32,
            instance_count: ni as u32,
            color: color
                .map(|c| [c.0, c.1, c.2, c.3])
                .unwrap_or([0.85, 0.85, 0.9, 1.0]),
            light_dir: light_dir
                .map(|d| [d.0, d.1, d.2])
                .unwrap_or([0.3, 0.7, 0.2]),
            light_intensity: light_intensity.unwrap_or(1.2).max(0.0),
        };
        self.instanced_batches.push(batch);
        Ok(self.instanced_batches.len() - 1)
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[pyo3(text_signature = "($self)")]
    pub fn clear_instanced_meshes(&mut self) -> PyResult<()> {
        self.instanced_batches.clear();
        Ok(())
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[pyo3(text_signature = "($self, batch_index, transforms)")]
    pub fn update_instanced_transforms(
        &mut self,
        batch_index: usize,
        transforms: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<()> {
        if batch_index >= self.instanced_batches.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "instanced batch index out of range",
            ));
        }
        let trs = transforms.as_array();
        if trs.ndim() != 2 || trs.shape()[1] != 16 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "transforms must have shape (K,16) row-major 4x4",
            ));
        }
        let g = crate::core::gpu::ctx();
        let ni = trs.shape()[0];
        let mut packed: Vec<f32> = Vec::with_capacity(ni * 16);
        for i in 0..ni {
            let r = trs.row(i);
            packed.extend_from_slice(&[
                r[0], r[4], r[8], r[12], r[1], r[5], r[9], r[13], r[2], r[6], r[10], r[14], r[3],
                r[7], r[11], r[15],
            ]);
        }
        let b = &mut self.instanced_batches[batch_index];
        // Recreate buffer if needed (simplified: recreate always to match size)
        b.instbuf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-instanced-instbuf"),
            size: (packed.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        g.queue
            .write_buffer(&b.instbuf, 0, bytemuck::cast_slice(&packed));
        b.instance_count = ni as u32;
        Ok(())
    }
}
