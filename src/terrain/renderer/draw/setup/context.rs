use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::*;

pub(in crate::terrain::renderer) struct UploadedHeightInputs {
    pub(in crate::terrain::renderer) width: u32,
    pub(in crate::terrain::renderer) height: u32,
    pub(in crate::terrain::renderer) heightmap_data: Vec<f32>,
    pub(in crate::terrain::renderer) terrain_data_hash: u64,
    #[allow(dead_code)]
    pub(in crate::terrain::renderer) heightmap_texture: wgpu::Texture,
    pub(in crate::terrain::renderer) heightmap_view: wgpu::TextureView,
    #[allow(dead_code)]
    pub(in crate::terrain::renderer) water_mask_texture: Option<wgpu::Texture>,
    pub(in crate::terrain::renderer) water_mask_view_uploaded: Option<wgpu::TextureView>,
}

pub(in crate::terrain::renderer) struct PreparedMaterials {
    pub(in crate::terrain::renderer) gpu_materials:
        Arc<crate::render::material_set::GpuMaterialSet>,
    pub(in crate::terrain::renderer) shading_buffer: wgpu::Buffer,
    pub(in crate::terrain::renderer) overlay_buffer: wgpu::Buffer,
    pub(in crate::terrain::renderer) overlay_binding: OverlayBinding,
    pub(in crate::terrain::renderer) fallback_colormap_view: Option<wgpu::TextureView>,
}

impl PreparedMaterials {
    pub(in crate::terrain::renderer) fn material_view(&self) -> &wgpu::TextureView {
        &self.gpu_materials.view
    }

    pub(in crate::terrain::renderer) fn material_sampler(&self) -> &wgpu::Sampler {
        &self.gpu_materials.sampler
    }

    pub(in crate::terrain::renderer) fn colormap_view(&self) -> &wgpu::TextureView {
        if let Some(lut) = self.overlay_binding.lut.as_ref() {
            &lut.view
        } else {
            self.fallback_colormap_view.as_ref().unwrap()
        }
    }

    pub(in crate::terrain::renderer) fn colormap_sampler(&self) -> &wgpu::Sampler {
        if let Some(lut) = self.overlay_binding.lut.as_ref() {
            &lut.sampler
        } else {
            &self.gpu_materials.sampler
        }
    }
}

impl TerrainScene {
    pub(in crate::terrain::renderer) fn upload_height_inputs(
        &mut self,
        heightmap: PyReadonlyArray2<f32>,
        water_mask: Option<PyReadonlyArray2<f32>>,
        terrain_data_revision: Option<u64>,
    ) -> Result<UploadedHeightInputs> {
        let heightmap_array = heightmap.as_array();
        let (height, width) = (heightmap_array.shape()[0], heightmap_array.shape()[1]);
        if width == 0 || height == 0 {
            return Err(anyhow!("Heightmap dimensions must be > 0"));
        }

        let mut heightmap_data = Vec::with_capacity(width * height);
        let terrain_data_hash = if let Some(revision) = terrain_data_revision {
            // Caller-managed revisions let probe preparation skip per-sample hashing.
            for value in heightmap_array.iter() {
                heightmap_data.push(*value);
            }
            let mut hasher = DefaultHasher::new();
            (width as u32, height as u32).hash(&mut hasher);
            revision.hash(&mut hasher);
            hasher.finish()
        } else {
            let mut hasher = DefaultHasher::new();
            (width as u32, height as u32).hash(&mut hasher);
            for value in heightmap_array.iter() {
                let height = *value;
                heightmap_data.push(height);
                height.to_bits().hash(&mut hasher);
            }
            hasher.finish()
        };
        let heightmap_texture =
            self.upload_heightmap_texture(width as u32, height as u32, &heightmap_data)?;
        let heightmap_view = heightmap_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.compute_coarse_ao_from_heightmap(width as u32, height as u32, &heightmap_data)?;

        let (water_mask_texture, water_mask_view_uploaded) = if let Some(mask) = water_mask {
            let mask_array = mask.as_array();
            if mask_array.shape() == heightmap_array.shape() {
                let mut mask_bytes = Vec::with_capacity(width * height);
                let mut water_count = 0usize;
                let mut has_gradient = false;
                for value in mask_array.iter() {
                    let v = value.clamp(0.0, 1.0);
                    if v > 0.0 {
                        water_count += 1;
                        if v > 0.01 && v < 0.99 {
                            has_gradient = true;
                        }
                    }
                    mask_bytes.push((v * 255.0) as u8);
                }
                log::info!(
                    target: "terrain.water",
                    "Uploading water mask: {}x{}, {} water pixels ({:.2}%), distance_encoded={}",
                    width, height, water_count,
                    100.0 * water_count as f64 / (width * height) as f64,
                    has_gradient
                );
                let tex =
                    self.upload_water_mask_texture(width as u32, height as u32, &mask_bytes)?;
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                (Some(tex), Some(view))
            } else {
                log::warn!(
                    target: "terrain.water",
                    "Water mask shape {:?} does not match heightmap shape {:?}; using fallback",
                    mask_array.shape(),
                    heightmap_array.shape()
                );
                (None, None)
            }
        } else {
            (None, None)
        };

        Ok(UploadedHeightInputs {
            width: width as u32,
            height: height as u32,
            heightmap_data,
            terrain_data_hash,
            heightmap_texture,
            heightmap_view,
            water_mask_texture,
            water_mask_view_uploaded,
        })
    }

    pub(in crate::terrain::renderer) fn prepare_material_context(
        &self,
        material_set: &crate::render::material_set::MaterialSet,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
    ) -> Result<PreparedMaterials> {
        self.prepare_material_context_with_mode(material_set, params, decoded, false)
    }

    pub(in crate::terrain::renderer) fn prepare_material_context_with_mode(
        &self,
        material_set: &crate::render::material_set::MaterialSet,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        offline_hdr_output: bool,
    ) -> Result<PreparedMaterials> {
        let gpu_materials = material_set
            .gpu(self.device.as_ref(), self.queue.as_ref())
            .map_err(|err| {
                PyRuntimeError::new_err(format!("Failed to prepare material textures: {err:#}"))
            })?;

        let shading_uniforms =
            self.build_shading_uniforms(material_set, gpu_materials.as_ref(), params, decoded)?;
        let shading_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain.shading_buffer"),
                contents: bytemuck::cast_slice(&shading_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let overlay_binding = self.extract_overlay_binding(params, offline_hdr_output);
        self.log_color_debug(params, &overlay_binding);

        let overlay_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain.overlay_buffer"),
                contents: bytemuck::bytes_of(&overlay_binding.uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let fallback_colormap_view = if overlay_binding.lut.is_none() {
            Some(
                gpu_materials
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor {
                        label: Some("terrain.fallback.colormap.view"),
                        format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        aspect: wgpu::TextureAspect::All,
                        base_mip_level: 0,
                        mip_level_count: Some(1),
                        base_array_layer: 0,
                        array_layer_count: Some(1),
                    }),
            )
        } else {
            None
        };

        Ok(PreparedMaterials {
            gpu_materials,
            shading_buffer,
            overlay_buffer,
            overlay_binding,
            fallback_colormap_view,
        })
    }

    pub(in crate::terrain::renderer) fn prepare_ibl_bind_group(
        &self,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
    ) -> Result<wgpu::BindGroup> {
        let ibl_resources = env_maps.ensure_gpu_resources(&self.device, &self.queue)?;
        let (sin_theta, cos_theta) = env_maps.rotation_rad().sin_cos();
        let ibl_uniforms = IblUniforms {
            intensity: env_maps.intensity.max(0.0),
            sin_theta,
            cos_theta,
            specular_mip_count: ibl_resources.specular_mip_count.max(1) as f32,
        };
        let ibl_uniform_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("terrain.ibl_uniform_buffer"),
                    contents: bytemuck::bytes_of(&ibl_uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain_pbr_pom.ibl_bind_group"),
            layout: &self.ibl_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        ibl_resources.specular_view.as_ref(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        ibl_resources.irradiance_view.as_ref(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(ibl_resources.sampler.as_ref()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(ibl_resources.brdf_view.as_ref()),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: ibl_uniform_buffer.as_entire_binding(),
                },
            ],
        }))
    }
}
