use super::*;

/// PBR material with GPU resources
#[derive(Debug)]
pub struct PbrMaterialGpu {
    /// Material properties
    pub material: PbrMaterial,

    /// Material uniform buffer
    pub uniform_buffer: Buffer,

    /// PBR textures
    pub textures: PbrTextures,

    /// Texture views for binding
    pub texture_views: HashMap<String, TextureView>,

    /// Material bind group
    pub bind_group: Option<BindGroup>,
}

impl PbrMaterialGpu {
    /// Create PBR material GPU resources
    pub fn new(device: &Device, material: PbrMaterial) -> Self {
        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("pbr_material_uniforms"),
            size: std::mem::size_of::<PbrMaterial>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            material,
            uniform_buffer,
            textures: PbrTextures {
                base_color: None,
                metallic_roughness: None,
                normal: None,
                occlusion: None,
                emissive: None,
            },
            texture_views: HashMap::new(),
            bind_group: None,
        }
    }

    /// Update material properties on GPU
    pub fn update_uniforms(&self, queue: &Queue) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.material]),
        );
    }

    /// Set base color texture
    pub fn set_base_color_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        texture_data: &[u8],
        width: u32,
        height: u32,
    ) {
        let texture = create_texture_from_data(
            device,
            queue,
            "pbr_base_color",
            texture_data,
            width,
            height,
            TextureFormat::Rgba8UnormSrgb, // sRGB for color textures
        );

        let view = texture.create_view(&TextureViewDescriptor::default());

        self.textures.base_color = Some(texture);
        self.texture_views.insert("base_color".to_string(), view);
        self.material.texture_flags |= texture_flags::BASE_COLOR;
    }

    /// Set metallic-roughness texture
    pub fn set_metallic_roughness_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        texture_data: &[u8],
        width: u32,
        height: u32,
    ) {
        let texture = create_texture_from_data(
            device,
            queue,
            "pbr_metallic_roughness",
            texture_data,
            width,
            height,
            TextureFormat::Rgba8Unorm, // Linear for material properties
        );

        let view = texture.create_view(&TextureViewDescriptor::default());

        self.textures.metallic_roughness = Some(texture);
        self.texture_views
            .insert("metallic_roughness".to_string(), view);
        self.material.texture_flags |= texture_flags::METALLIC_ROUGHNESS;
    }

    /// Set normal map texture
    pub fn set_normal_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        texture_data: &[u8],
        width: u32,
        height: u32,
    ) {
        let texture = create_texture_from_data(
            device,
            queue,
            "pbr_normal",
            texture_data,
            width,
            height,
            TextureFormat::Rgba8Unorm, // Linear for normal maps
        );

        let view = texture.create_view(&TextureViewDescriptor::default());

        self.textures.normal = Some(texture);
        self.texture_views.insert("normal".to_string(), view);
        self.material.texture_flags |= texture_flags::NORMAL;
    }

    /// Set occlusion texture
    pub fn set_occlusion_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        texture_data: &[u8],
        width: u32,
        height: u32,
    ) {
        let texture = create_texture_from_data(
            device,
            queue,
            "pbr_occlusion",
            texture_data,
            width,
            height,
            TextureFormat::R8Unorm, // Single channel for AO
        );

        let view = texture.create_view(&TextureViewDescriptor::default());

        self.textures.occlusion = Some(texture);
        self.texture_views.insert("occlusion".to_string(), view);
        self.material.texture_flags |= texture_flags::OCCLUSION;
    }

    /// Set emissive texture
    pub fn set_emissive_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        texture_data: &[u8],
        width: u32,
        height: u32,
    ) {
        let texture = create_texture_from_data(
            device,
            queue,
            "pbr_emissive",
            texture_data,
            width,
            height,
            TextureFormat::Rgba8UnormSrgb, // sRGB for emissive colors
        );

        let view = texture.create_view(&TextureViewDescriptor::default());

        self.textures.emissive = Some(texture);
        self.texture_views.insert("emissive".to_string(), view);
        self.material.texture_flags |= texture_flags::EMISSIVE;
    }

    /// Create bind group for material
    pub fn create_bind_group(
        &mut self,
        device: &Device,
        queue: &Queue,
        layout: &wgpu::BindGroupLayout,
        sampler: &Sampler,
    ) {
        // Create default textures for missing ones
        let default_white =
            create_default_texture(device, queue, "default_white", [255, 255, 255, 255]);
        let default_normal =
            create_default_texture(device, queue, "default_normal", [128, 128, 255, 255]);
        let default_metallic_roughness =
            create_default_texture(device, queue, "default_mr", [0, 255, 0, 255]); // No metallic, full roughness
        let default_black = create_default_texture(device, queue, "default_black", [0, 0, 0, 0]);

        // Create views for default textures
        let default_white_view = default_white.create_view(&TextureViewDescriptor::default());
        let default_normal_view = default_normal.create_view(&TextureViewDescriptor::default());
        let default_mr_view =
            default_metallic_roughness.create_view(&TextureViewDescriptor::default());
        let default_black_view = default_black.create_view(&TextureViewDescriptor::default());

        let base_color_view = self
            .texture_views
            .get("base_color")
            .unwrap_or(&default_white_view);
        let metallic_roughness_view = self
            .texture_views
            .get("metallic_roughness")
            .unwrap_or(&default_mr_view);
        let normal_view = self
            .texture_views
            .get("normal")
            .unwrap_or(&default_normal_view);
        let occlusion_view = self
            .texture_views
            .get("occlusion")
            .unwrap_or(&default_white_view);
        let emissive_view = self
            .texture_views
            .get("emissive")
            .unwrap_or(&default_black_view);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("pbr_material_bind_group"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(base_color_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(metallic_roughness_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(normal_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(occlusion_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(emissive_view),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::Sampler(sampler),
                },
            ],
        });

        self.bind_group = Some(bind_group);
    }
}
