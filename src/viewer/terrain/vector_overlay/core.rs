use super::*;

impl VectorOverlayStack {
    /// Create a new vector overlay stack
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            layers: Vec::new(),
            next_id: 0,
            enabled: true,
            dirty: false,
            global_opacity: 1.0,
            pipeline_triangles: None,
            pipeline_lines: None,
            pipeline_points: None,
            bind_group_layout: None,
            uniform_buffer: None,
            bind_group: None,
            sampler: None,
            oit_pipeline_triangles: None,
            oit_pipeline_lines: None,
            oit_pipeline_points: None,
        }
    }

    /// Add a vector overlay layer. Returns layer ID.
    pub fn add_layer(&mut self, layer: VectorOverlayLayer) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        // Create vertex buffer
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("vector_overlay_vertices_{}", id)),
                contents: bytemuck::cast_slice(&layer.vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        // Create index buffer
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("vector_overlay_indices_{}", id)),
                contents: bytemuck::cast_slice(&layer.indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            });

        let layer_name = layer.name.clone();
        let gpu_layer = VectorOverlayGpu {
            vertex_count: layer.vertices.len() as u32,
            index_count: layer.indices.len() as u32,
            config: layer,
            vertex_buffer,
            index_buffer,
            id,
        };

        self.layers.push(gpu_layer);
        self.dirty = true;

        // Sort by z_order
        self.layers.sort_by_key(|l| l.config.z_order);

        println!("[vector_overlay] Added layer '{layer_name}' (id={id})");
        id
    }

    /// Remove a vector overlay by ID. Returns true if found and removed.
    pub fn remove(&mut self, id: u32) -> bool {
        if let Some(pos) = self.layers.iter().position(|l| l.id == id) {
            self.layers.remove(pos);
            self.dirty = true;
            true
        } else {
            false
        }
    }

    /// Set vector overlay visibility
    pub fn set_visible(&mut self, id: u32, visible: bool) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
            layer.config.visible = visible;
            self.dirty = true;
        }
    }

    /// Set vector overlay opacity
    pub fn set_opacity(&mut self, id: u32, opacity: f32) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
            layer.config.opacity = opacity.clamp(0.0, 1.0);
            self.dirty = true;
        }
    }

    /// List all vector overlay IDs in z-order
    pub fn list_ids(&self) -> Vec<u32> {
        self.layers.iter().map(|l| l.id).collect()
    }

    /// Check if enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable the overlay system
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Set global opacity
    pub fn set_global_opacity(&mut self, opacity: f32) {
        self.global_opacity = opacity.clamp(0.0, 1.0);
    }

    /// Get visible layers in z-order
    pub fn visible_layers(&self) -> impl Iterator<Item = &VectorOverlayGpu> {
        self.layers.iter().filter(|l| l.config.visible)
    }
}
