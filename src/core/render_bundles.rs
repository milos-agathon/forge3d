/*!
 * Render Bundles implementation for GPU command optimization
 *
 * Provides reusable command buffers that group multiple draw calls
 * for improved rendering performance, especially for repeated geometry.
 */

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Bundle resource binding configuration
#[derive(Debug, Clone)]
pub struct BundleResourceConfig {
    /// Buffer bindings (vertex, index, uniform, etc.)
    pub buffers: Vec<BundleBuffer>,
    /// Texture bindings
    pub textures: Vec<BundleTexture>,
    /// Bind group layouts
    pub bind_group_layouts: Vec<Arc<wgpu::BindGroupLayout>>,
}

/// Buffer configuration for bundle
#[derive(Debug, Clone)]
pub struct BundleBuffer {
    /// Buffer usage type
    pub usage: BundleBufferUsage,
    /// Buffer size in bytes
    pub size: u64,
    /// Buffer data (optional, for static data)
    pub data: Option<Vec<u8>>,
    /// Buffer label for debugging
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BundleBufferUsage {
    Vertex,
    Index,
    Uniform,
    Storage,
}

/// Texture configuration for bundle
#[derive(Debug, Clone)]
pub struct BundleTexture {
    /// Texture dimensions
    pub size: wgpu::Extent3d,
    /// Texture format
    pub format: wgpu::TextureFormat,
    /// Texture usage flags
    pub usage: wgpu::TextureUsages,
    /// Texture label for debugging
    pub label: Option<String>,
}

/// Draw call command for bundle
#[derive(Debug, Clone)]
pub struct BundleDrawCall {
    /// Vertex buffer slot
    pub vertex_buffer_slot: u32,
    /// Index buffer (optional)
    pub index_buffer_slot: Option<u32>,
    /// Number of vertices or indices
    pub count: u32,
    /// First vertex/index offset
    pub offset: u32,
    /// Number of instances to draw
    pub instance_count: u32,
    /// First instance offset
    pub first_instance: u32,
    /// Bind groups to use
    pub bind_groups: Vec<u32>,
}

/// Render bundle configuration
#[derive(Debug, Clone)]
pub struct RenderBundleConfig {
    /// Bundle label for debugging
    pub label: Option<String>,
    /// Target render pass format
    pub color_format: wgpu::TextureFormat,
    /// Depth stencil format (optional)
    pub depth_format: Option<wgpu::TextureFormat>,
    /// Sample count for MSAA
    pub sample_count: u32,
    /// Resource configuration
    pub resources: BundleResourceConfig,
    /// Draw calls in this bundle
    pub draw_calls: Vec<BundleDrawCall>,
}

impl Default for RenderBundleConfig {
    fn default() -> Self {
        Self {
            label: None,
            color_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            depth_format: Some(wgpu::TextureFormat::Depth32Float),
            sample_count: 1,
            resources: BundleResourceConfig {
                buffers: Vec::new(),
                textures: Vec::new(),
                bind_group_layouts: Vec::new(),
            },
            draw_calls: Vec::new(),
        }
    }
}

/// Compiled render bundle with GPU resources
pub struct CompiledRenderBundle {
    /// The actual wgpu render bundle
    pub bundle: wgpu::RenderBundle,
    /// Bundle configuration
    pub config: RenderBundleConfig,
    /// GPU buffers created for this bundle
    pub buffers: Vec<wgpu::Buffer>,
    /// GPU textures created for this bundle
    pub textures: Vec<wgpu::Texture>,
    /// Bind groups created for this bundle
    pub bind_groups: Vec<wgpu::BindGroup>,
    /// Bundle statistics
    pub stats: BundleStats,
}

/// Bundle rendering and performance statistics
#[derive(Debug, Clone, Default)]
pub struct BundleStats {
    /// Number of draw calls in bundle
    pub draw_call_count: u32,
    /// Total vertex count across all draws
    pub total_vertices: u32,
    /// Total triangle count across all draws
    pub total_triangles: u32,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Compilation time in milliseconds
    pub compile_time_ms: f32,
    /// Last execution time in milliseconds
    pub execution_time_ms: f32,
    /// Number of times this bundle executed
    pub execution_count: u32,
}

/// Render bundle encoder for building bundles
pub struct RenderBundleBuilder {
    device: Arc<wgpu::Device>,
    #[allow(dead_code)]
    queue: Arc<wgpu::Queue>,
    config: RenderBundleConfig,
    buffers: Vec<wgpu::Buffer>,
    textures: Vec<wgpu::Texture>,
    bind_groups: Vec<wgpu::BindGroup>,
}

impl RenderBundleBuilder {
    /// Create new render bundle builder
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: RenderBundleConfig,
    ) -> Self {
        Self {
            device,
            queue,
            config,
            buffers: Vec::new(),
            textures: Vec::new(),
            bind_groups: Vec::new(),
        }
    }

    /// Add vertex buffer to bundle
    pub fn add_vertex_buffer(&mut self, data: &[u8], label: Option<&str>) -> u32 {
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: data,
                usage: wgpu::BufferUsages::VERTEX,
            });

        let slot = self.buffers.len() as u32;
        self.buffers.push(buffer);
        slot
    }

    /// Add index buffer to bundle
    pub fn add_index_buffer(&mut self, data: &[u8], label: Option<&str>) -> u32 {
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: data,
                usage: wgpu::BufferUsages::INDEX,
            });

        let slot = self.buffers.len() as u32;
        self.buffers.push(buffer);
        slot
    }

    /// Add uniform buffer to bundle
    pub fn add_uniform_buffer(&mut self, data: &[u8], label: Option<&str>) -> u32 {
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: data,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let slot = self.buffers.len() as u32;
        self.buffers.push(buffer);
        slot
    }

    /// Add texture to bundle
    pub fn add_texture(&mut self, config: BundleTexture) -> u32 {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: config.label.as_deref(),
            size: config.size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: config.usage,
            view_formats: &[],
        });

        let slot = self.textures.len() as u32;
        self.textures.push(texture);
        slot
    }

    /// Create bind group for bundle
    pub fn create_bind_group(
        &mut self,
        layout: &wgpu::BindGroupLayout,
        entries: &[wgpu::BindGroupEntry],
    ) -> u32 {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries,
        });

        let slot = self.bind_groups.len() as u32;
        self.bind_groups.push(bind_group);
        slot
    }

    /// Compile the render bundle
    pub fn build(self, render_pipeline: &wgpu::RenderPipeline) -> CompiledRenderBundle {
        let start_time = std::time::Instant::now();

        // Create render bundle encoder
        let mut bundle_encoder =
            self.device
                .create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                    label: self.config.label.as_deref(),
                    color_formats: &[Some(self.config.color_format)],
                    depth_stencil: self.config.depth_format.map(|format| {
                        wgpu::RenderBundleDepthStencil {
                            format,
                            depth_read_only: false,
                            stencil_read_only: false,
                        }
                    }),
                    sample_count: self.config.sample_count,
                    multiview: None,
                });

        // Set render pipeline
        bundle_encoder.set_pipeline(render_pipeline);

        // Calculate statistics
        let mut stats = BundleStats {
            draw_call_count: self.config.draw_calls.len() as u32,
            total_vertices: 0,
            total_triangles: 0,
            memory_usage: 0,
            compile_time_ms: 0.0,
            execution_time_ms: 0.0,
            execution_count: 0,
        };

        // Execute draw calls
        for draw_call in &self.config.draw_calls {
            // Set vertex buffer
            if let Some(vertex_buffer) = self.buffers.get(draw_call.vertex_buffer_slot as usize) {
                bundle_encoder.set_vertex_buffer(0, vertex_buffer.slice(..));
            }

            // Set index buffer if present
            if let Some(index_slot) = draw_call.index_buffer_slot {
                if let Some(index_buffer) = self.buffers.get(index_slot as usize) {
                    bundle_encoder
                        .set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                }
            }

            // Set bind groups
            for (group_index, &bind_group_slot) in draw_call.bind_groups.iter().enumerate() {
                if let Some(bind_group) = self.bind_groups.get(bind_group_slot as usize) {
                    bundle_encoder.set_bind_group(group_index as u32, bind_group, &[]);
                }
            }

            // Execute draw command
            if draw_call.index_buffer_slot.is_some() {
                bundle_encoder.draw_indexed(
                    draw_call.offset..draw_call.offset + draw_call.count,
                    0,
                    draw_call.first_instance..draw_call.first_instance + draw_call.instance_count,
                );
                stats.total_triangles += draw_call.count / 3 * draw_call.instance_count;
            } else {
                bundle_encoder.draw(
                    draw_call.offset..draw_call.offset + draw_call.count,
                    draw_call.first_instance..draw_call.first_instance + draw_call.instance_count,
                );
                stats.total_vertices += draw_call.count * draw_call.instance_count;
            }
        }

        // Finish bundle creation
        let bundle = bundle_encoder.finish(&wgpu::RenderBundleDescriptor {
            label: self.config.label.as_deref(),
        });

        // Calculate memory usage
        for buffer in &self.buffers {
            stats.memory_usage += buffer.size();
        }
        for texture in &self.textures {
            // Approximate texture memory usage
            let extent = texture.size();
            let pixel_size = match texture.format() {
                wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm => 4,
                wgpu::TextureFormat::Rgba16Float => 8,
                wgpu::TextureFormat::Rgba32Float => 16,
                wgpu::TextureFormat::Depth32Float => 4,
                _ => 4, // Default estimate
            };
            stats.memory_usage +=
                (extent.width * extent.height * extent.depth_or_array_layers) as u64 * pixel_size;
        }

        stats.compile_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        CompiledRenderBundle {
            bundle,
            config: self.config,
            buffers: self.buffers,
            textures: self.textures,
            bind_groups: self.bind_groups,
            stats,
        }
    }
}

/// Render bundle manager for organizing and executing bundles
pub struct RenderBundleManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bundles: HashMap<String, CompiledRenderBundle>,
    execution_stats: HashMap<String, Vec<f32>>, // Rolling execution times
}

impl RenderBundleManager {
    /// Create new render bundle manager
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            bundles: HashMap::new(),
            execution_stats: HashMap::new(),
        }
    }

    /// Add compiled bundle to manager
    pub fn add_bundle(&mut self, name: String, bundle: CompiledRenderBundle) {
        self.bundles.insert(name.clone(), bundle);
        self.execution_stats.insert(name, Vec::new());
    }

    /// Create and add bundle from configuration
    pub fn create_bundle(
        &mut self,
        name: String,
        config: RenderBundleConfig,
        render_pipeline: &wgpu::RenderPipeline,
    ) -> Result<(), String> {
        let builder = RenderBundleBuilder::new(self.device.clone(), self.queue.clone(), config);

        let bundle = builder.build(render_pipeline);
        self.add_bundle(name, bundle);
        Ok(())
    }

    /// Execute bundle by name in render pass
    pub fn execute_bundle<'a>(
        &'a mut self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bundle_name: &str,
    ) -> Result<(), String> {
        let start_time = std::time::Instant::now();

        if let Some(bundle) = self.bundles.get_mut(bundle_name) {
            render_pass.execute_bundles([&bundle.bundle]);

            let execution_time = start_time.elapsed().as_secs_f32() * 1000.0;
            bundle.stats.execution_time_ms = execution_time;

            // Track execution times for performance monitoring
            let times = self.execution_stats.get_mut(bundle_name).unwrap();
            times.push(execution_time);
            if times.len() > 100 {
                times.remove(0); // Keep rolling window
            }

            Ok(())
        } else {
            Err(format!("Bundle '{}' not found", bundle_name))
        }
    }

    /// Execute multiple bundles in sequence
    pub fn execute_bundles<'a>(
        &'a mut self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bundle_names: &[&str],
    ) -> Result<(), String> {
        for &bundle_name in bundle_names {
            if let Some(bundle_ref) = self.bundles.get(bundle_name) {
                render_pass.execute_bundles([&bundle_ref.bundle]);
            } else {
                return Err(format!("Bundle '{}' not found", bundle_name));
            }
        }
        Ok(())
    }

    /// Get bundle statistics
    pub fn get_bundle_stats(&self, bundle_name: &str) -> Option<&BundleStats> {
        self.bundles.get(bundle_name).map(|b| &b.stats)
    }

    /// Get all bundle names
    pub fn get_bundle_names(&self) -> Vec<&String> {
        self.bundles.keys().collect()
    }

    /// Get performance statistics for bundle
    pub fn get_performance_stats(&self, bundle_name: &str) -> Option<BundlePerformanceStats> {
        if let Some(times) = self.execution_stats.get(bundle_name) {
            if times.is_empty() {
                return None;
            }

            let avg_time = times.iter().sum::<f32>() / times.len() as f32;
            let min_time = times.iter().copied().fold(f32::INFINITY, f32::min);
            let max_time = times.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            // Calculate standard deviation
            let variance =
                times.iter().map(|&t| (t - avg_time).powi(2)).sum::<f32>() / times.len() as f32;
            let std_dev = variance.sqrt();

            Some(BundlePerformanceStats {
                avg_execution_time_ms: avg_time,
                min_execution_time_ms: min_time,
                max_execution_time_ms: max_time,
                std_dev_ms: std_dev,
                sample_count: times.len(),
            })
        } else {
            None
        }
    }

    /// Remove bundle from manager
    pub fn remove_bundle(&mut self, bundle_name: &str) -> bool {
        self.execution_stats.remove(bundle_name);
        self.bundles.remove(bundle_name).is_some()
    }

    /// Clear all bundles
    pub fn clear(&mut self) {
        self.bundles.clear();
        self.execution_stats.clear();
    }

    /// Get total memory usage of all bundles
    pub fn get_total_memory_usage(&self) -> u64 {
        self.bundles
            .values()
            .map(|bundle| bundle.stats.memory_usage)
            .sum()
    }

    /// Get bundle count
    pub fn bundle_count(&self) -> usize {
        self.bundles.len()
    }
}

/// Performance statistics for render bundle
#[derive(Debug, Clone)]
pub struct BundlePerformanceStats {
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f32,
    /// Minimum execution time in milliseconds
    pub min_execution_time_ms: f32,
    /// Maximum execution time in milliseconds
    pub max_execution_time_ms: f32,
    /// Standard deviation in milliseconds
    pub std_dev_ms: f32,
    /// Number of execution samples
    pub sample_count: usize,
}

/// Utility functions for creating common bundle configurations
impl RenderBundleConfig {
    /// Create configuration for instanced rendering (many objects, same geometry)
    pub fn for_instanced_rendering(
        vertex_data: &[u8],
        index_data: &[u8],
        instance_count: u32,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            label: Some("Instanced Bundle".to_string()),
            color_format,
            depth_format: Some(wgpu::TextureFormat::Depth32Float),
            sample_count: 1,
            resources: BundleResourceConfig {
                buffers: vec![
                    BundleBuffer {
                        usage: BundleBufferUsage::Vertex,
                        size: vertex_data.len() as u64,
                        data: Some(vertex_data.to_vec()),
                        label: Some("Vertex Buffer".to_string()),
                    },
                    BundleBuffer {
                        usage: BundleBufferUsage::Index,
                        size: index_data.len() as u64,
                        data: Some(index_data.to_vec()),
                        label: Some("Index Buffer".to_string()),
                    },
                ],
                textures: Vec::new(),
                bind_group_layouts: Vec::new(),
            },
            draw_calls: vec![BundleDrawCall {
                vertex_buffer_slot: 0,
                index_buffer_slot: Some(1),
                count: index_data.len() as u32 / 4, // Assume 32-bit indices
                offset: 0,
                instance_count,
                first_instance: 0,
                bind_groups: Vec::new(),
            }],
        }
    }

    /// Create configuration for UI rendering (many quads, different textures)
    pub fn for_ui_rendering(quad_count: u32, color_format: wgpu::TextureFormat) -> Self {
        let mut draw_calls = Vec::new();

        // Create draw call for each quad
        for i in 0..quad_count {
            draw_calls.push(BundleDrawCall {
                vertex_buffer_slot: 0,      // Shared vertex buffer
                index_buffer_slot: Some(1), // Shared index buffer
                count: 6,                   // 2 triangles per quad
                offset: 0,
                instance_count: 1,
                first_instance: i,
                bind_groups: vec![i], // Different texture per quad
            });
        }

        Self {
            label: Some("UI Bundle".to_string()),
            color_format,
            depth_format: None, // UI usually doesn't need depth
            sample_count: 1,
            resources: BundleResourceConfig {
                buffers: vec![
                    BundleBuffer {
                        usage: BundleBufferUsage::Vertex,
                        size: 32 * 4, // 4 vertices * 8 floats * 4 bytes
                        data: None,   // Will be filled by builder
                        label: Some("UI Vertex Buffer".to_string()),
                    },
                    BundleBuffer {
                        usage: BundleBufferUsage::Index,
                        size: 6 * 4, // 6 indices * 4 bytes
                        data: Some(
                            vec![0, 1, 2, 0, 2, 3]
                                .iter()
                                .flat_map(|&i| (i as u32).to_ne_bytes())
                                .collect(),
                        ),
                        label: Some("UI Index Buffer".to_string()),
                    },
                ],
                textures: Vec::new(),
                bind_group_layouts: Vec::new(),
            },
            draw_calls,
        }
    }

    /// Create configuration for particle rendering
    pub fn for_particle_rendering(particle_count: u32, color_format: wgpu::TextureFormat) -> Self {
        Self {
            label: Some("Particle Bundle".to_string()),
            color_format,
            depth_format: Some(wgpu::TextureFormat::Depth32Float),
            sample_count: 1,
            resources: BundleResourceConfig {
                buffers: vec![BundleBuffer {
                    usage: BundleBufferUsage::Vertex,
                    size: particle_count as u64 * 4 * 4, // 4 vertices per particle, 4 floats per vertex
                    data: None,
                    label: Some("Particle Vertex Buffer".to_string()),
                }],
                textures: Vec::new(),
                bind_group_layouts: Vec::new(),
            },
            draw_calls: vec![BundleDrawCall {
                vertex_buffer_slot: 0,
                index_buffer_slot: None,
                count: 4, // 4 vertices per particle (quad)
                offset: 0,
                instance_count: particle_count,
                first_instance: 0,
                bind_groups: vec![0], // Shared texture atlas
            }],
        }
    }
}
