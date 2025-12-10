/*!
 * Render Bundles types and configuration structures
 *
 * Contains configuration, statistics, and data structures for render bundles.
 */

use std::sync::Arc;

/// Buffer configuration for bundle
#[derive(Debug, Clone)]
pub struct BundleBuffer {
    pub usage: BundleBufferUsage,
    pub size: u64,
    pub data: Option<Vec<u8>>,
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
    pub size: wgpu::Extent3d,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
    pub label: Option<String>,
}

/// Draw call command for bundle
#[derive(Debug, Clone)]
pub struct BundleDrawCall {
    pub vertex_buffer_slot: u32,
    pub index_buffer_slot: Option<u32>,
    pub count: u32,
    pub offset: u32,
    pub instance_count: u32,
    pub first_instance: u32,
    pub bind_groups: Vec<u32>,
}

/// Bundle resource binding configuration
#[derive(Debug, Clone)]
pub struct BundleResourceConfig {
    pub buffers: Vec<BundleBuffer>,
    pub textures: Vec<BundleTexture>,
    pub bind_group_layouts: Vec<Arc<wgpu::BindGroupLayout>>,
}

impl Default for BundleResourceConfig {
    fn default() -> Self {
        Self {
            buffers: Vec::new(),
            textures: Vec::new(),
            bind_group_layouts: Vec::new(),
        }
    }
}

/// Render bundle configuration
#[derive(Debug, Clone)]
pub struct RenderBundleConfig {
    pub label: Option<String>,
    pub color_format: wgpu::TextureFormat,
    pub depth_format: Option<wgpu::TextureFormat>,
    pub sample_count: u32,
    pub resources: BundleResourceConfig,
    pub draw_calls: Vec<BundleDrawCall>,
}

impl Default for RenderBundleConfig {
    fn default() -> Self {
        Self {
            label: None,
            color_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            depth_format: Some(wgpu::TextureFormat::Depth32Float),
            sample_count: 1,
            resources: BundleResourceConfig::default(),
            draw_calls: Vec::new(),
        }
    }
}

/// Bundle rendering and performance statistics
#[derive(Debug, Clone, Default)]
pub struct BundleStats {
    pub draw_call_count: u32,
    pub total_vertices: u32,
    pub total_triangles: u32,
    pub memory_usage: u64,
    pub compile_time_ms: f32,
    pub execution_time_ms: f32,
    pub execution_count: u32,
}

/// Performance statistics for render bundle
#[derive(Debug, Clone)]
pub struct BundlePerformanceStats {
    pub avg_execution_time_ms: f32,
    pub min_execution_time_ms: f32,
    pub max_execution_time_ms: f32,
    pub std_dev_ms: f32,
    pub sample_count: usize,
}

/// Compiled render bundle with GPU resources
pub struct CompiledRenderBundle {
    pub bundle: wgpu::RenderBundle,
    pub config: RenderBundleConfig,
    pub buffers: Vec<wgpu::Buffer>,
    pub textures: Vec<wgpu::Texture>,
    pub bind_groups: Vec<wgpu::BindGroup>,
    pub stats: BundleStats,
}

impl RenderBundleConfig {
    /// Create configuration for instanced rendering
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
                count: index_data.len() as u32 / 4,
                offset: 0,
                instance_count,
                first_instance: 0,
                bind_groups: Vec::new(),
            }],
        }
    }

    /// Create configuration for UI rendering
    pub fn for_ui_rendering(quad_count: u32, color_format: wgpu::TextureFormat) -> Self {
        let draw_calls: Vec<_> = (0..quad_count)
            .map(|i| BundleDrawCall {
                vertex_buffer_slot: 0,
                index_buffer_slot: Some(1),
                count: 6,
                offset: 0,
                instance_count: 1,
                first_instance: i,
                bind_groups: vec![i],
            })
            .collect();

        Self {
            label: Some("UI Bundle".to_string()),
            color_format,
            depth_format: None,
            sample_count: 1,
            resources: BundleResourceConfig {
                buffers: vec![
                    BundleBuffer {
                        usage: BundleBufferUsage::Vertex,
                        size: 32 * 4,
                        data: None,
                        label: Some("UI Vertex Buffer".to_string()),
                    },
                    BundleBuffer {
                        usage: BundleBufferUsage::Index,
                        size: 6 * 4,
                        data: Some(
                            vec![0u32, 1, 2, 0, 2, 3]
                                .iter()
                                .flat_map(|&i| i.to_ne_bytes())
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
                    size: particle_count as u64 * 4 * 4,
                    data: None,
                    label: Some("Particle Vertex Buffer".to_string()),
                }],
                textures: Vec::new(),
                bind_group_layouts: Vec::new(),
            },
            draw_calls: vec![BundleDrawCall {
                vertex_buffer_slot: 0,
                index_buffer_slot: None,
                count: 4,
                offset: 0,
                instance_count: particle_count,
                first_instance: 0,
                bind_groups: vec![0],
            }],
        }
    }
}
