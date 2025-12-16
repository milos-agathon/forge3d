/*!
 * Render Bundles implementation for GPU command optimization
 *
 * Provides reusable command buffers that group multiple draw calls
 * for improved rendering performance, especially for repeated geometry.
 */

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub use super::render_bundles_types::{
    BundleBuffer, BundleBufferUsage, BundleDrawCall, BundlePerformanceStats, BundleResourceConfig,
    BundleStats, BundleTexture, CompiledRenderBundle, RenderBundleConfig,
};

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

    pub fn add_vertex_buffer(&mut self, data: &[u8], label: Option<&str>) -> u32 {
        self.add_buffer(data, label, wgpu::BufferUsages::VERTEX)
    }

    pub fn add_index_buffer(&mut self, data: &[u8], label: Option<&str>) -> u32 {
        self.add_buffer(data, label, wgpu::BufferUsages::INDEX)
    }

    pub fn add_uniform_buffer(&mut self, data: &[u8], label: Option<&str>) -> u32 {
        self.add_buffer(
            data,
            label,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        )
    }

    fn add_buffer(&mut self, data: &[u8], label: Option<&str>, usage: wgpu::BufferUsages) -> u32 {
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: data,
                usage,
            });
        let slot = self.buffers.len() as u32;
        self.buffers.push(buffer);
        slot
    }

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

    pub fn build(self, render_pipeline: &wgpu::RenderPipeline) -> CompiledRenderBundle {
        let start_time = std::time::Instant::now();
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
        bundle_encoder.set_pipeline(render_pipeline);

        let mut stats = BundleStats {
            draw_call_count: self.config.draw_calls.len() as u32,
            ..Default::default()
        };
        self.encode_draw_calls(&mut bundle_encoder, &mut stats);
        let bundle = bundle_encoder.finish(&wgpu::RenderBundleDescriptor {
            label: self.config.label.as_deref(),
        });

        stats.memory_usage = self.calculate_memory_usage();
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

    fn encode_draw_calls<'a>(
        &'a self,
        encoder: &mut wgpu::RenderBundleEncoder<'a>,
        stats: &mut BundleStats,
    ) {
        for draw_call in &self.config.draw_calls {
            if let Some(vb) = self.buffers.get(draw_call.vertex_buffer_slot as usize) {
                encoder.set_vertex_buffer(0, vb.slice(..));
            }
            if let Some(idx) = draw_call.index_buffer_slot {
                if let Some(ib) = self.buffers.get(idx as usize) {
                    encoder.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                }
            }
            for (i, &bg_slot) in draw_call.bind_groups.iter().enumerate() {
                if let Some(bg) = self.bind_groups.get(bg_slot as usize) {
                    encoder.set_bind_group(i as u32, bg, &[]);
                }
            }
            let range = draw_call.offset..draw_call.offset + draw_call.count;
            let instances =
                draw_call.first_instance..draw_call.first_instance + draw_call.instance_count;
            if draw_call.index_buffer_slot.is_some() {
                encoder.draw_indexed(range, 0, instances);
                stats.total_triangles += draw_call.count / 3 * draw_call.instance_count;
            } else {
                encoder.draw(range, instances);
                stats.total_vertices += draw_call.count * draw_call.instance_count;
            }
        }
    }

    fn calculate_memory_usage(&self) -> u64 {
        let buf_mem: u64 = self.buffers.iter().map(|b| b.size()).sum();
        let tex_mem: u64 = self
            .textures
            .iter()
            .map(|t| {
                let e = t.size();
                let px = match t.format() {
                    wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm => 4,
                    wgpu::TextureFormat::Rgba16Float => 8,
                    wgpu::TextureFormat::Rgba32Float => 16,
                    wgpu::TextureFormat::Depth32Float => 4,
                    _ => 4,
                };
                (e.width * e.height * e.depth_or_array_layers) as u64 * px
            })
            .sum();
        buf_mem + tex_mem
    }
}

/// Render bundle manager for organizing and executing bundles
pub struct RenderBundleManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bundles: HashMap<String, CompiledRenderBundle>,
    execution_stats: HashMap<String, Vec<f32>>,
}

impl RenderBundleManager {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            bundles: HashMap::new(),
            execution_stats: HashMap::new(),
        }
    }

    pub fn add_bundle(&mut self, name: String, bundle: CompiledRenderBundle) {
        self.bundles.insert(name.clone(), bundle);
        self.execution_stats.insert(name, Vec::new());
    }

    pub fn create_bundle(
        &mut self,
        name: String,
        config: RenderBundleConfig,
        pipeline: &wgpu::RenderPipeline,
    ) -> Result<(), String> {
        let builder = RenderBundleBuilder::new(self.device.clone(), self.queue.clone(), config);
        self.add_bundle(name, builder.build(pipeline));
        Ok(())
    }

    pub fn execute_bundle<'a>(
        &'a mut self,
        pass: &mut wgpu::RenderPass<'a>,
        name: &str,
    ) -> Result<(), String> {
        let start = std::time::Instant::now();
        let bundle = self
            .bundles
            .get_mut(name)
            .ok_or_else(|| format!("Bundle '{}' not found", name))?;
        pass.execute_bundles([&bundle.bundle]);
        let exec_time = start.elapsed().as_secs_f32() * 1000.0;
        bundle.stats.execution_time_ms = exec_time;
        let times = self.execution_stats.get_mut(name).unwrap();
        times.push(exec_time);
        if times.len() > 100 {
            times.remove(0);
        }
        Ok(())
    }

    pub fn execute_bundles<'a>(
        &'a mut self,
        pass: &mut wgpu::RenderPass<'a>,
        names: &[&str],
    ) -> Result<(), String> {
        for &name in names {
            let b = self
                .bundles
                .get(name)
                .ok_or_else(|| format!("Bundle '{}' not found", name))?;
            pass.execute_bundles([&b.bundle]);
        }
        Ok(())
    }

    pub fn get_bundle_stats(&self, name: &str) -> Option<&BundleStats> {
        self.bundles.get(name).map(|b| &b.stats)
    }

    pub fn get_bundle_names(&self) -> Vec<&String> {
        self.bundles.keys().collect()
    }

    pub fn get_performance_stats(&self, name: &str) -> Option<BundlePerformanceStats> {
        let times = self.execution_stats.get(name)?;
        if times.is_empty() {
            return None;
        }
        let avg = times.iter().sum::<f32>() / times.len() as f32;
        let min = times.iter().copied().fold(f32::INFINITY, f32::min);
        let max = times.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let variance = times.iter().map(|&t| (t - avg).powi(2)).sum::<f32>() / times.len() as f32;
        Some(BundlePerformanceStats {
            avg_execution_time_ms: avg,
            min_execution_time_ms: min,
            max_execution_time_ms: max,
            std_dev_ms: variance.sqrt(),
            sample_count: times.len(),
        })
    }

    pub fn remove_bundle(&mut self, name: &str) -> bool {
        self.execution_stats.remove(name);
        self.bundles.remove(name).is_some()
    }

    pub fn clear(&mut self) {
        self.bundles.clear();
        self.execution_stats.clear();
    }
    pub fn get_total_memory_usage(&self) -> u64 {
        self.bundles.values().map(|b| b.stats.memory_usage).sum()
    }
    pub fn bundle_count(&self) -> usize {
        self.bundles.len()
    }
}
