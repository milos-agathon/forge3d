//! Q1: Post-processing compute pipeline
//!
//! Provides a flexible effect chain manager for post-processing operations including
//! temporal effects, ping-pong resource management, and GPU compute-based filters.

use crate::core::async_compute::AsyncComputeConfig;
use crate::core::gpu_timing::GpuTimingManager;
use crate::error::{RenderError, RenderResult};
use std::collections::{HashMap, VecDeque};
use wgpu::*;

/// Post-processing effect configuration
#[derive(Debug, Clone)]
pub struct PostFxConfig {
    /// Effect name/identifier
    pub name: String,
    /// Whether the effect is enabled
    pub enabled: bool,
    /// Effect-specific parameters
    pub parameters: HashMap<String, f32>,
    /// Priority for effect ordering (higher = later)
    pub priority: i32,
    /// Whether this effect needs temporal data
    pub temporal: bool,
    /// Number of ping-pong buffers needed
    pub ping_pong_count: u32,
}

impl Default for PostFxConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            enabled: true,
            parameters: HashMap::new(),
            priority: 0,
            temporal: false,
            ping_pong_count: 2,
        }
    }
}

/// Resource descriptor for post-processing textures
#[derive(Debug, Clone)]
pub struct PostFxResourceDesc {
    /// Resource width (0 = match input)
    pub width: u32,
    /// Resource height (0 = match input)  
    pub height: u32,
    /// Texture format
    pub format: TextureFormat,
    /// Usage flags
    pub usage: TextureUsages,
    /// Mip level count
    pub mip_count: u32,
    /// Sample count for MSAA
    pub sample_count: u32,
}

impl Default for PostFxResourceDesc {
    fn default() -> Self {
        Self {
            width: 0,  // Match input
            height: 0, // Match input
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            mip_count: 1,
            sample_count: 1,
        }
    }
}

/// Resource pool for ping-pong and temporal textures
#[derive(Debug)]
pub struct PostFxResourcePool {
    /// Ping-pong texture pairs
    ping_pong_textures: Vec<Vec<Texture>>,
    /// Texture views for ping-pong resources
    ping_pong_views: Vec<Vec<TextureView>>,
    /// Temporal textures (for effects that need history)
    temporal_textures: HashMap<String, Vec<Texture>>,
    /// Temporal texture views
    temporal_views: HashMap<String, Vec<TextureView>>,
    /// Current ping-pong index
    ping_pong_index: usize,
    /// Resource dimensions
    width: u32,
    height: u32,
}

impl PostFxResourcePool {
    /// Create new resource pool
    pub fn new(_device: &Device, width: u32, height: u32, max_ping_pong_pairs: usize) -> Self {
        Self {
            ping_pong_textures: Vec::with_capacity(max_ping_pong_pairs),
            ping_pong_views: Vec::with_capacity(max_ping_pong_pairs),
            temporal_textures: HashMap::new(),
            temporal_views: HashMap::new(),
            ping_pong_index: 0,
            width,
            height,
        }
    }

    /// Get current ping-pong texture
    pub fn get_current_ping_pong(&self, pair_index: usize) -> Option<&TextureView> {
        self.ping_pong_views
            .get(pair_index)?
            .get(self.ping_pong_index)
    }

    /// Get previous ping-pong texture
    pub fn get_previous_ping_pong(&self, pair_index: usize) -> Option<&TextureView> {
        let prev_index = (self.ping_pong_index + 1) % 2;
        self.ping_pong_views.get(pair_index)?.get(prev_index)
    }

    /// Swap ping-pong buffers
    pub fn swap_ping_pong(&mut self) {
        self.ping_pong_index = (self.ping_pong_index + 1) % 2;
    }

    /// Allocate ping-pong texture pair
    pub fn allocate_ping_pong_pair(
        &mut self,
        device: &Device,
        desc: &PostFxResourceDesc,
    ) -> RenderResult<usize> {
        let actual_width = if desc.width == 0 {
            self.width
        } else {
            desc.width
        };
        let actual_height = if desc.height == 0 {
            self.height
        } else {
            desc.height
        };

        let mut textures = Vec::new();
        let mut views = Vec::new();

        // Create pair of textures
        for i in 0..2 {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some(&format!(
                    "postfx_ping_pong_{}_{}",
                    self.ping_pong_textures.len(),
                    i
                )),
                size: Extent3d {
                    width: actual_width,
                    height: actual_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: desc.mip_count,
                sample_count: desc.sample_count,
                dimension: TextureDimension::D2,
                format: desc.format,
                usage: desc.usage,
                view_formats: &[],
            });

            let view = texture.create_view(&TextureViewDescriptor::default());

            textures.push(texture);
            views.push(view);
        }

        let pair_index = self.ping_pong_textures.len();
        self.ping_pong_textures.push(textures);
        self.ping_pong_views.push(views);

        Ok(pair_index)
    }

    /// Allocate temporal texture
    pub fn allocate_temporal_texture(
        &mut self,
        device: &Device,
        name: &str,
        frame_count: usize,
        desc: &PostFxResourceDesc,
    ) -> RenderResult<()> {
        let actual_width = if desc.width == 0 {
            self.width
        } else {
            desc.width
        };
        let actual_height = if desc.height == 0 {
            self.height
        } else {
            desc.height
        };

        let mut textures = Vec::new();
        let mut views = Vec::new();

        for i in 0..frame_count {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some(&format!("postfx_temporal_{}_{}", name, i)),
                size: Extent3d {
                    width: actual_width,
                    height: actual_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: desc.mip_count,
                sample_count: desc.sample_count,
                dimension: TextureDimension::D2,
                format: desc.format,
                usage: desc.usage,
                view_formats: &[],
            });

            let view = texture.create_view(&TextureViewDescriptor::default());

            textures.push(texture);
            views.push(view);
        }

        self.temporal_textures.insert(name.to_string(), textures);
        self.temporal_views.insert(name.to_string(), views);

        Ok(())
    }

    /// Get temporal texture by name and frame index
    pub fn get_temporal_texture(&self, name: &str, frame_index: usize) -> Option<&TextureView> {
        self.temporal_views.get(name)?.get(frame_index)
    }
}

/// Post-processing effect definition
pub trait PostFxEffect: Send + Sync {
    /// Get effect name
    fn name(&self) -> &str;

    /// Get effect configuration
    fn config(&self) -> &PostFxConfig;

    /// Set effect parameter
    fn set_parameter(&mut self, name: &str, value: f32) -> RenderResult<()>;

    /// Get effect parameter
    fn get_parameter(&self, name: &str) -> Option<f32>;

    /// Initialize effect resources
    fn initialize(
        &mut self,
        device: &Device,
        resource_pool: &mut PostFxResourcePool,
    ) -> RenderResult<()>;

    /// Execute effect compute pass
    fn execute(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        input: &TextureView,
        output: &TextureView,
        resource_pool: &PostFxResourcePool,
        timing_manager: Option<&mut GpuTimingManager>,
    ) -> RenderResult<()>;

    /// Cleanup effect resources
    fn cleanup(&mut self) -> RenderResult<()> {
        Ok(())
    }
}

/// Post-processing effect chain manager
pub struct PostFxChain {
    /// Registered effects
    effects: HashMap<String, Box<dyn PostFxEffect>>,
    /// Effect execution order
    execution_order: VecDeque<String>,
    /// Resource pool for ping-pong and temporal textures
    resource_pool: PostFxResourcePool,
    /// Async compute configuration
    #[allow(dead_code)]
    async_config: AsyncComputeConfig,
    /// Whether chain is enabled
    enabled: bool,
    /// Chain timing statistics
    timing_stats: HashMap<String, f32>,
}

impl PostFxChain {
    /// Create new post-processing chain
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        let max_ping_pong_pairs = 8; // Reasonable default
        let resource_pool = PostFxResourcePool::new(device, width, height, max_ping_pong_pairs);

        Self {
            effects: HashMap::new(),
            execution_order: VecDeque::new(),
            resource_pool,
            async_config: AsyncComputeConfig::default(),
            enabled: true,
            timing_stats: HashMap::new(),
        }
    }

    /// Register a post-processing effect
    pub fn register_effect(
        &mut self,
        mut effect: Box<dyn PostFxEffect>,
        device: &Device,
    ) -> RenderResult<()> {
        let name = effect.name().to_string();

        // Initialize effect resources
        effect.initialize(device, &mut self.resource_pool)?;

        // Insert in priority order
        let priority = effect.config().priority;
        let mut insert_index = self.execution_order.len();

        for (i, existing_name) in self.execution_order.iter().enumerate() {
            if let Some(existing_effect) = self.effects.get(existing_name) {
                if existing_effect.config().priority > priority {
                    insert_index = i;
                    break;
                }
            }
        }

        self.execution_order.insert(insert_index, name.clone());
        self.effects.insert(name, effect);

        Ok(())
    }

    /// Unregister an effect
    pub fn unregister_effect(&mut self, name: &str) -> RenderResult<()> {
        if let Some(mut effect) = self.effects.remove(name) {
            effect.cleanup()?;
            self.execution_order.retain(|n| n != name);
        }

        Ok(())
    }

    /// Enable/disable an effect
    pub fn set_effect_enabled(&mut self, name: &str, enabled: bool) -> RenderResult<()> {
        if let Some(effect) = self.effects.get_mut(name) {
            // Modify the effect's config - this is a bit hacky since we need mutable access
            // In a real implementation, we might store configs separately
            // For now, we'll track enabled state in execution order
            if enabled && !self.execution_order.contains(&name.to_string()) {
                let priority = effect.config().priority;
                let mut insert_index = self.execution_order.len();

                for (i, existing_name) in self.execution_order.iter().enumerate() {
                    if let Some(existing_effect) = self.effects.get(existing_name) {
                        if existing_effect.config().priority > priority {
                            insert_index = i;
                            break;
                        }
                    }
                }

                self.execution_order.insert(insert_index, name.to_string());
            } else if !enabled {
                self.execution_order.retain(|n| n != name);
            }
        }

        Ok(())
    }

    /// Set effect parameter
    pub fn set_effect_parameter(
        &mut self,
        effect_name: &str,
        param_name: &str,
        value: f32,
    ) -> RenderResult<()> {
        if let Some(effect) = self.effects.get_mut(effect_name) {
            effect.set_parameter(param_name, value)?;
        }
        Ok(())
    }

    /// Get effect parameter
    pub fn get_effect_parameter(&self, effect_name: &str, param_name: &str) -> Option<f32> {
        self.effects.get(effect_name)?.get_parameter(param_name)
    }

    /// Execute the entire post-processing chain
    pub fn execute_chain(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        input: &TextureView,
        output: &TextureView,
        mut timing_manager: Option<&mut GpuTimingManager>,
    ) -> RenderResult<()> {
        if !self.enabled || self.execution_order.is_empty() {
            // No effects - copy input to output
            // TODO: Implement blit/copy operation
            return Ok(());
        }

        let chain_scope = if let Some(timer) = timing_manager.as_mut() {
            Some(timer.begin_scope(encoder, "postfx_chain"))
        } else {
            None
        };

        // Execute effects in order
        for (i, effect_name) in self.execution_order.iter().enumerate() {
            if let Some(effect) = self.effects.get(effect_name) {
                let is_last = i == self.execution_order.len() - 1;
                let is_first = i == 0;

                if is_last {
                    // Execute final effect to output
                    let effect_input = if is_first {
                        input
                    } else {
                        // Use previous ping-pong buffer as input
                        self.resource_pool
                            .get_previous_ping_pong(0)
                            .ok_or_else(|| {
                                RenderError::Render(
                                    "No previous ping-pong buffer available".to_string(),
                                )
                            })?
                    };

                    effect.execute(
                        device,
                        encoder,
                        effect_input,
                        output,
                        &self.resource_pool,
                        None,
                    )?;
                } else {
                    // Execute intermediate effect with ping-pong buffers
                    let effect_input = if is_first {
                        input
                    } else {
                        // Use previous ping-pong buffer as input
                        self.resource_pool
                            .get_previous_ping_pong(0)
                            .ok_or_else(|| {
                                RenderError::Render(
                                    "No previous ping-pong buffer available".to_string(),
                                )
                            })?
                    };

                    // Use current ping-pong buffer as output
                    let ping_pong_output =
                        self.resource_pool.get_current_ping_pong(0).ok_or_else(|| {
                            RenderError::Render("No current ping-pong buffer available".to_string())
                        })?;

                    effect.execute(
                        device,
                        encoder,
                        effect_input,
                        ping_pong_output,
                        &self.resource_pool,
                        None,
                    )?;

                    // Swap ping-pong buffers for next effect
                    self.resource_pool.swap_ping_pong();
                }
            }
        }

        // End chain timing scope
        if let (Some(timer), Some(scope_id)) = (timing_manager, chain_scope) {
            timer.end_scope(encoder, scope_id);
        }

        Ok(())
    }

    /// Get list of registered effects
    pub fn list_effects(&self) -> Vec<String> {
        self.effects.keys().cloned().collect()
    }

    /// Get list of enabled effects in execution order
    pub fn list_enabled_effects(&self) -> Vec<String> {
        self.execution_order.iter().cloned().collect()
    }

    /// Enable/disable entire chain
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if chain is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get timing statistics for effects
    pub fn get_timing_stats(&self) -> &HashMap<String, f32> {
        &self.timing_stats
    }

    /// Update timing statistics
    pub fn update_timing_stats(&mut self, stats: HashMap<String, f32>) {
        self.timing_stats = stats;
    }
}

/// No-op application hook for integration points in the render path.
pub fn postfx_apply_noop(_device: &Device, _encoder: &mut CommandEncoder, _input: &TextureView) {}
