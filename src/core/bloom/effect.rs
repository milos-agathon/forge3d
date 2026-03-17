use crate::core::error::RenderResult;
use crate::core::gpu_timing::GpuTimingManager;
use crate::core::postfx::{PostFxConfig, PostFxEffect, PostFxResourcePool};

use super::execute::execute_effect;
use super::init::initialize_effect;
use super::{BloomConfig, BloomEffect};

impl BloomEffect {
    /// Create a new bloom effect
    pub fn new() -> Self {
        let mut config = PostFxConfig::default();
        config.name = "bloom".to_string();
        config.priority = 800;
        config.temporal = false;
        config.ping_pong_count = 2;

        let bloom_config = BloomConfig::default();
        config
            .parameters
            .insert("threshold".to_string(), bloom_config.threshold);
        config
            .parameters
            .insert("softness".to_string(), bloom_config.softness);
        config
            .parameters
            .insert("strength".to_string(), bloom_config.strength);
        config
            .parameters
            .insert("radius".to_string(), bloom_config.radius);
        config.parameters.insert(
            "enabled".to_string(),
            if bloom_config.enabled { 1.0 } else { 0.0 },
        );

        Self {
            config,
            bloom_config,
            brightpass_pipeline: None,
            blur_h_pipeline: None,
            blur_v_pipeline: None,
            composite_pipeline: None,
            brightpass_layout: None,
            blur_layout: None,
            composite_layout: None,
            brightpass_uniform_buffer: None,
            blur_uniform_buffer: None,
            composite_uniform_buffer: None,
            brightpass_texture_index: None,
            blur_temp_texture_index: None,
        }
    }

    /// Update the bloom configuration.
    pub fn update_bloom_config(&mut self, bloom_config: BloomConfig) {
        self.bloom_config = bloom_config;
        self.config
            .parameters
            .insert("threshold".to_string(), bloom_config.threshold);
        self.config
            .parameters
            .insert("softness".to_string(), bloom_config.softness);
        self.config
            .parameters
            .insert("strength".to_string(), bloom_config.strength);
        self.config
            .parameters
            .insert("radius".to_string(), bloom_config.radius);
        self.config.parameters.insert(
            "enabled".to_string(),
            if bloom_config.enabled { 1.0 } else { 0.0 },
        );
    }

    /// Return the current bloom configuration.
    pub fn bloom_config(&self) -> &BloomConfig {
        &self.bloom_config
    }
}

impl PostFxEffect for BloomEffect {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn config(&self) -> &PostFxConfig {
        &self.config
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> RenderResult<()> {
        self.config.parameters.insert(name.to_string(), value);
        match name {
            "threshold" => self.bloom_config.threshold = value,
            "softness" => self.bloom_config.softness = value,
            "strength" => self.bloom_config.strength = value,
            "radius" => self.bloom_config.radius = value,
            "enabled" => self.bloom_config.enabled = value > 0.5,
            _ => {}
        }
        Ok(())
    }

    fn get_parameter(&self, name: &str) -> Option<f32> {
        self.config.parameters.get(name).copied()
    }

    fn initialize(
        &mut self,
        device: &wgpu::Device,
        resource_pool: &mut PostFxResourcePool,
    ) -> RenderResult<()> {
        initialize_effect(self, device, resource_pool)
    }

    fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
        resource_pool: &PostFxResourcePool,
        timing_manager: Option<&mut GpuTimingManager>,
    ) -> RenderResult<()> {
        execute_effect(
            self,
            device,
            queue,
            encoder,
            input,
            output,
            resource_pool,
            timing_manager,
        )
    }
}
