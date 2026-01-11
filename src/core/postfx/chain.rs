use super::effect::PostFxEffect;
use super::resources::PostFxResourcePool;
use crate::core::async_compute::AsyncComputeConfig;
use crate::core::error::{RenderError, RenderResult};
use crate::core::gpu_timing::GpuTimingManager;
use std::collections::{HashMap, VecDeque};
use wgpu::*;

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
            // Mutate config in-place because the effect owns its config.
            // In a fuller implementation, configs would live separately.
            // Enabled state is represented by presence in the execution order.
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
            // No effects: skip copy to avoid an extra pass.
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
