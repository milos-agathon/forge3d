use super::effect::PostFxEffect;
use super::resources::PostFxResourcePool;
use crate::core::error::RenderResult;
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
