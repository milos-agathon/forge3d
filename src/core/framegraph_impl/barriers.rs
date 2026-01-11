//! Framegraph barrier planning and resource transition management
//!
//! This module handles automatic insertion of barriers between passes
//! to ensure correct resource state transitions.

use super::types::{
    BarrierType, PassHandle, PassInfo, ResourceBarrier, ResourceHandle, ResourceInfo, ResourceType,
};
use std::collections::HashMap;
use wgpu::{BufferUsages, TextureUsages};

// ToPassHandle trait is defined below

/// Barrier planner for automatic resource state transitions
#[derive(Debug)]
pub struct BarrierPlanner {
    /// Current usage state of each resource
    resource_states: HashMap<ResourceHandle, ResourceUsage>,
}

/// Current usage state of a resource
#[derive(Debug, Clone)]
enum ResourceUsage {
    /// Texture usage state
    Texture(TextureUsages),
    /// Buffer usage state  
    Buffer(BufferUsages),
    /// Uninitialized/unknown state
    Unknown,
}

impl BarrierPlanner {
    /// Create a new barrier planner
    pub fn new() -> Self {
        Self {
            resource_states: HashMap::new(),
        }
    }

    /// Plan barriers for a sequence of passes
    pub fn plan_barriers(
        &mut self,
        passes: &[PassInfo],
        resources: &HashMap<ResourceHandle, ResourceInfo>,
    ) -> Vec<ResourceBarrier> {
        let mut barriers = Vec::new();

        // Reset resource states
        self.resource_states.clear();

        // Process each pass in execution order
        for pass_info in passes {
            let pass_barriers = self.plan_pass_barriers(pass_info, resources);
            barriers.extend(pass_barriers);

            // Update resource states after pass execution
            self.update_resource_states_after_pass(pass_info, resources);
        }

        barriers
    }

    /// Plan barriers needed before a specific pass
    fn plan_pass_barriers(
        &self,
        pass_info: &PassInfo,
        resources: &HashMap<ResourceHandle, ResourceInfo>,
    ) -> Vec<ResourceBarrier> {
        let mut barriers = Vec::new();

        // Check all resources used by this pass
        let mut all_resources = pass_info.desc.reads.clone();
        all_resources.extend(&pass_info.desc.writes);

        for &resource_handle in &all_resources {
            if let Some(resource_info) = resources.get(&resource_handle) {
                if let Some(barrier) = self.plan_resource_barrier(
                    resource_handle,
                    resource_info,
                    pass_info,
                    &pass_info.desc.reads,
                    &pass_info.desc.writes,
                ) {
                    barriers.push(barrier);
                }
            }
        }

        barriers
    }

    /// Plan barrier for a specific resource
    fn plan_resource_barrier(
        &self,
        resource_handle: ResourceHandle,
        resource_info: &ResourceInfo,
        pass_info: &PassInfo,
        reads: &[ResourceHandle],
        writes: &[ResourceHandle],
    ) -> Option<ResourceBarrier> {
        let current_usage = self
            .resource_states
            .get(&resource_handle)
            .cloned()
            .unwrap_or(ResourceUsage::Unknown);

        let required_usage =
            self.determine_required_usage(resource_handle, resource_info, reads, writes);

        // Check if we need a transition
        match (&current_usage, &required_usage) {
            (ResourceUsage::Texture(old), ResourceUsage::Texture(new)) => {
                if old != new {
                    Some(ResourceBarrier {
                        resource: resource_handle,
                        barrier_type: BarrierType::TextureBarrier {
                            old_usage: *old,
                            new_usage: *new,
                        },
                        before_pass: pass_info.desc.pass_type.to_pass_handle(),
                    })
                } else {
                    None
                }
            }
            (ResourceUsage::Buffer(old), ResourceUsage::Buffer(new)) => {
                if old != new {
                    Some(ResourceBarrier {
                        resource: resource_handle,
                        barrier_type: BarrierType::BufferBarrier {
                            old_usage: *old,
                            new_usage: *new,
                        },
                        before_pass: pass_info.desc.pass_type.to_pass_handle(),
                    })
                } else {
                    None
                }
            }
            (ResourceUsage::Unknown, _) => {
                // First use, no barrier needed but we'll track state
                None
            }
            _ => {
                // Mismatched resource types, insert memory barrier as fallback
                Some(ResourceBarrier {
                    resource: resource_handle,
                    barrier_type: BarrierType::MemoryBarrier,
                    before_pass: pass_info.desc.pass_type.to_pass_handle(),
                })
            }
        }
    }

    /// Determine the required usage for a resource in a pass
    fn determine_required_usage(
        &self,
        resource_handle: ResourceHandle,
        resource_info: &ResourceInfo,
        reads: &[ResourceHandle],
        writes: &[ResourceHandle],
    ) -> ResourceUsage {
        let _is_read = reads.contains(&resource_handle);
        let is_written = writes.contains(&resource_handle);

        match resource_info.desc.resource_type {
            ResourceType::ColorAttachment => {
                if is_written {
                    ResourceUsage::Texture(TextureUsages::RENDER_ATTACHMENT)
                } else {
                    ResourceUsage::Texture(TextureUsages::TEXTURE_BINDING)
                }
            }
            ResourceType::DepthStencilAttachment => {
                if is_written {
                    ResourceUsage::Texture(TextureUsages::RENDER_ATTACHMENT)
                } else {
                    ResourceUsage::Texture(TextureUsages::TEXTURE_BINDING)
                }
            }
            ResourceType::SampledTexture => ResourceUsage::Texture(TextureUsages::TEXTURE_BINDING),
            ResourceType::StorageBuffer => {
                if is_written {
                    ResourceUsage::Buffer(BufferUsages::STORAGE)
                } else {
                    ResourceUsage::Buffer(BufferUsages::STORAGE)
                }
            }
            ResourceType::UniformBuffer => ResourceUsage::Buffer(BufferUsages::UNIFORM),
        }
    }

    /// Update resource states after a pass completes
    fn update_resource_states_after_pass(
        &mut self,
        pass_info: &PassInfo,
        resources: &HashMap<ResourceHandle, ResourceInfo>,
    ) {
        // Update states for all resources used by this pass
        let mut all_resources = pass_info.desc.reads.clone();
        all_resources.extend(&pass_info.desc.writes);

        for &resource_handle in &all_resources {
            if let Some(resource_info) = resources.get(&resource_handle) {
                let new_usage = self.determine_required_usage(
                    resource_handle,
                    resource_info,
                    &pass_info.desc.reads,
                    &pass_info.desc.writes,
                );
                self.resource_states.insert(resource_handle, new_usage);
            }
        }
    }
}

impl Default for BarrierPlanner {
    fn default() -> Self {
        Self::new()
    }
}

// Helper trait to convert pass types to pass handles
trait ToPassHandle {
    fn to_pass_handle(&self) -> PassHandle;
}

impl ToPassHandle for super::types::PassType {
    fn to_pass_handle(&self) -> PassHandle {
        // Legacy pass types do not track handles yet; use a dummy handle.
        PassHandle(0)
    }
}
