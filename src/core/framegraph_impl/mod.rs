//! Framegraph module for render pass organization and resource management
//!
//! This module provides a framegraph system for automatic resource lifetime management,
//! transient resource aliasing, and barrier insertion between render passes.

pub mod barriers;
pub mod types;

use crate::error::{RenderError, RenderResult};
use std::collections::HashMap;

use barriers::BarrierPlanner;
pub use types::{
    FrameGraphMetrics, PassDesc, PassHandle, PassInfo, PassType, ResourceBarrier, ResourceDesc,
    ResourceHandle, ResourceInfo, ResourceType,
};

/// Pass builder for configuring render passes
pub struct PassBuilder {
    desc: PassDesc,
}

impl PassBuilder {
    /// Create a new pass builder
    fn new(name: String, pass_type: PassType) -> Self {
        Self {
            desc: PassDesc {
                name,
                pass_type,
                reads: Vec::new(),
                writes: Vec::new(),
                can_parallelize: false,
            },
        }
    }

    /// Mark a resource as read by this pass
    pub fn read(&mut self, resource: ResourceHandle) -> &mut Self {
        self.desc.reads.push(resource);
        self
    }

    /// Mark a resource as written by this pass
    pub fn write(&mut self, resource: ResourceHandle) -> &mut Self {
        self.desc.writes.push(resource);
        self
    }

    /// Allow this pass to run in parallel with others
    pub fn allow_parallel(&mut self) -> &mut Self {
        self.desc.can_parallelize = true;
        self
    }
}

/// Main framegraph for managing render passes and resources
#[derive(Debug)]
pub struct FrameGraph {
    /// All resources in the graph
    resources: HashMap<ResourceHandle, ResourceInfo>,
    /// All passes in the graph  
    passes: HashMap<PassHandle, PassInfo>,
    /// Next resource ID to assign
    next_resource_id: usize,
    /// Next pass ID to assign
    next_pass_id: usize,
    /// Barrier planner for automatic transitions
    barrier_planner: BarrierPlanner,
    /// Execution metrics
    metrics: FrameGraphMetrics,
}

impl FrameGraph {
    /// Create a new framegraph
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
            passes: HashMap::new(),
            next_resource_id: 0,
            next_pass_id: 0,
            barrier_planner: BarrierPlanner::new(),
            metrics: FrameGraphMetrics::default(),
        }
    }

    /// Add a resource to the framegraph
    pub fn add_resource(&mut self, desc: ResourceDesc) -> ResourceHandle {
        let handle = ResourceHandle(self.next_resource_id);
        self.next_resource_id += 1;

        let info = ResourceInfo {
            desc,
            first_use: None,
            last_use: None,
            is_transient: true,
            aliased_with: None,
        };

        self.resources.insert(handle, info);
        handle
    }

    /// Add a render pass to the framegraph
    pub fn add_pass<F>(
        &mut self,
        name: &str,
        pass_type: PassType,
        setup: F,
    ) -> RenderResult<PassHandle>
    where
        F: FnOnce(&mut PassBuilder) -> RenderResult<()>,
    {
        let handle = PassHandle(self.next_pass_id);
        self.next_pass_id += 1;

        let mut builder = PassBuilder::new(name.to_string(), pass_type);
        setup(&mut builder)?;

        // Update resource usage information
        for &resource_handle in &builder.desc.reads {
            if let Some(resource_info) = self.resources.get_mut(&resource_handle) {
                if resource_info.first_use.is_none() {
                    resource_info.first_use = Some(handle);
                }
                resource_info.last_use = Some(handle);
            }
        }

        for &resource_handle in &builder.desc.writes {
            if let Some(resource_info) = self.resources.get_mut(&resource_handle) {
                if resource_info.first_use.is_none() {
                    resource_info.first_use = Some(handle);
                }
                resource_info.last_use = Some(handle);
            }
        }

        let info = PassInfo {
            desc: builder.desc,
            dependencies: Vec::new(),
            dependents: Vec::new(),
        };

        self.passes.insert(handle, info);
        Ok(handle)
    }

    /// Compile the framegraph and perform optimizations
    pub fn compile(&mut self) -> RenderResult<()> {
        // Build dependency graph
        self.build_dependencies()?;

        // Perform transient resource aliasing
        self.alias_transient_resources()?;

        // Update metrics
        self.update_metrics();

        Ok(())
    }

    /// Get execution plan with barriers
    pub fn get_execution_plan(&mut self) -> RenderResult<(Vec<PassHandle>, Vec<ResourceBarrier>)> {
        // Topological sort of passes
        let sorted_passes = self.topological_sort()?;

        // Convert to PassInfo vec for barrier planning
        let pass_infos: Vec<_> = sorted_passes
            .iter()
            .filter_map(|&handle| self.passes.get(&handle))
            .cloned()
            .collect();

        // Plan barriers
        let barriers = self
            .barrier_planner
            .plan_barriers(&pass_infos, &self.resources);

        Ok((sorted_passes, barriers))
    }

    /// Get metrics from the last compilation
    pub fn metrics(&self) -> &FrameGraphMetrics {
        &self.metrics
    }

    /// Reset the framegraph for a new frame
    pub fn reset(&mut self) {
        self.resources.clear();
        self.passes.clear();
        self.next_resource_id = 0;
        self.next_pass_id = 0;
        self.metrics = FrameGraphMetrics::default();
    }

    /// Build dependency relationships between passes
    fn build_dependencies(&mut self) -> RenderResult<()> {
        // Clear existing dependencies
        for pass_info in self.passes.values_mut() {
            pass_info.dependencies.clear();
            pass_info.dependents.clear();
        }

        // Find dependencies based on resource usage
        let pass_handles: Vec<_> = self.passes.keys().cloned().collect();

        for &pass_a in &pass_handles {
            for &pass_b in &pass_handles {
                if pass_a == pass_b {
                    continue;
                }

                if self.has_dependency(pass_a, pass_b)? {
                    // pass_a depends on pass_b
                    if let Some(info_a) = self.passes.get_mut(&pass_a) {
                        if !info_a.dependencies.contains(&pass_b) {
                            info_a.dependencies.push(pass_b);
                        }
                    }
                    if let Some(info_b) = self.passes.get_mut(&pass_b) {
                        if !info_b.dependents.contains(&pass_a) {
                            info_b.dependents.push(pass_a);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if pass_a depends on pass_b
    fn has_dependency(&self, pass_a: PassHandle, pass_b: PassHandle) -> RenderResult<bool> {
        let info_a = self
            .passes
            .get(&pass_a)
            .ok_or_else(|| RenderError::render("Invalid pass handle A"))?;
        let info_b = self
            .passes
            .get(&pass_b)
            .ok_or_else(|| RenderError::render("Invalid pass handle B"))?;

        // Check if pass_a reads something that pass_b writes
        for &read_resource in &info_a.desc.reads {
            if info_b.desc.writes.contains(&read_resource) {
                return Ok(true);
            }
        }

        // Check if pass_a writes something that pass_b reads (reverse dependency)
        for &write_resource in &info_a.desc.writes {
            if info_b.desc.reads.contains(&write_resource) {
                return Ok(false); // This would be pass_b depends on pass_a
            }
        }

        Ok(false)
    }

    /// Perform transient resource aliasing optimization
    fn alias_transient_resources(&mut self) -> RenderResult<()> {
        // Simple aliasing: resources that don't overlap in lifetime can share memory
        let resource_handles: Vec<_> = self.resources.keys().cloned().collect();
        let mut aliased_count = 0;
        let mut memory_saved = 0u64;

        for &resource_a in &resource_handles {
            for &resource_b in &resource_handles {
                if resource_a >= resource_b {
                    continue;
                } // Avoid double-checking

                if self.can_alias_resources(resource_a, resource_b)? {
                    // Alias resource_b with resource_a
                    if let Some(info_b) = self.resources.get_mut(&resource_b) {
                        info_b.aliased_with = Some(resource_a);
                        aliased_count += 1;

                        // Estimate memory saved
                        if let Some(size) = info_b.desc.size {
                            memory_saved += size;
                        } else if let Some(extent) = info_b.desc.extent {
                            // Rough estimate: 4 bytes per pixel for RGBA
                            memory_saved +=
                                (extent.width * extent.height * extent.depth_or_array_layers * 4)
                                    as u64;
                        }
                    }
                    break; // Only alias with one resource
                }
            }
        }

        self.metrics.aliased_count = aliased_count;
        self.metrics.memory_saved_bytes = memory_saved;

        Ok(())
    }

    /// Check if two resources can be aliased
    fn can_alias_resources(
        &self,
        resource_a: ResourceHandle,
        resource_b: ResourceHandle,
    ) -> RenderResult<bool> {
        let info_a = self
            .resources
            .get(&resource_a)
            .ok_or_else(|| RenderError::render("Invalid resource handle A"))?;
        let info_b = self
            .resources
            .get(&resource_b)
            .ok_or_else(|| RenderError::render("Invalid resource handle B"))?;

        // Can only alias transient resources of the same type
        if !info_a.is_transient || !info_b.is_transient {
            return Ok(false);
        }

        if !info_a.desc.can_alias || !info_b.desc.can_alias {
            return Ok(false);
        }

        if info_a.desc.resource_type != info_b.desc.resource_type {
            return Ok(false);
        }

        // Check lifetime overlap
        let (first_a, last_a) = (info_a.first_use, info_a.last_use);
        let (first_b, last_b) = (info_b.first_use, info_b.last_use);

        match (first_a, last_a, first_b, last_b) {
            (Some(fa), Some(la), Some(fb), Some(lb)) => {
                // No overlap if one ends before the other starts
                Ok(la.0 < fb.0 || lb.0 < fa.0)
            }
            _ => Ok(false), // Can't alias if lifetime is unclear
        }
    }

    /// Topologically sort passes for execution
    fn topological_sort(&self) -> RenderResult<Vec<PassHandle>> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut temp_visited = std::collections::HashSet::new();

        for &pass_handle in self.passes.keys() {
            if !visited.contains(&pass_handle) {
                self.dfs_visit(pass_handle, &mut visited, &mut temp_visited, &mut result)?;
            }
        }

        result.reverse(); // DFS gives reverse topological order
        Ok(result)
    }

    /// DFS helper for topological sort
    fn dfs_visit(
        &self,
        pass_handle: PassHandle,
        visited: &mut std::collections::HashSet<PassHandle>,
        temp_visited: &mut std::collections::HashSet<PassHandle>,
        result: &mut Vec<PassHandle>,
    ) -> RenderResult<()> {
        if temp_visited.contains(&pass_handle) {
            return Err(RenderError::render("Circular dependency in framegraph"));
        }

        if visited.contains(&pass_handle) {
            return Ok(());
        }

        temp_visited.insert(pass_handle);

        if let Some(pass_info) = self.passes.get(&pass_handle) {
            for &dep in &pass_info.dependencies {
                self.dfs_visit(dep, visited, temp_visited, result)?;
            }
        }

        temp_visited.remove(&pass_handle);
        visited.insert(pass_handle);
        result.push(pass_handle);

        Ok(())
    }

    /// Update execution metrics
    fn update_metrics(&mut self) {
        self.metrics.pass_count = self.passes.len();
        self.metrics.resource_count = self.resources.len();
        self.metrics.transient_count = self
            .resources
            .values()
            .filter(|info| info.is_transient)
            .count();
        // aliased_count and memory_saved_bytes are updated in alias_transient_resources
    }
}

impl Default for FrameGraph {
    fn default() -> Self {
        Self::new()
    }
}
