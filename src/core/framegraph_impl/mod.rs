//! Framegraph module for render pass organization and resource management
//!
//! This module provides a framegraph system for automatic resource lifetime management,
//! transient resource aliasing, and barrier insertion between render passes.

pub mod barriers;
pub mod types;

use super::error::{RenderError, RenderResult};
use std::collections::{BTreeMap, BTreeSet};

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
                pipeline_descriptor_bytes: Vec::new(),
                uniform_bytes: Vec::new(),
                cache_disabled_reason: None,
                pass_key: None,
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

    /// Declare the canonical complete pipeline/copy descriptor.
    pub fn pipeline_descriptor(&mut self, bytes: impl Into<Vec<u8>>) -> &mut Self {
        self.desc.pipeline_descriptor_bytes = bytes.into();
        self
    }

    /// Declare the exact uploaded bytes, including initialized padding.
    pub fn uniform_bytes(&mut self, bytes: impl Into<Vec<u8>>) -> &mut Self {
        self.desc.uniform_bytes = bytes.into();
        self
    }

    /// Disable cache reuse when any pixel-affecting declaration is incomplete.
    pub fn disable_cache(&mut self, reason: impl Into<String>) -> &mut Self {
        self.desc.cache_disabled_reason = Some(reason.into());
        self
    }

    /// Attach the already-computed ANAMNESIS key for this pass. The framegraph
    /// does not invent missing key inputs; callers leave the hook unset when
    /// their pass declaration is incomplete.
    pub fn pass_key(&mut self, key: crate::core::anamnesis::PassKey) -> &mut Self {
        self.desc.pass_key = Some(key);
        self
    }
}

/// Main framegraph for managing render passes and resources
#[derive(Debug)]
pub struct FrameGraph {
    /// All resources in the graph
    resources: BTreeMap<ResourceHandle, ResourceInfo>,
    /// All passes in the graph  
    passes: BTreeMap<PassHandle, PassInfo>,
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
            resources: BTreeMap::new(),
            passes: BTreeMap::new(),
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
            is_transient: desc.is_transient,
            desc,
            first_use: None,
            last_use: None,
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
        if builder.desc.pipeline_descriptor_bytes.is_empty() {
            return Err(RenderError::render(format!(
                "framegraph pass {name:?} omitted its pipeline descriptor"
            )));
        }
        if builder
            .desc
            .cache_disabled_reason
            .as_ref()
            .is_some_and(|reason| reason.trim().is_empty())
        {
            return Err(RenderError::render(format!(
                "framegraph pass {name:?} has an empty cache-disabled reason"
            )));
        }
        if self
            .passes
            .values()
            .any(|pass| pass.desc.name == builder.desc.name)
        {
            return Err(RenderError::render(format!(
                "duplicate framegraph pass label {name:?}"
            )));
        }

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
            handle,
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
        self.metrics.barrier_count = barriers.len();

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
        let pass_handles: Vec<_> = self.passes.keys().copied().collect();

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

        for pass_info in self.passes.values_mut() {
            pass_info.dependencies.sort_by_key(|handle| handle.0);
            pass_info.dependents.sort_by_key(|handle| handle.0);
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
        let resource_handles: Vec<_> = self.resources.keys().copied().collect();
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

    /// Topologically sort passes with a label-first tie break. A `BTreeMap`
    /// alone would only stabilize insertion handles; label ordering makes the
    /// plan invariant when independent declarations are presented shuffled.
    fn topological_sort(&self) -> RenderResult<Vec<PassHandle>> {
        let mut indegree: BTreeMap<PassHandle, usize> = self
            .passes
            .iter()
            .map(|(handle, info)| (*handle, info.dependencies.len()))
            .collect();
        let mut ready: BTreeSet<(String, usize)> = self
            .passes
            .iter()
            .filter(|(handle, _)| indegree.get(handle) == Some(&0))
            .map(|(handle, info)| (info.desc.name.clone(), handle.0))
            .collect();
        let mut result = Vec::with_capacity(self.passes.len());
        while let Some((label, raw_handle)) = ready.pop_first() {
            let handle = PassHandle(raw_handle);
            let _ = label;
            result.push(handle);
            let mut dependents = self.passes[&handle].dependents.clone();
            dependents.sort_by(|a, b| {
                self.passes[a]
                    .desc
                    .name
                    .cmp(&self.passes[b].desc.name)
                    .then_with(|| a.0.cmp(&b.0))
            });
            for dependent in dependents {
                let count = indegree.get_mut(&dependent).expect("dependent is declared");
                *count -= 1;
                if *count == 0 {
                    ready.insert((self.passes[&dependent].desc.name.clone(), dependent.0));
                }
            }
        }
        if result.len() != self.passes.len() {
            return Err(RenderError::render("Circular dependency in framegraph"));
        }
        Ok(result)
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

impl FrameGraph {
    /// Stable labels for an execution plan, used by ANAMNESIS dry-run and
    /// diagnostics without exposing insertion-dependent handles.
    pub fn execution_labels(&mut self) -> RenderResult<Vec<String>> {
        let (plan, _) = self.get_execution_plan()?;
        Ok(plan
            .into_iter()
            .filter_map(|handle| self.passes.get(&handle).map(|info| info.desc.name.clone()))
            .collect())
    }

    pub fn pass_info(&self, handle: PassHandle) -> Option<&PassInfo> {
        self.passes.get(&handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn independent_plan(order: &[&str]) -> Vec<String> {
        let mut graph = FrameGraph::new();
        for label in order {
            graph
                .add_pass(label, PassType::Compute, |pass| {
                    pass.pipeline_descriptor(format!("test:{label}"));
                    pass.uniform_bytes(Vec::new());
                    Ok(())
                })
                .unwrap();
        }
        graph.compile().unwrap();
        graph.execution_labels().unwrap()
    }

    #[test]
    fn compile_order_is_stable_across_shuffled_declarations() {
        let expected = vec![
            "alpha".to_string(),
            "middle".to_string(),
            "zeta".to_string(),
        ];
        let permutations = [
            ["zeta", "alpha", "middle"],
            ["middle", "zeta", "alpha"],
            ["alpha", "middle", "zeta"],
        ];
        for index in 0..100 {
            assert_eq!(
                independent_plan(&permutations[index % permutations.len()]),
                expected
            );
        }
    }

    #[test]
    fn dependent_compile_order_is_stable_across_one_hundred_shuffles() {
        let expected = vec![
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string(),
            "independent".to_string(),
            "omega".to_string(),
        ];
        let mut seed = 0x00A1_1A17_u64;
        for _ in 0..100 {
            let mut labels = ["alpha", "beta", "gamma", "omega", "independent"];
            for index in (1..labels.len()).rev() {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                labels.swap(index, (seed as usize) % (index + 1));
            }
            let mut graph = FrameGraph::new();
            let alpha_output = graph.add_resource(ResourceDesc {
                name: "alpha.output".into(),
                resource_type: ResourceType::StorageBuffer,
                format: None,
                extent: None,
                size: Some(4),
                usage: None,
                can_alias: false,
                is_transient: true,
            });
            let beta_output = graph.add_resource(ResourceDesc {
                name: "beta.output".into(),
                resource_type: ResourceType::StorageBuffer,
                format: None,
                extent: None,
                size: Some(4),
                usage: None,
                can_alias: false,
                is_transient: true,
            });
            let gamma_output = graph.add_resource(ResourceDesc {
                name: "gamma.output".into(),
                resource_type: ResourceType::StorageBuffer,
                format: None,
                extent: None,
                size: Some(4),
                usage: None,
                can_alias: false,
                is_transient: true,
            });
            for label in labels {
                graph
                    .add_pass(label, PassType::Compute, |pass| {
                        pass.pipeline_descriptor(format!("test:{label}"));
                        match label {
                            "alpha" => {
                                pass.write(alpha_output);
                            }
                            "beta" => {
                                pass.read(alpha_output).write(beta_output);
                            }
                            "gamma" => {
                                pass.read(alpha_output).write(gamma_output);
                            }
                            "omega" => {
                                pass.read(beta_output).read(gamma_output);
                            }
                            "independent" => {}
                            _ => unreachable!(),
                        }
                        Ok(())
                    })
                    .unwrap();
            }
            graph.compile().unwrap();
            assert_eq!(graph.execution_labels().unwrap(), expected);
        }
    }

    #[test]
    fn pass_declaration_requires_pipeline_material() {
        let mut graph = FrameGraph::new();
        let error = graph
            .add_pass("incomplete", PassType::Compute, |_| Ok(()))
            .unwrap_err();
        assert!(error.to_string().contains("pipeline descriptor"));
    }

    #[test]
    fn production_execution_receives_compiled_resource_transitions() {
        let mut builder = RendererGraphBuilder::new();
        let output = builder.add_resource(ResourceDesc {
            name: "production.color".into(),
            resource_type: ResourceType::ColorAttachment,
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            extent: Some(wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            }),
            size: None,
            usage: Some(
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            can_alias: false,
            is_transient: true,
        });
        builder
            .add_pass("render", PassType::Graphics, |pass| {
                pass.write(output)
                    .pipeline_descriptor(b"render-pipeline".to_vec())
                    .uniform_bytes(Vec::new());
                Ok(())
            })
            .unwrap();
        builder
            .add_pass("consume", PassType::Transfer, |pass| {
                pass.read(output)
                    .pipeline_descriptor(b"copy-pipeline".to_vec())
                    .uniform_bytes(Vec::new());
                Ok(())
            })
            .unwrap();
        let mut plan = builder.compile().unwrap();
        plan.execute_with_barriers("render", |barriers| {
            assert!(barriers.is_empty());
            Ok::<(), RenderError>(())
        })
        .unwrap();
        plan.execute_with_barriers("consume", |barriers| {
            assert!(
                !barriers.is_empty(),
                "consumer must receive the compiled attachment-to-read transition"
            );
            Ok::<(), RenderError>(())
        })
        .unwrap();
        plan.finish().unwrap();
    }
}

impl Default for FrameGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// The sole production construction path for an offline framegraph.
///
/// Diagnostics read the most recently compiled renderer report; they never
/// instantiate a synthetic graph.
pub struct RendererGraphBuilder {
    graph: FrameGraph,
}

impl RendererGraphBuilder {
    pub fn new() -> Self {
        Self {
            graph: FrameGraph::new(),
        }
    }

    pub fn add_resource(&mut self, desc: ResourceDesc) -> ResourceHandle {
        self.graph.add_resource(desc)
    }

    pub fn add_pass<F>(
        &mut self,
        label: &str,
        pass_type: PassType,
        setup: F,
    ) -> RenderResult<PassHandle>
    where
        F: FnOnce(&mut PassBuilder) -> RenderResult<()>,
    {
        self.graph.add_pass(label, pass_type, setup)
    }

    pub fn compile(mut self) -> RenderResult<RendererGraphPlan> {
        self.graph.compile()?;
        let (handles, barriers) = self.graph.get_execution_plan()?;
        let labels = handles
            .iter()
            .filter_map(|handle| {
                self.graph
                    .pass_info(*handle)
                    .map(|info| info.desc.name.clone())
            })
            .collect::<Vec<_>>();
        let passes = handles
            .iter()
            .filter_map(|handle| {
                self.graph
                    .pass_info(*handle)
                    .map(|info| (info.desc.name.clone(), info.desc.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        let pass_handles = handles
            .iter()
            .filter_map(|handle| {
                self.graph
                    .pass_info(*handle)
                    .map(|info| (info.desc.name.clone(), *handle))
            })
            .collect::<BTreeMap<_, _>>();
        let plan = RendererGraphPlan {
            labels,
            passes,
            pass_handles,
            resources: self.graph.resources.clone(),
            metrics: self.graph.metrics().clone(),
            barriers,
            next_pass: 0,
        };
        record_renderer_graph(&plan);
        Ok(plan)
    }
}

impl Default for RendererGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiled real-resource declaration plan shared by offline renderer paths.
#[derive(Clone, Debug)]
pub struct RendererGraphPlan {
    pub labels: Vec<String>,
    passes: BTreeMap<String, PassDesc>,
    pass_handles: BTreeMap<String, PassHandle>,
    resources: BTreeMap<ResourceHandle, ResourceInfo>,
    pub metrics: FrameGraphMetrics,
    pub barriers: Vec<ResourceBarrier>,
    next_pass: usize,
}

impl RendererGraphPlan {
    fn advance(&mut self, label: &str) -> RenderResult<()> {
        let expected = self.labels.get(self.next_pass).ok_or_else(|| {
            RenderError::render(format!(
                "renderer executed undeclared pass {label:?} after graph completion"
            ))
        })?;
        if expected != label {
            return Err(RenderError::render(format!(
                "renderer pass order mismatch: expected {expected:?}, got {label:?}"
            )));
        }
        self.next_pass += 1;
        Ok(())
    }

    /// Execute one declared production phase through the compiled plan.
    pub fn execute<T, E, F>(&mut self, label: &str, execute: F) -> Result<T, E>
    where
        E: From<RenderError>,
        F: FnOnce() -> Result<T, E>,
    {
        self.execute_with_barriers(label, |_| execute())
    }

    /// Execute one declared production phase with its compiled transition plan.
    ///
    /// wgpu performs the backend transition encoding, but the renderer still
    /// consumes this explicit plan so a framegraph regression cannot silently
    /// reduce production execution to label-order validation.
    pub fn execute_with_barriers<T, E, F>(&mut self, label: &str, execute: F) -> Result<T, E>
    where
        E: From<RenderError>,
        F: FnOnce(&[&ResourceBarrier]) -> Result<T, E>,
    {
        self.advance(label).map_err(E::from)?;
        let barriers = self.barriers_before(label);
        execute(&barriers)
    }

    pub fn pass(&self, label: &str) -> Option<&PassDesc> {
        self.passes.get(label)
    }

    pub fn resource(&self, handle: ResourceHandle) -> Option<&ResourceInfo> {
        self.resources.get(&handle)
    }

    pub fn ordered_passes(&self) -> impl Iterator<Item = &PassDesc> {
        self.labels
            .iter()
            .filter_map(|label| self.passes.get(label))
    }

    pub fn barriers_before(&self, label: &str) -> Vec<&ResourceBarrier> {
        let Some(handle) = self.pass_handles.get(label).copied() else {
            return Vec::new();
        };
        self.barriers
            .iter()
            .filter(|barrier| barrier.before_pass == handle)
            .collect()
    }

    pub fn finish(self) -> RenderResult<()> {
        if self.next_pass != self.labels.len() {
            return Err(RenderError::render(format!(
                "renderer stopped after {} of {} compiled passes",
                self.next_pass,
                self.labels.len()
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct RendererGraphReport {
    pub labels: Vec<String>,
    pub metrics: FrameGraphMetrics,
    pub barrier_count: usize,
    pub cache_disabled_passes: Vec<String>,
}

fn renderer_report_slot() -> &'static std::sync::Mutex<RendererGraphReport> {
    static SLOT: std::sync::OnceLock<std::sync::Mutex<RendererGraphReport>> =
        std::sync::OnceLock::new();
    SLOT.get_or_init(|| std::sync::Mutex::new(RendererGraphReport::default()))
}

fn record_renderer_graph(plan: &RendererGraphPlan) {
    let cache_disabled_passes = plan
        .passes
        .iter()
        .filter(|(_, pass)| pass.cache_disabled_reason.is_some())
        .map(|(label, _)| label.clone())
        .collect();
    if let Ok(mut slot) = renderer_report_slot().lock() {
        *slot = RendererGraphReport {
            labels: plan.labels.clone(),
            metrics: plan.metrics.clone(),
            barrier_count: plan.barriers.len(),
            cache_disabled_passes,
        };
    }
}

pub fn last_renderer_graph_report() -> RendererGraphReport {
    renderer_report_slot()
        .lock()
        .map(|report| report.clone())
        .unwrap_or_default()
}
