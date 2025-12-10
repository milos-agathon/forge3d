//! Async compute scheduler and executor.

use super::types::{
    AsyncComputeConfig, ComputeBarrier, ComputeMetrics, ComputePassDescriptor, ComputePassId,
    ComputePassInfo, ComputePassStatus, ResourceUsage,
};
use crate::core::error::{RenderError, RenderResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::{CommandEncoderDescriptor, Device, Queue};

/// Async compute scheduler and executor
pub struct AsyncComputeScheduler {
    device: Arc<Device>,
    queue: Arc<Queue>,
    config: AsyncComputeConfig,
    passes: HashMap<ComputePassId, ComputePassInfo>,
    next_pass_id: usize,
    resource_states: HashMap<String, ResourceUsage>,
    mutex: Mutex<()>,
}

impl AsyncComputeScheduler {
    /// Create a new async compute scheduler
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, config: AsyncComputeConfig) -> Self {
        Self {
            device,
            queue,
            config,
            passes: HashMap::new(),
            next_pass_id: 0,
            resource_states: HashMap::new(),
            mutex: Mutex::new(()),
        }
    }

    /// Submit a compute pass for async execution
    pub fn submit_compute_pass(
        &mut self,
        descriptor: ComputePassDescriptor,
    ) -> RenderResult<ComputePassId> {
        let _lock = self.mutex.lock().unwrap();

        let pass_id = ComputePassId(self.next_pass_id);
        self.next_pass_id += 1;

        let active_passes = self.count_active_passes();
        if active_passes >= self.config.max_concurrent_passes {
            return Err(RenderError::render("Too many concurrent compute passes"));
        }

        let pass_info = ComputePassInfo {
            descriptor,
            status: ComputePassStatus::Queued,
            start_time: None,
            end_time: None,
            command_buffer: None,
        };

        self.passes.insert(pass_id, pass_info);
        Ok(pass_id)
    }

    fn count_active_passes(&self) -> usize {
        self.passes
            .values()
            .filter(|info| {
                matches!(
                    info.status,
                    ComputePassStatus::Queued | ComputePassStatus::Executing
                )
            })
            .count()
    }

    /// Execute all queued compute passes
    pub fn execute_queued_passes(&mut self) -> RenderResult<Vec<ComputePassId>> {
        let queued_passes = self.collect_queued_passes_by_priority();
        let mut executed_passes = Vec::new();

        for (pass_id, _) in queued_passes {
            match self.execute_compute_pass_internal(pass_id) {
                Ok(()) => executed_passes.push(pass_id),
                Err(e) => {
                    let _lock = self.mutex.lock().unwrap();
                    if let Some(pass_info) = self.passes.get_mut(&pass_id) {
                        pass_info.status = ComputePassStatus::Failed(e.to_string());
                    }
                }
            }
        }

        Ok(executed_passes)
    }

    fn collect_queued_passes_by_priority(&self) -> Vec<(ComputePassId, u32)> {
        let _lock = self.mutex.lock().unwrap();
        let mut queued: Vec<_> = self
            .passes
            .iter()
            .filter(|(_, info)| info.status == ComputePassStatus::Queued)
            .map(|(&id, info)| (id, info.descriptor.priority))
            .collect();
        queued.sort_by(|a, b| b.1.cmp(&a.1));
        queued
    }

    /// Wait for specific compute passes to complete
    pub fn wait_for_passes(&mut self, pass_ids: &[ComputePassId]) -> RenderResult<()> {
        let timeout = std::time::Duration::from_millis(self.config.timeout_ms);
        let start_time = std::time::Instant::now();

        loop {
            if self.all_passes_completed(pass_ids) {
                break;
            }
            if start_time.elapsed() > timeout {
                return Err(RenderError::render(
                    "Timeout waiting for compute passes to complete",
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        Ok(())
    }

    fn all_passes_completed(&self, pass_ids: &[ComputePassId]) -> bool {
        let _lock = self.mutex.lock().unwrap();
        pass_ids.iter().all(|&pass_id| {
            self.passes
                .get(&pass_id)
                .map(|info| {
                    matches!(
                        info.status,
                        ComputePassStatus::Completed | ComputePassStatus::Failed(_)
                    )
                })
                .unwrap_or(true)
        })
    }

    /// Insert barriers for compute/graphics synchronization
    pub fn insert_barriers(&mut self, barriers: Vec<ComputeBarrier>) -> RenderResult<()> {
        if barriers.is_empty() {
            return Ok(());
        }

        let encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("compute_barriers"),
            });

        for barrier in barriers {
            let resource_name = self.get_barrier_resource_name(&barrier);
            if let Some(name) = resource_name {
                self.resource_states.insert(name, barrier.dst_usage);
            }
        }

        let command_buffer = encoder.finish();
        self.queue.submit([command_buffer]);
        Ok(())
    }

    fn get_barrier_resource_name(&self, barrier: &ComputeBarrier) -> Option<String> {
        if let Some(ref buffer) = barrier.buffer {
            Some(format!("buffer_{:p}", buffer.as_ref() as *const _))
        } else if let Some(ref texture) = barrier.texture {
            Some(format!("texture_{:p}", texture.as_ref() as *const _))
        } else {
            None
        }
    }

    /// Get status of a compute pass
    pub fn get_pass_status(&self, pass_id: ComputePassId) -> Option<ComputePassStatus> {
        let _lock = self.mutex.lock().unwrap();
        self.passes.get(&pass_id).map(|info| info.status.clone())
    }

    /// Get performance metrics for completed passes
    pub fn get_metrics(&self) -> ComputeMetrics {
        let _lock = self.mutex.lock().unwrap();
        self.compute_metrics_internal()
    }

    fn compute_metrics_internal(&self) -> ComputeMetrics {
        let mut completed_passes = 0;
        let mut failed_passes = 0;
        let mut total_execution_time_ms = 0.0;
        let mut total_workgroups = 0;

        for info in self.passes.values() {
            match &info.status {
                ComputePassStatus::Completed => {
                    completed_passes += 1;
                    total_workgroups += info.descriptor.dispatch.total_workgroups();
                    if let (Some(start), Some(end)) = (info.start_time, info.end_time) {
                        total_execution_time_ms += end.duration_since(start).as_millis() as f32;
                    }
                }
                ComputePassStatus::Failed(_) => failed_passes += 1,
                _ => {}
            }
        }

        ComputeMetrics {
            total_passes: self.passes.len(),
            completed_passes,
            failed_passes,
            total_execution_time_ms,
            total_workgroups,
            average_execution_time_ms: if completed_passes > 0 {
                total_execution_time_ms / completed_passes as f32
            } else {
                0.0
            },
        }
    }

    /// Cancel a queued compute pass
    pub fn cancel_pass(&mut self, pass_id: ComputePassId) -> RenderResult<()> {
        let _lock = self.mutex.lock().unwrap();

        if let Some(pass_info) = self.passes.get_mut(&pass_id) {
            match pass_info.status {
                ComputePassStatus::Queued => {
                    pass_info.status = ComputePassStatus::Cancelled;
                    Ok(())
                }
                _ => Err(RenderError::render(
                    "Cannot cancel compute pass that is not queued",
                )),
            }
        } else {
            Err(RenderError::render("Compute pass not found"))
        }
    }

    /// Clear completed and failed passes
    pub fn cleanup_completed_passes(&mut self) {
        let _lock = self.mutex.lock().unwrap();
        self.passes.retain(|_, info| {
            !matches!(
                info.status,
                ComputePassStatus::Completed
                    | ComputePassStatus::Failed(_)
                    | ComputePassStatus::Cancelled
            )
        });
    }

    fn execute_compute_pass_internal(&mut self, pass_id: ComputePassId) -> RenderResult<()> {
        let (label, label_prefix) = {
            let _lock = self.mutex.lock().unwrap();
            let pass_info = self
                .passes
                .get_mut(&pass_id)
                .ok_or_else(|| RenderError::render("Compute pass not found"))?;

            pass_info.status = ComputePassStatus::Executing;
            pass_info.start_time = Some(std::time::Instant::now());

            (
                pass_info.descriptor.label.clone(),
                self.config.label_prefix.clone(),
            )
        };

        let encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some(&format!("{}_{}", label_prefix, label)),
            });

        let command_buffer = encoder.finish();

        {
            let _lock = self.mutex.lock().unwrap();
            if let Some(pass_info) = self.passes.get_mut(&pass_id) {
                pass_info.command_buffer = Some(command_buffer);
                pass_info.status = ComputePassStatus::Completed;
                pass_info.end_time = Some(std::time::Instant::now());
            }
        }

        Ok(())
    }
}
