use super::*;

impl SsrRenderer {
    pub fn collect_stats_into(
        &mut self,
        device: &Device,
        _queue: &Queue,
        stats: &mut SsrStats,
    ) -> RenderResult<()> {
        if !self.stats_readback_pending {
            // No stats were requested during execute; just keep timings.
            stats.trace_ms = self.last_trace_ms;
            stats.shade_ms = self.last_shade_ms;
            stats.fallback_ms = self.last_fallback_ms;
            return Ok(());
        }

        let slice = self.counters_readback.slice(..);
        let (sender, receiver) = oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);

        let map_result = block_on(receiver.receive()).ok_or_else(|| {
            RenderError::Readback("failed to receive SSR stats map signal".to_string())
        })?;
        map_result.map_err(|e| RenderError::Readback(format!("SSR stats map failed: {e:?}")))?;

        let data = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&data);
        if words.len() < 5 {
            return Err(RenderError::Readback(
                "SSR stats buffer size was smaller than expected".to_string(),
            ));
        }

        stats.num_rays = words[0];
        stats.num_hits = words[1];
        stats.total_steps = words[2];
        stats.num_misses = words[3];
        stats.miss_ibl_samples = words[4];
        stats.trace_ms = self.last_trace_ms;
        stats.shade_ms = self.last_shade_ms;
        stats.fallback_ms = self.last_fallback_ms;

        drop(data);
        self.counters_readback.unmap();

        self.stats_readback_pending = false;

        Ok(())
    }
}
