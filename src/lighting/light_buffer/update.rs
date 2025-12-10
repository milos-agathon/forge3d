use super::types::{LightBuffer, MAX_LIGHTS};
use crate::lighting::types::Light;
use wgpu::{Device, Queue};

impl LightBuffer {
    /// Update lights for the current frame
    ///
    /// # Arguments
    /// * `queue` - GPU command queue for buffer upload
    /// * `lights` - Slice of lights to upload (max MAX_LIGHTS)
    ///
    /// # Returns
    /// * Result indicating success or error if too many lights
    pub fn update(
        &mut self,
        device: &Device,
        queue: &Queue,
        lights: &[Light],
    ) -> Result<(), String> {
        if lights.len() > MAX_LIGHTS {
            return Err(format!(
                "Too many lights: {} (max {})",
                lights.len(),
                MAX_LIGHTS
            ));
        }

        // Get current buffer
        let buffer = &self.buffers[self.frame_index];
        let count_buffer = &self.count_buffers[self.frame_index];

        // Upload light data
        if !lights.is_empty() {
            let light_bytes = bytemuck::cast_slice(lights);
            queue.write_buffer(buffer, 0, light_bytes);
        }

        // Upload metadata: light count, frame counter (lower 32 bits), and R2 seed encoded as bits
        let seed = self.sequence_seed;
        let count_data = [
            lights.len() as u32,
            (self.frame_counter & 0xFFFF_FFFF) as u32,
            seed[0].to_bits(),
            seed[1].to_bits(),
        ];
        queue.write_buffer(count_buffer, 0, bytemuck::cast_slice(&count_data));
        queue.write_buffer(&self.environment_stub, 0, &[0u8; 16]);

        self.light_count = lights.len() as u32;

        // P1-07: Store copy for debug inspection
        self.last_uploaded_lights = lights.to_vec();

        // Recreate bind group for current frame
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Light Bind Group Frame {}", self.frame_index)),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.environment_stub.as_entire_binding(),
                },
            ],
        }));

        Ok(())
    }
}
