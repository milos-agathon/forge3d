use super::*;

impl PointRenderer {
    pub fn pack_points(&self, points: &[PointDef]) -> Result<Vec<PointInstance>, RenderError> {
        let mut instances = Vec::with_capacity(points.len());

        for point in points {
            if !point.position.x.is_finite() || !point.position.y.is_finite() {
                return Err(RenderError::Upload(format!(
                    "Point has non-finite coordinates: ({}, {})",
                    point.position.x, point.position.y
                )));
            }

            if point.style.point_size <= 0.0 || !point.style.point_size.is_finite() {
                return Err(RenderError::Upload(format!(
                    "Point size must be positive and finite, got {}",
                    point.style.point_size
                )));
            }

            instances.push(PointInstance {
                position: [point.position.x, point.position.y],
                size: point.style.point_size,
                color: point.style.fill_color,
                rotation: 0.0,
                uv_offset: [0.0, 0.0],
                _pad: 0.0,
            });
        }

        let validation_result = validate_point_instances(&instances);
        if !validation_result.is_valid {
            return Err(RenderError::Upload(
                validation_result
                    .error_message
                    .unwrap_or_else(|| "Point instance validation failed".to_string()),
            ));
        }

        Ok(instances)
    }

    pub fn upload_points(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[PointInstance],
    ) -> Result<(), RenderError> {
        if instances.is_empty() {
            return Ok(());
        }

        if instances.len() > self.instance_capacity {
            let new_capacity = (instances.len() * 2).max(1024);
            self.instance_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vf.Vector.Point.InstanceBuffer"),
                size: (new_capacity * std::mem::size_of::<PointInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.instance_capacity = new_capacity;
        }

        if let Some(instance_buffer) = &self.instance_buffer {
            queue.write_buffer(instance_buffer, 0, bytemuck::cast_slice(instances));
        }

        Ok(())
    }
}

pub fn cluster_points(points: &[Vec2], cluster_radius: f32) -> Vec<(Vec2, u32)> {
    if points.is_empty() {
        return Vec::new();
    }

    let mut clusters = Vec::new();
    let mut used = vec![false; points.len()];

    for (i, &point) in points.iter().enumerate() {
        if used[i] {
            continue;
        }

        let mut cluster_center = point;
        let mut cluster_count = 1;
        used[i] = true;

        for (j, &other_point) in points.iter().enumerate().skip(i + 1) {
            if used[j] {
                continue;
            }

            if (other_point - point).length() <= cluster_radius {
                cluster_center = (cluster_center * cluster_count as f32 + other_point)
                    / (cluster_count + 1) as f32;
                cluster_count += 1;
                used[j] = true;
            }
        }

        clusters.push((cluster_center, cluster_count));
    }

    clusters
}
