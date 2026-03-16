use super::*;

impl GpuBvhBuilder {
    pub fn build(&mut self, triangles: &[Triangle], _options: &BuildOptions) -> Result<BvhHandle> {
        let start_time = Instant::now();

        if triangles.is_empty() {
            anyhow::bail!("Cannot build BVH from empty triangle list");
        }

        if triangles.len() > (1 << 20) {
            anyhow::bail!(
                "Triangle count {} exceeds maximum of 1M triangles",
                triangles.len()
            );
        }

        let prim_count = triangles.len() as u32;
        let node_count = 2 * prim_count - 1; // Complete binary tree

        // Check memory budget (<= 512 MiB host-visible heap).
        let estimated_memory = self.estimate_memory_usage(prim_count)?;
        if estimated_memory > 512 * 1024 * 1024 {
            anyhow::bail!(
                "Estimated GPU memory usage {}MB exceeds 512MB budget",
                estimated_memory / (1024 * 1024)
            );
        }

        // Compute scene AABB and centroids
        let world_aabb = crate::accel::types::compute_scene_aabb(triangles);
        let centroids = crate::accel::types::compute_triangle_centroids(triangles);
        let triangle_aabbs = crate::accel::types::compute_triangle_aabbs(triangles);

        // Create GPU buffers
        let buffers = self.create_buffers(prim_count, &centroids, &triangle_aabbs)?;

        let mut stats = BuildStats::default();

        // Step 1: Morton codes on GPU
        let morton_start = Instant::now();
        self.generate_morton_codes(&buffers, &world_aabb, prim_count)?;
        stats.morton_time_ms = morton_start.elapsed().as_secs_f32() * 1000.0;

        // Step 2: CPU sort fallback (readback/sort/writeback)
        let sort_start = Instant::now();
        self.sort_morton_codes(&buffers, prim_count)?;
        stats.sort_time_ms = sort_start.elapsed().as_secs_f32() * 1000.0;

        // Step 3: Link nodes on GPU (init leaves + internal link)
        let link_start = Instant::now();
        self.build_bvh_topology(&buffers, prim_count, node_count)?;
        stats.link_time_ms = link_start.elapsed().as_secs_f32() * 1000.0;

        stats.build_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        stats.memory_usage_bytes = estimated_memory;
        stats.leaf_count = prim_count;
        stats.internal_count = prim_count - 1;

        let gpu_data = GpuBvhData {
            nodes_buffer: buffers.nodes_buffer,
            indices_buffer: buffers.sorted_indices_buffer,
            node_count,
            primitive_count: prim_count,
            world_aabb,
        };

        Ok(BvhHandle {
            backend: BvhBackend::Gpu(gpu_data),
            triangle_count: prim_count,
            node_count,
            world_aabb,
            build_stats: stats,
        })
    }
}
