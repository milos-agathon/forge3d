use super::*;

/// GPU buffer collection for BVH construction
pub(super) struct GpuBuffers {
    pub(super) centroids_buffer: Buffer,
    pub(super) prim_indices_buffer: Buffer,
    pub(super) morton_codes_buffer: Buffer,
    pub(super) sorted_indices_buffer: Buffer,
    pub(super) primitive_aabbs_buffer: Buffer,
    pub(super) nodes_buffer: Buffer,
    pub(super) sort_temp_keys: Buffer,
    pub(super) sort_temp_values: Buffer,
    pub(super) sort_histogram: Buffer,
    pub(super) sort_prefix_sums: Buffer,
    pub(super) _node_flags_buffer: Buffer,
}

impl GpuBvhBuilder {
    pub(super) fn estimate_memory_usage(&self, prim_count: u32) -> Result<u64> {
        let node_count = 2 * prim_count - 1;

        // Buffer sizes in bytes
        let centroids_size = prim_count * 12; // vec3<f32>
        let morton_codes_size = prim_count * 4; // u32
        let indices_size = prim_count * 4; // u32
        let nodes_size = node_count * std::mem::size_of::<BvhNode>() as u32;
        let aabbs_size = prim_count * std::mem::size_of::<Aabb>() as u32;
        let sort_temp_size = prim_count * 8; // Temporary sorting buffers
        let histogram_size = 1024; // Sort histogram

        let total = centroids_size
            + morton_codes_size
            + indices_size
            + nodes_size
            + aabbs_size
            + sort_temp_size
            + histogram_size;

        Ok(total as u64)
    }

    pub(super) fn create_buffers(
        &self,
        prim_count: u32,
        centroids: &[[f32; 3]],
        aabbs: &[Aabb],
    ) -> Result<GpuBuffers> {
        use wgpu::util::{BufferInitDescriptor, DeviceExt};

        let node_count = 2 * prim_count - 1;
        let indices: Vec<u32> = (0..prim_count).collect();

        let buffers = GpuBuffers {
            centroids_buffer: self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Centroids"),
                contents: cast_slice(centroids),
                usage: BufferUsages::STORAGE,
            }),

            prim_indices_buffer: self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Prim Indices"),
                contents: cast_slice(&indices),
                usage: BufferUsages::STORAGE,
            }),

            morton_codes_buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Morton Codes"),
                size: (prim_count * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),

            sorted_indices_buffer: self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Sorted Indices"),
                contents: cast_slice(&indices),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            }),

            primitive_aabbs_buffer: self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Primitive AABBs"),
                contents: cast_slice(aabbs),
                usage: BufferUsages::STORAGE,
            }),

            nodes_buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("BVH Nodes"),
                size: (node_count * std::mem::size_of::<BvhNode>() as u32) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),

            sort_temp_keys: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Sort Temp Keys"),
                size: (prim_count * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),

            sort_temp_values: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Sort Temp Values"),
                size: (prim_count * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),

            sort_histogram: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Sort Histogram"),
                size: 1024,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),

            sort_prefix_sums: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Sort Prefix Sums"),
                size: 1024,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),

            _node_flags_buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Node Flags"),
                size: (node_count * 4) as u64,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
        };

        Ok(buffers)
    }
}
