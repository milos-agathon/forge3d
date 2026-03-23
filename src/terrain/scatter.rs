#![cfg(feature = "enable-gpu-instancing")]

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use glam::{Mat4, Vec3};

use crate::core::resource_tracker::{register_buffer, ResourceHandle};
use crate::geometry::simplify_mesh;
use crate::geometry::MeshBuffers;
use crate::render::mesh_instanced::VertexPN;

#[derive(Debug, Clone)]
pub struct TerrainScatterLevelSpec {
    pub mesh: MeshBuffers,
    pub max_distance: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct HlodConfig {
    pub hlod_distance: f32,
    pub cluster_radius: f32,
    pub simplify_ratio: f32,
}

#[derive(Debug, Clone, Default)]
pub struct TerrainScatterBatchStats {
    pub total_instances: u32,
    pub visible_instances: u32,
    pub culled_instances: u32,
    pub lod_instance_counts: Vec<u32>,
    pub hlod_cluster_draws: u32,
    pub hlod_covered_instances: u32,
    pub effective_draws: u32,
}

#[derive(Debug, Clone, Default)]
pub struct TerrainScatterFrameStats {
    pub batch_count: u32,
    pub rendered_batches: u32,
    pub total_instances: u32,
    pub visible_instances: u32,
    pub culled_instances: u32,
    pub lod_instance_counts: Vec<u32>,
    pub hlod_cluster_draws: u32,
    pub hlod_covered_instances: u32,
    pub effective_draws: u32,
}

#[derive(Debug, Clone, Default)]
pub struct TerrainScatterMemoryReport {
    pub batch_count: u32,
    pub level_count: u32,
    pub total_instances: u32,
    pub vertex_buffer_bytes: u64,
    pub index_buffer_bytes: u64,
    pub instance_buffer_bytes: u64,
    pub hlod_cluster_count: u32,
    pub hlod_buffer_bytes: u64,
}

impl TerrainScatterMemoryReport {
    pub fn total_buffer_bytes(&self) -> u64 {
        self.vertex_buffer_bytes
            + self.index_buffer_bytes
            + self.instance_buffer_bytes
            + self.hlod_buffer_bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedScatterDraw {
    pub level_index: usize,
    pub instance_count: u32,
}

struct GpuScatterLevel {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    index_count: u32,
    max_distance: f32,
    vertex_buffer_bytes: u64,
    index_buffer_bytes: u64,
    _vertex_handle: ResourceHandle,
    _index_handle: ResourceHandle,
}

struct ScatterInstanceBuffer {
    buffer: wgpu::Buffer,
    capacity: usize,
    bytes: u64,
    _handle: ResourceHandle,
}

struct GpuHlodCluster {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    index_count: u32,
    center: Vec3,
    radius: f32,
    vertex_buffer_bytes: u64,
    index_buffer_bytes: u64,
    _vertex_handle: ResourceHandle,
    _index_handle: ResourceHandle,
}

struct HlodCache {
    clusters: Vec<GpuHlodCluster>,
    instance_to_cluster: Vec<Option<usize>>,
    hlod_distance: f32,
    total_buffer_bytes: u64,
}

pub struct TerrainScatterBatch {
    pub name: Option<String>,
    pub color: [f32; 4],
    pub max_draw_distance: f32,
    levels: Vec<GpuScatterLevel>,
    transforms_rowmajor: Vec<[f32; 16]>,
    positions: Vec<[f32; 3]>,
    instance_buffers: Vec<Option<ScatterInstanceBuffer>>,
    last_stats: TerrainScatterBatchStats,
    hlod_cache: Option<HlodCache>,
    hlod_config: Option<HlodConfig>,
    hlod_source_mesh: Option<MeshBuffers>,
}

impl TerrainScatterBatch {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        levels: Vec<TerrainScatterLevelSpec>,
        transforms_rowmajor: &[[f32; 16]],
        color: [f32; 4],
        max_draw_distance: Option<f32>,
        name: Option<String>,
        hlod_config: Option<HlodConfig>,
    ) -> Result<Self> {
        if levels.is_empty() {
            return Err(anyhow!("terrain scatter requires at least one LOD level"));
        }
        if transforms_rowmajor.is_empty() {
            return Err(anyhow!("terrain scatter requires at least one transform"));
        }
        validate_level_specs(&levels)?;
        validate_transforms(transforms_rowmajor)?;
        let max_draw_distance =
            validate_optional_distance(max_draw_distance, "terrain scatter max_draw_distance")?
                .unwrap_or(f32::INFINITY);

        // Extract coarsest LOD mesh BEFORE consuming levels (for HLOD rebuild)
        let hlod_source_mesh = if hlod_config.is_some() {
            Some(levels.last().unwrap().mesh.clone())
        } else {
            None
        };

        let gpu_levels = levels
            .into_iter()
            .map(|spec| build_gpu_level(device, queue, spec))
            .collect::<Result<Vec<_>>>()?;
        let level_count = gpu_levels.len();

        let positions = extract_positions(transforms_rowmajor);

        let hlod_cache = if let Some(ref config) = hlod_config {
            if let Some(ref source_mesh) = hlod_source_mesh {
                Some(build_hlod_cache(
                    device,
                    queue,
                    source_mesh,
                    transforms_rowmajor,
                    &positions,
                    config,
                )?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            name,
            color,
            max_draw_distance,
            levels: gpu_levels,
            transforms_rowmajor: transforms_rowmajor.to_vec(),
            positions,
            instance_buffers: std::iter::repeat_with(|| None).take(level_count).collect(),
            last_stats: TerrainScatterBatchStats::default(),
            hlod_cache,
            hlod_config,
            hlod_source_mesh,
        })
    }

    pub fn update_transforms(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        transforms_rowmajor: &[[f32; 16]],
    ) -> Result<()> {
        if transforms_rowmajor.is_empty() {
            return Err(anyhow!("terrain scatter requires at least one transform"));
        }
        validate_transforms(transforms_rowmajor)?;

        self.transforms_rowmajor.clear();
        self.transforms_rowmajor
            .extend_from_slice(transforms_rowmajor);
        self.positions = extract_positions(transforms_rowmajor);
        self.last_stats = TerrainScatterBatchStats::default();

        // Rebuild HLOD cache when config+source mesh are present
        if let (Some(ref config), Some(ref source_mesh)) =
            (&self.hlod_config, &self.hlod_source_mesh)
        {
            self.hlod_cache = Some(build_hlod_cache(
                device,
                queue,
                source_mesh,
                transforms_rowmajor,
                &self.positions,
                config,
            )?);
        }

        Ok(())
    }

    pub fn last_stats(&self) -> &TerrainScatterBatchStats {
        &self.last_stats
    }

    pub fn level_vbuf(&self, level_index: usize) -> &wgpu::Buffer {
        &self.levels[level_index].vbuf
    }

    pub fn level_ibuf(&self, level_index: usize) -> &wgpu::Buffer {
        &self.levels[level_index].ibuf
    }

    pub fn level_instbuf(&self, level_index: usize) -> Option<&wgpu::Buffer> {
        self.instance_buffers[level_index]
            .as_ref()
            .map(|buffer| &buffer.buffer)
    }

    pub fn level_index_count(&self, level_index: usize) -> u32 {
        self.levels[level_index].index_count
    }

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    pub fn hlod_active_clusters(&self, eye_contract: Vec3) -> Vec<usize> {
        let cache = match &self.hlod_cache {
            Some(c) => c,
            None => return Vec::new(),
        };
        cache
            .clusters
            .iter()
            .enumerate()
            .filter(|(_, cluster)| {
                let dist = eye_contract.distance(cluster.center) - cluster.radius;
                dist > cache.hlod_distance && dist < self.max_draw_distance
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn hlod_cluster_vbuf(&self, idx: usize) -> Option<&wgpu::Buffer> {
        self.hlod_cache
            .as_ref()
            .and_then(|c| c.clusters.get(idx))
            .map(|cluster| &cluster.vbuf)
    }

    pub fn hlod_cluster_ibuf(&self, idx: usize) -> Option<&wgpu::Buffer> {
        self.hlod_cache
            .as_ref()
            .and_then(|c| c.clusters.get(idx))
            .map(|cluster| &cluster.ibuf)
    }

    pub fn hlod_cluster_index_count(&self, idx: usize) -> u32 {
        self.hlod_cache
            .as_ref()
            .and_then(|c| c.clusters.get(idx))
            .map(|cluster| cluster.index_count)
            .unwrap_or(0)
    }

    pub fn prepare_draws(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        eye_contract: Vec3,
        render_from_contract: Mat4,
        instance_scale: f32,
    ) -> Result<(TerrainScatterBatchStats, Vec<PreparedScatterDraw>)> {
        let mut per_level = vec![Vec::<[f32; 16]>::new(); self.levels.len()];
        let mut stats = TerrainScatterBatchStats {
            total_instances: self.transforms_rowmajor.len() as u32,
            lod_instance_counts: vec![0; self.levels.len()],
            ..Default::default()
        };

        // Determine which HLOD clusters are active
        let cluster_active: Vec<bool> = if let Some(ref cache) = self.hlod_cache {
            cache
                .clusters
                .iter()
                .map(|cluster| {
                    let dist = eye_contract.distance(cluster.center) - cluster.radius;
                    dist > cache.hlod_distance && dist < self.max_draw_distance
                })
                .collect()
        } else {
            Vec::new()
        };

        for (i, (transform, position)) in self
            .transforms_rowmajor
            .iter()
            .zip(self.positions.iter())
            .enumerate()
        {
            let dist = eye_contract.distance(Vec3::new(position[0], position[1], position[2]));
            if dist > self.max_draw_distance {
                continue;
            }

            // Check if this instance is covered by an active HLOD cluster
            if let Some(ref cache) = self.hlod_cache {
                if let Some(Some(cluster_idx)) = cache.instance_to_cluster.get(i) {
                    if *cluster_idx < cluster_active.len() && cluster_active[*cluster_idx] {
                        stats.hlod_covered_instances += 1;
                        stats.visible_instances += 1;
                        continue;
                    }
                }
            }

            let level_index = select_level_index(&self.levels, dist);
            per_level[level_index].push(*transform);
            stats.visible_instances += 1;
            stats.lod_instance_counts[level_index] += 1;
        }

        stats.culled_instances = stats
            .total_instances
            .saturating_sub(stats.visible_instances);

        // Count active HLOD clusters
        stats.hlod_cluster_draws = cluster_active.iter().filter(|&&a| a).count() as u32;

        let mut draws = Vec::new();
        for (level_index, transforms) in per_level.iter().enumerate() {
            if transforms.is_empty() {
                continue;
            }

            ensure_instance_capacity(
                device,
                &mut self.instance_buffers[level_index],
                transforms.len(),
            )?;
            let packed = pack_instance_transforms(transforms, render_from_contract, instance_scale);
            if let Some(instance_buffer) = self.instance_buffers[level_index].as_ref() {
                queue.write_buffer(&instance_buffer.buffer, 0, bytemuck::cast_slice(&packed));
            }

            draws.push(PreparedScatterDraw {
                level_index,
                instance_count: transforms.len() as u32,
            });
        }

        stats.effective_draws = draws.len() as u32 + stats.hlod_cluster_draws;

        self.last_stats = stats.clone();
        Ok((stats, draws))
    }

    pub fn memory_report(&self) -> TerrainScatterMemoryReport {
        let mut report = TerrainScatterMemoryReport {
            batch_count: 1,
            level_count: self.levels.len() as u32,
            total_instances: self.transforms_rowmajor.len() as u32,
            ..Default::default()
        };

        for level in &self.levels {
            report.vertex_buffer_bytes += level.vertex_buffer_bytes;
            report.index_buffer_bytes += level.index_buffer_bytes;
        }

        for buffer in self.instance_buffers.iter().flatten() {
            report.instance_buffer_bytes += buffer.bytes;
        }

        if let Some(ref cache) = self.hlod_cache {
            report.hlod_cluster_count = cache.clusters.len() as u32;
            report.hlod_buffer_bytes = cache.total_buffer_bytes;
        }

        report
    }
}

pub fn summarize_memory(batches: &[TerrainScatterBatch]) -> TerrainScatterMemoryReport {
    let mut report = TerrainScatterMemoryReport::default();
    for batch in batches {
        let batch_report = batch.memory_report();
        report.batch_count += batch_report.batch_count;
        report.level_count += batch_report.level_count;
        report.total_instances += batch_report.total_instances;
        report.vertex_buffer_bytes += batch_report.vertex_buffer_bytes;
        report.index_buffer_bytes += batch_report.index_buffer_bytes;
        report.instance_buffer_bytes += batch_report.instance_buffer_bytes;
        report.hlod_cluster_count += batch_report.hlod_cluster_count;
        report.hlod_buffer_bytes += batch_report.hlod_buffer_bytes;
    }
    report
}

pub fn accumulate_frame_stats(
    stats: &mut TerrainScatterFrameStats,
    batch_stats: &TerrainScatterBatchStats,
) {
    stats.batch_count += 1;
    if batch_stats.visible_instances > 0 {
        stats.rendered_batches += 1;
    }
    stats.total_instances += batch_stats.total_instances;
    stats.visible_instances += batch_stats.visible_instances;
    stats.culled_instances += batch_stats.culled_instances;
    stats.hlod_cluster_draws += batch_stats.hlod_cluster_draws;
    stats.hlod_covered_instances += batch_stats.hlod_covered_instances;
    stats.effective_draws += batch_stats.effective_draws;

    if stats.lod_instance_counts.len() < batch_stats.lod_instance_counts.len() {
        stats
            .lod_instance_counts
            .resize(batch_stats.lod_instance_counts.len(), 0);
    }

    for (dst, src) in stats
        .lod_instance_counts
        .iter_mut()
        .zip(batch_stats.lod_instance_counts.iter())
    {
        *dst += *src;
    }
}

// ---------------------------------------------------------------------------
// HLOD build logic
// ---------------------------------------------------------------------------

fn grid_cell_key(pos: &[f32; 3], cell_size: f32) -> (i32, i32, i32) {
    (
        (pos[0] / cell_size).floor() as i32,
        (pos[1] / cell_size).floor() as i32,
        (pos[2] / cell_size).floor() as i32,
    )
}

/// Compute the maximum distance from the origin to any vertex in the mesh.
fn mesh_bounding_radius(mesh: &MeshBuffers) -> f32 {
    mesh.positions
        .iter()
        .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
        .fold(0.0f32, f32::max)
}

/// Extract the maximum axis scale from a row-major 4x4 transform.
fn max_instance_scale(row_major: &[f32; 16]) -> f32 {
    // Row-major layout: col0=(r[0],r[4],r[8]), col1=(r[1],r[5],r[9]), col2=(r[2],r[6],r[10])
    let sx = (row_major[0] * row_major[0] + row_major[4] * row_major[4] + row_major[8] * row_major[8]).sqrt();
    let sy = (row_major[1] * row_major[1] + row_major[5] * row_major[5] + row_major[9] * row_major[9]).sqrt();
    let sz = (row_major[2] * row_major[2] + row_major[6] * row_major[6] + row_major[10] * row_major[10]).sqrt();
    sx.max(sy).max(sz)
}

fn build_hlod_cache(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source_mesh: &MeshBuffers,
    transforms_rowmajor: &[[f32; 16]],
    positions: &[[f32; 3]],
    config: &HlodConfig,
) -> Result<HlodCache> {
    let base_mesh_radius = mesh_bounding_radius(source_mesh);

    // Step 1: Hash instances into grid cells
    let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, pos) in positions.iter().enumerate() {
        let key = grid_cell_key(pos, config.cluster_radius);
        cells.entry(key).or_default().push(i);
    }

    let mut instance_to_cluster: Vec<Option<usize>> = vec![None; transforms_rowmajor.len()];
    let mut clusters = Vec::new();
    let mut total_buffer_bytes = 0u64;

    for (_cell_key, instance_indices) in &cells {
        if instance_indices.len() < 2 {
            // Cells with 1 instance: leave as individual draw
            continue;
        }

        let cluster_idx = clusters.len();

        // Merge geometry: bake transforms into vertices of coarsest LOD mesh
        let mut merged = MeshBuffers::with_capacity(
            source_mesh.positions.len() * instance_indices.len(),
            source_mesh.indices.len() * instance_indices.len(),
        );

        let mut center_sum = Vec3::ZERO;
        for &inst_idx in instance_indices {
            let m = row_major_to_mat4(transforms_rowmajor[inst_idx]);
            let vertex_offset = merged.positions.len() as u32;

            // Bake transform into positions
            for pos in &source_mesh.positions {
                let p = Vec3::new(pos[0], pos[1], pos[2]);
                let tp = m.transform_point3(p);
                merged.positions.push([tp.x, tp.y, tp.z]);
            }

            // Bake transform into normals (use inverse transpose for correct normal transform)
            let normal_mat = m.inverse().transpose();
            if !source_mesh.normals.is_empty() {
                for normal in &source_mesh.normals {
                    let n = Vec3::new(normal[0], normal[1], normal[2]);
                    let tn = normal_mat.transform_vector3(n).normalize_or_zero();
                    merged.normals.push([tn.x, tn.y, tn.z]);
                }
            }

            // Offset indices
            for idx in &source_mesh.indices {
                merged.indices.push(idx + vertex_offset);
            }

            center_sum += Vec3::new(
                positions[inst_idx][0],
                positions[inst_idx][1],
                positions[inst_idx][2],
            );
            instance_to_cluster[inst_idx] = Some(cluster_idx);
        }

        let center = center_sum / instance_indices.len() as f32;

        // Compute radius: max distance from center to any instance's furthest
        // geometry extent, accounting for mesh bounding radius and per-instance scale.
        let radius = instance_indices
            .iter()
            .map(|&i| {
                let pos_dist = Vec3::new(positions[i][0], positions[i][1], positions[i][2]).distance(center);
                let inst_scale = max_instance_scale(&transforms_rowmajor[i]);
                pos_dist + base_mesh_radius * inst_scale
            })
            .fold(0.0f32, f32::max);

        // Simplify the merged mesh
        let simplified = simplify_mesh(&merged, config.simplify_ratio).map_err(|e| {
            anyhow!(
                "HLOD cluster simplification failed: {}",
                e.message()
            )
        })?;

        // Upload to GPU
        let vertices: Vec<VertexPN> = simplified
            .positions
            .iter()
            .enumerate()
            .map(|(index, position)| VertexPN {
                position: *position,
                normal: simplified
                    .normals
                    .get(index)
                    .copied()
                    .unwrap_or([0.0, 1.0, 0.0]),
            })
            .collect();

        let vertex_buffer_bytes = (vertices.len() * std::mem::size_of::<VertexPN>()) as u64;
        let index_buffer_bytes =
            (simplified.indices.len() * std::mem::size_of::<u32>()) as u64;

        if vertex_buffer_bytes == 0 || index_buffer_bytes == 0 {
            // Degenerate cluster after simplification — skip
            for &inst_idx in instance_indices {
                instance_to_cluster[inst_idx] = None;
            }
            continue;
        }

        let vertex_handle = register_buffer(
            vertex_buffer_bytes,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );
        let index_handle = register_buffer(
            index_buffer_bytes,
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        );

        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain.scatter.hlod.vertex_buffer"),
            size: vertex_buffer_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain.scatter.hlod.index_buffer"),
            size: index_buffer_bytes,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(&ibuf, 0, bytemuck::cast_slice(&simplified.indices));

        total_buffer_bytes += vertex_buffer_bytes + index_buffer_bytes;

        clusters.push(GpuHlodCluster {
            vbuf,
            ibuf,
            index_count: simplified.indices.len() as u32,
            center,
            radius,
            vertex_buffer_bytes,
            index_buffer_bytes,
            _vertex_handle: vertex_handle,
            _index_handle: index_handle,
        });
    }

    Ok(HlodCache {
        clusters,
        instance_to_cluster,
        hlod_distance: config.hlod_distance,
        total_buffer_bytes,
    })
}

/// Pack a single identity instance transform for HLOD cluster drawing.
/// The HLOD geometry is already in contract space, so we just apply
/// render_from_contract as the instance transform.
pub fn pack_hlod_identity_instance(render_from_contract: Mat4) -> [f32; 16] {
    render_from_contract.to_cols_array()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_optional_distance(value: Option<f32>, label: &str) -> Result<Option<f32>> {
    match value {
        Some(distance) if !distance.is_finite() || distance <= 0.0 => Err(anyhow!(
            "{label} must be a positive finite float when provided"
        )),
        Some(distance) => Ok(Some(distance)),
        None => Ok(None),
    }
}

fn validate_level_specs(levels: &[TerrainScatterLevelSpec]) -> Result<()> {
    let mut previous_max_distance = 0.0_f32;
    for (index, level) in levels.iter().enumerate() {
        match validate_optional_distance(
            level.max_distance,
            &format!("terrain scatter level {index} max_distance"),
        )? {
            Some(max_distance) => {
                if max_distance <= previous_max_distance {
                    return Err(anyhow!(
                        "terrain scatter LOD max_distance values must be strictly increasing"
                    ));
                }
                previous_max_distance = max_distance;
            }
            None if index != levels.len().saturating_sub(1) => {
                return Err(anyhow!(
                    "only the final terrain scatter LOD level may omit max_distance"
                ));
            }
            None => {}
        }
    }
    Ok(())
}

fn validate_transforms(transforms_rowmajor: &[[f32; 16]]) -> Result<()> {
    if transforms_rowmajor
        .iter()
        .flat_map(|transform| transform.iter())
        .any(|value| !value.is_finite())
    {
        return Err(anyhow!(
            "terrain scatter transforms must contain only finite values"
        ));
    }
    Ok(())
}

fn extract_positions(transforms_rowmajor: &[[f32; 16]]) -> Vec<[f32; 3]> {
    transforms_rowmajor
        .iter()
        .map(|row| [row[3], row[7], row[11]])
        .collect()
}

fn pack_instance_transforms(
    transforms_rowmajor: &[[f32; 16]],
    render_from_contract: Mat4,
    instance_scale: f32,
) -> Vec<f32> {
    let mut packed = Vec::with_capacity(transforms_rowmajor.len() * 16);
    let uniform = Mat4::from_scale(Vec3::splat(instance_scale));
    for row_major in transforms_rowmajor {
        let m = row_major_to_mat4(*row_major);
        // Map position through the (possibly non-uniform) contract-to-render transform.
        let pos = Vec3::new(row_major[3], row_major[7], row_major[11]);
        let render_pos = render_from_contract.transform_point3(pos);
        // Scale the instance's local rotation/scale uniformly so geometry keeps its proportions.
        let local = Mat4::from_cols(
            m.x_axis,
            m.y_axis,
            m.z_axis,
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
        );
        let render_mat = Mat4::from_translation(render_pos) * uniform * local;
        packed.extend_from_slice(&render_mat.to_cols_array());
    }
    packed
}

fn build_gpu_level(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    spec: TerrainScatterLevelSpec,
) -> Result<GpuScatterLevel> {
    let mesh = spec.mesh;
    if mesh.is_empty() {
        return Err(anyhow!("terrain scatter mesh level is empty"));
    }
    if !mesh.normals.is_empty() && mesh.normals.len() != mesh.positions.len() {
        return Err(anyhow!(
            "terrain scatter normals must match positions length when provided"
        ));
    }

    let vertices = mesh
        .positions
        .iter()
        .enumerate()
        .map(|(index, position)| VertexPN {
            position: *position,
            normal: mesh.normals.get(index).copied().unwrap_or([0.0, 1.0, 0.0]),
        })
        .collect::<Vec<_>>();

    let vertex_buffer_bytes = (vertices.len() * std::mem::size_of::<VertexPN>()) as u64;
    let index_buffer_bytes = (mesh.indices.len() * std::mem::size_of::<u32>()) as u64;

    let vertex_handle = register_buffer(
        vertex_buffer_bytes,
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    );
    let index_handle = register_buffer(
        index_buffer_bytes,
        wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
    );

    let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain.scatter.vertex_buffer"),
        size: vertex_buffer_bytes,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain.scatter.index_buffer"),
        size: index_buffer_bytes,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(&vertices));
    queue.write_buffer(&ibuf, 0, bytemuck::cast_slice(&mesh.indices));

    Ok(GpuScatterLevel {
        vbuf,
        ibuf,
        index_count: mesh.indices.len() as u32,
        max_distance: spec.max_distance.unwrap_or(f32::INFINITY),
        vertex_buffer_bytes,
        index_buffer_bytes,
        _vertex_handle: vertex_handle,
        _index_handle: index_handle,
    })
}

fn ensure_instance_capacity(
    device: &wgpu::Device,
    slot: &mut Option<ScatterInstanceBuffer>,
    count: usize,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }

    let needs_new = slot
        .as_ref()
        .map(|buffer| buffer.capacity < count)
        .unwrap_or(true);
    if !needs_new {
        return Ok(());
    }

    let capacity = count.next_power_of_two().max(64);
    let bytes = (capacity * 64) as u64;
    let handle = register_buffer(
        bytes,
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    );
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain.scatter.instance_buffer"),
        size: bytes,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    *slot = Some(ScatterInstanceBuffer {
        buffer,
        capacity,
        bytes,
        _handle: handle,
    });
    Ok(())
}

fn select_level_index(levels: &[GpuScatterLevel], distance: f32) -> usize {
    levels
        .iter()
        .position(|level| distance <= level.max_distance)
        .unwrap_or_else(|| levels.len().saturating_sub(1))
}

fn select_level_index_from_distances(max_distances: &[f32], distance: f32) -> usize {
    max_distances
        .iter()
        .position(|max_distance| distance <= *max_distance)
        .unwrap_or_else(|| max_distances.len().saturating_sub(1))
}

pub fn row_major_to_mat4(row_major: [f32; 16]) -> Mat4 {
    Mat4::from_cols_array(&[
        row_major[0],
        row_major[4],
        row_major[8],
        row_major[12],
        row_major[1],
        row_major[5],
        row_major[9],
        row_major[13],
        row_major[2],
        row_major[6],
        row_major[10],
        row_major[14],
        row_major[3],
        row_major[7],
        row_major[11],
        row_major[15],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_translation_uses_last_column() {
        let matrix = row_major_to_mat4([
            1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 20.0, 0.0, 0.0, 1.0, 30.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let translation = matrix.transform_point3(Vec3::ZERO);
        assert_eq!(translation, Vec3::new(10.0, 20.0, 30.0));
    }

    #[test]
    fn select_level_uses_first_matching_distance() {
        let max_distances = [10.0, 25.0];
        assert_eq!(select_level_index_from_distances(&max_distances, 4.0), 0);
        assert_eq!(select_level_index_from_distances(&max_distances, 20.0), 1);
        assert_eq!(select_level_index_from_distances(&max_distances, 200.0), 1);
    }

    #[test]
    fn total_buffer_bytes_sums_all_components() {
        let report = TerrainScatterMemoryReport {
            vertex_buffer_bytes: 10,
            index_buffer_bytes: 20,
            instance_buffer_bytes: 30,
            ..Default::default()
        };
        assert_eq!(report.total_buffer_bytes(), 60);
    }

    #[test]
    fn accumulate_frame_stats_resizes_lod_counts() {
        let mut frame = TerrainScatterFrameStats::default();
        accumulate_frame_stats(
            &mut frame,
            &TerrainScatterBatchStats {
                total_instances: 4,
                visible_instances: 3,
                culled_instances: 1,
                lod_instance_counts: vec![2, 1],
                ..Default::default()
            },
        );
        accumulate_frame_stats(
            &mut frame,
            &TerrainScatterBatchStats {
                total_instances: 2,
                visible_instances: 1,
                culled_instances: 1,
                lod_instance_counts: vec![0, 0, 1],
                ..Default::default()
            },
        );

        assert_eq!(frame.batch_count, 2);
        assert_eq!(frame.rendered_batches, 2);
        assert_eq!(frame.total_instances, 6);
        assert_eq!(frame.visible_instances, 4);
        assert_eq!(frame.culled_instances, 2);
        assert_eq!(frame.lod_instance_counts, vec![2, 1, 1]);
    }

    #[test]
    fn pack_preserves_uniform_instance_proportions() {
        // Instance with uniform scale=5 at position (10, 20, 30)
        let row_major = [
            5.0, 0.0, 0.0, 10.0, 0.0, 5.0, 0.0, 20.0, 0.0, 0.0, 5.0, 30.0, 0.0, 0.0, 0.0, 1.0,
        ];
        // Non-uniform render_from_contract: scale_xy=3 for X/Z, 1.0 for Y (swapped to Z)
        let render_from_contract = Mat4::from_cols_array(&[
            3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, -50.0, -50.0, -10.0, 1.0,
        ]);
        let instance_scale = 3.0_f32; // matches scale_xy

        let packed = pack_instance_transforms(&[row_major], render_from_contract, instance_scale);
        let m = Mat4::from_cols_array(packed[..16].try_into().unwrap());

        // The three column vectors (local axes) should all have equal length.
        let col0_len = Vec3::new(m.x_axis.x, m.x_axis.y, m.x_axis.z).length();
        let col1_len = Vec3::new(m.y_axis.x, m.y_axis.y, m.y_axis.z).length();
        let col2_len = Vec3::new(m.z_axis.x, m.z_axis.y, m.z_axis.z).length();
        let expected = 5.0 * 3.0; // original scale * instance_scale
        assert!((col0_len - expected).abs() < 1e-4, "col0={col0_len}");
        assert!((col1_len - expected).abs() < 1e-4, "col1={col1_len}");
        assert!((col2_len - expected).abs() < 1e-4, "col2={col2_len}");

        // Position should still go through render_from_contract properly.
        let pos = render_from_contract.transform_point3(Vec3::new(10.0, 20.0, 30.0));
        assert!((m.w_axis.x - pos.x).abs() < 1e-4);
        assert!((m.w_axis.y - pos.y).abs() < 1e-4);
        assert!((m.w_axis.z - pos.z).abs() < 1e-4);
    }

    #[test]
    fn validate_level_specs_rejects_non_increasing_lod_distances() {
        let levels = vec![
            TerrainScatterLevelSpec {
                mesh: MeshBuffers::default(),
                max_distance: Some(50.0),
            },
            TerrainScatterLevelSpec {
                mesh: MeshBuffers::default(),
                max_distance: Some(40.0),
            },
        ];
        let err = validate_level_specs(&levels).unwrap_err().to_string();
        assert!(err.contains("strictly increasing"));
    }

    #[test]
    fn validate_level_specs_rejects_non_final_open_ended_lod() {
        let levels = vec![
            TerrainScatterLevelSpec {
                mesh: MeshBuffers::default(),
                max_distance: None,
            },
            TerrainScatterLevelSpec {
                mesh: MeshBuffers::default(),
                max_distance: Some(80.0),
            },
        ];
        let err = validate_level_specs(&levels).unwrap_err().to_string();
        assert!(err.contains("final terrain scatter LOD level"));
    }

    #[test]
    fn validate_transforms_rejects_non_finite_values() {
        let err = validate_transforms(&[[
            1.0,
            0.0,
            0.0,
            10.0,
            0.0,
            1.0,
            0.0,
            20.0,
            0.0,
            0.0,
            f32::NAN,
            30.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]])
        .unwrap_err()
        .to_string();
        assert!(err.contains("finite"));
    }

    #[test]
    fn validate_optional_distance_rejects_invalid_values() {
        let err = validate_optional_distance(Some(-1.0), "distance")
            .unwrap_err()
            .to_string();
        assert!(err.contains("positive finite"));
    }

    #[test]
    fn hlod_stats_fields_default_to_zero() {
        let stats = TerrainScatterBatchStats::default();
        assert_eq!(stats.hlod_cluster_draws, 0);
        assert_eq!(stats.hlod_covered_instances, 0);
        assert_eq!(stats.effective_draws, 0);
    }

    #[test]
    fn memory_report_includes_hlod_in_total() {
        let report = TerrainScatterMemoryReport {
            vertex_buffer_bytes: 10,
            index_buffer_bytes: 20,
            instance_buffer_bytes: 30,
            hlod_buffer_bytes: 40,
            ..Default::default()
        };
        assert_eq!(report.total_buffer_bytes(), 100);
    }

    #[test]
    fn mesh_bounding_radius_computes_max_distance_from_origin() {
        let mesh = crate::geometry::MeshBuffers {
            positions: vec![
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.5],
            ],
            normals: vec![[0.0, 1.0, 0.0]; 3],
            indices: vec![0, 1, 2],
            ..Default::default()
        };
        let r = mesh_bounding_radius(&mesh);
        assert!((r - 2.0).abs() < 1e-5, "radius should be 2.0, got {r}");
    }

    #[test]
    fn max_instance_scale_extracts_uniform_scale() {
        // Identity with scale=5
        let row = [
            5.0, 0.0, 0.0, 0.0,
            0.0, 5.0, 0.0, 0.0,
            0.0, 0.0, 5.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let s = max_instance_scale(&row);
        assert!((s - 5.0).abs() < 1e-5, "scale should be 5.0, got {s}");
    }

    #[test]
    fn max_instance_scale_extracts_nonuniform_scale() {
        // scale_x=2, scale_y=7, scale_z=3
        let row = [
            2.0, 0.0, 0.0, 0.0,
            0.0, 7.0, 0.0, 0.0,
            0.0, 0.0, 3.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let s = max_instance_scale(&row);
        assert!((s - 7.0).abs() < 1e-5, "scale should be 7.0, got {s}");
    }

    #[test]
    fn accumulate_frame_stats_includes_hlod() {
        let mut frame = TerrainScatterFrameStats::default();
        accumulate_frame_stats(
            &mut frame,
            &TerrainScatterBatchStats {
                total_instances: 10,
                visible_instances: 5,
                culled_instances: 2,
                lod_instance_counts: vec![3, 2],
                hlod_cluster_draws: 2,
                hlod_covered_instances: 3,
                effective_draws: 5,
            },
        );
        assert_eq!(frame.hlod_cluster_draws, 2);
        assert_eq!(frame.hlod_covered_instances, 3);
        assert_eq!(frame.effective_draws, 5);
    }
}
