use super::types::*;
use crate::path_tracing::alias_table::AliasTable;
use crate::path_tracing::lighting::{GpuAreaLight, GpuDirectionalLight};
use std::f32::consts::PI;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferUsages, Device};

/// Create a GPU storage buffer of `Reservoir` elements initialized to default
pub fn create_reservoir_buffer(device: &Device, count: usize) -> Buffer {
    let data = vec![Reservoir::default(); count.max(1)];
    let bytes = bytemuck::cast_slice(&data);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir-reservoir-buffer"),
        contents: bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

/// Create a GPU storage buffer with provided `LightSample` elements
pub fn create_light_samples_buffer(device: &Device, samples: &[LightSample]) -> Buffer {
    let bytes = bytemuck::cast_slice(samples);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir-light-samples-buffer"),
        contents: bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

/// Create an empty light samples buffer (0 elements)
pub fn empty_light_samples_buffer(device: &Device) -> Buffer {
    // Allocate 64 bytes (minimum one WGSL LightSample with vec3 padding) to satisfy binding size
    let zeros: [u32; 16] = [0; 16];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir-light-samples-buffer"),
        contents: bytemuck::cast_slice(&zeros),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

/// Create a GPU storage buffer with AliasEntry items for light sampling on GPU
pub fn create_alias_entries_buffer(
    device: &Device,
    entries: &[crate::path_tracing::alias_table::AliasEntry],
) -> Buffer {
    let bytes = bytemuck::cast_slice(entries);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir-alias-entries-buffer"),
        contents: bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

/// Create an empty alias entries buffer (0 elements)
pub fn empty_alias_entries_buffer(device: &Device) -> Buffer {
    // Allocate at least 64 bytes to avoid small binding size issues
    let zeros: [u32; 16] = [0; 16];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir-alias-entries-buffer"),
        contents: bytemuck::cast_slice(&zeros),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

/// Build LightSample array and alias entries from area + directional lights, and upload to GPU buffers
pub fn build_light_samples_and_alias(
    device: &Device,
    area_lights: &[GpuAreaLight],
    directional_lights: &[GpuDirectionalLight],
) -> (Buffer, Buffer, Buffer) {
    fn luminance(c: [f32; 3]) -> f32 {
        0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    }

    let mut samples: Vec<LightSample> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();

    // Pack area lights (light_index = per-type index)
    for (i, a) in area_lights.iter().enumerate() {
        let idx = i as u32;
        let lum = luminance(a.color);
        let area = PI * a.radius * a.radius;
        let imp = if a.importance > 0.0 {
            a.importance
        } else {
            1.0
        };
        let w = (a.intensity * lum * area * imp).max(0.0);
        weights.push(w);

        samples.push(LightSample {
            position: a.position,
            light_index: idx,
            // Area lights do not use a delta direction here; keep a fixed up-vector.
            direction: [0.0, 1.0, 0.0],
            // Store photometric intensity proxy
            intensity: (a.intensity * lum).max(0.0),
            light_type: 2, // 2 = area
            // params.x = radius
            params: [a.radius, 0.0, 0.0],
            _pad: [0; 4],
        });
    }

    // Pack directional lights (light_index = per-type index)
    for (i, d) in directional_lights.iter().enumerate() {
        let idx = i as u32;
        let lum = luminance(d.color);
        let imp = if d.importance > 0.0 {
            d.importance
        } else {
            1.0
        };
        let w = (d.intensity * lum * imp).max(0.0);
        weights.push(w);

        // Directional lights travel along +direction; shading uses wi = -direction
        let dir = d.direction;
        let wi = [-dir[0], -dir[1], -dir[2]];
        samples.push(LightSample {
            position: [0.0, 0.0, 0.0],
            light_index: idx,
            direction: wi,
            intensity: (d.intensity * lum).max(0.0),
            light_type: 1, // 1 = directional
            params: [0.0, 0.0, 0.0],
            _pad: [0; 4],
        });
    }

    // If no lights, return empty buffers
    if samples.is_empty() {
        return (
            empty_light_samples_buffer(device),
            empty_alias_entries_buffer(device),
            empty_light_probs_buffer(device),
        );
    }

    let table = AliasTable::new(&weights);
    let light_buf = create_light_samples_buffer(device, &samples);
    let alias_buf = create_alias_entries_buffer(device, table.entries());
    // Build normalized probabilities
    let sum_w: f32 = weights.iter().copied().sum();
    let probs: Vec<f32> = if sum_w > 0.0 {
        weights.iter().map(|w| w / sum_w).collect()
    } else {
        let n = weights.len() as f32;
        weights.iter().map(|_| 1.0f32 / n).collect()
    };
    let probs_buf = create_light_probs_buffer(device, &probs);
    (light_buf, alias_buf, probs_buf)
}

/// Create GPU buffer for per-light probabilities
pub fn create_light_probs_buffer(device: &Device, probs: &[f32]) -> Buffer {
    let bytes = bytemuck::cast_slice(probs);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir-light-probs-buffer"),
        contents: bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

pub fn empty_light_probs_buffer(device: &Device) -> Buffer {
    // Allocate at least 64 bytes to avoid small binding size issues
    let zeros: [u32; 16] = [0; 16];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir-light-probs-buffer"),
        contents: bytemuck::cast_slice(&zeros),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

/// Create diagnostics flags buffer for spatial reuse visualization (one u32 per pixel)
pub fn create_diag_flags_buffer(device: &Device, pixel_count: usize) -> Buffer {
    let zeros: Vec<u32> = vec![0u32; pixel_count];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir-diag-flags-buffer"),
        contents: bytemuck::cast_slice(&zeros),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

/// Create debug AOV buffer (vec4 per pixel) for color-coded diagnostics
pub fn create_debug_aov_buffer(device: &Device, pixel_count: usize) -> Buffer {
    let zeros: Vec<[f32; 4]> = vec![[0.0, 0.0, 0.0, 0.0]; pixel_count];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir-debug-aov-buffer"),
        contents: bytemuck::cast_slice(&zeros),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    })
}

/// Create ReSTIR G-buffer (vec4 per pixel): normal.xyz, roughness
pub fn create_restir_gbuffer(device: &Device, pixel_count: usize) -> Buffer {
    let zeros: Vec<[f32; 4]> = vec![[0.0, 0.0, 1.0, 1.0]; pixel_count];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir-gbuffer-nrm-rough"),
        contents: bytemuck::cast_slice(&zeros),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    })
}

/// Create ReSTIR G-buffer for world position (vec4 per pixel): position.xyz, 1.0
pub fn create_restir_gbuffer_pos(device: &Device, pixel_count: usize) -> Buffer {
    let zeros: Vec<[f32; 4]> = vec![[0.0, 0.0, 0.0, 1.0]; pixel_count];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir-gbuffer-pos"),
        contents: bytemuck::cast_slice(&zeros),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    })
}
