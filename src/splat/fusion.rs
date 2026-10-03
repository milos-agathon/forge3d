// src/splat/fusion.rs
// Fused scene for the single ReSTIR integrator: Gaussian splat pages, COPC /
// LiDAR pages and terrain tiles under ONE top-level acceleration structure,
// with the GPU residency pool, page table and paging service that keep a
// billion-primitive index renderable inside the memory budget.
//
// GPU layout (hybrid kernel group 1, bindings 5..=11):
//   5  FusionUniforms            uniform
//   6  fused BVH                 TLAS nodes, then one per-page tree per slot
//   7  page table + feedback     atomic<u32>: header, then 4 words per page
//   8  splat SoA records         position/opacity, SH colour
//   9  packed inverse covariance one per splat record
//   10 COPC / LiDAR point pages  position/radius, colour
//   11 VolumetricParams          participating-media convention
// RELEVANT FILES: src/shaders/fusion/unified_occlusion.wgsl,
//                 src/path_tracing/hybrid_compute/render_fused.rs,
//                 src/splat/stream.rs

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use super::bvh::{
    blas_node_bound, build_tlas, Aabb, FusionBvhNode, TlasLeaf, KIND_LIDAR, KIND_SPLAT,
    KIND_TERRAIN,
};
use super::kernel::ACCEPT_THRESHOLD;
use super::stream::{
    DecodedPage, InvCovGpu, PageAddress, PageKind, PageLoader, PageMeta, PagePayload, PageSource,
    PointGpu, Residency, ResidencyStats, SplatGpu, NOT_RESIDENT,
};
use crate::core::error::RenderError;
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, tracked_host_allocation, AllocationOwner,
    ResourceHandle, TrackedBuffer,
};
use crate::viewer::viewer_types::VolumetricUniformsStd140;

/// Default primitives per page (and per residency slot).
pub const DEFAULT_PAGE_CAPACITY: u32 = 4096;
/// Words in the page-table header: `[0]` counts the frame's page misses.
pub const PAGE_TABLE_HEADER_WORDS: usize = 4;
/// Words per page-table entry: slot, request count, touched flag, reserved.
pub const PAGE_TABLE_ENTRY_WORDS: usize = 4;
/// Storage buffers the fused `main_terrain` stage binds (hybrid kernel's 8
/// plus the five fused ones). Devices below this cannot run the fused path.
pub const REQUIRED_STORAGE_BUFFERS_PER_STAGE: u32 = 13;

/// How a traversal miss on a non-resident page is resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PagingPolicy {
    /// Offline/deterministic: a frame that touched a missing page is redone
    /// after the page is streamed in, so every accumulated frame saw exact
    /// visibility. A working set larger than the pool is an error.
    Exact,
    /// Interactive: missing pages are requested asynchronously and the
    /// affected samples reuse the reservoir's last-known visibility for that
    /// frame; residency converges over the following frames.
    Progressive,
}

/// Surface response parameters, laid out as `ShadingParamsGPU`
/// (src/shaders/lighting.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct ShadingParams {
    pub brdf: u32,
    pub metallic: f32,
    pub roughness: f32,
    pub sheen: f32,
    pub clearcoat: f32,
    pub subsurface: f32,
    pub anisotropy: f32,
    pub _pad: f32,
}

impl Default for ShadingParams {
    fn default() -> Self {
        Self {
            brdf: 0, // BRDF_LAMBERT
            metallic: 0.0,
            roughness: 0.6,
            sheen: 0.0,
            clearcoat: 0.0,
            subsurface: 0.0,
            anisotropy: 0.0,
            _pad: 0.0,
        }
    }
}

/// Exponential height fog following the `VolumetricParams` convention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MediaParams {
    pub density: f32,
    pub height_falloff: f32,
    pub phase_g: f32,
}

impl Default for MediaParams {
    fn default() -> Self {
        Self {
            density: 0.0,
            height_falloff: 0.0,
            phase_g: 0.0,
        }
    }
}

/// Parameters of the unified occlusion model and the residency pool.
#[derive(Clone, Debug, PartialEq)]
pub struct FusionParams {
    /// Splat extinction scale: optical depth of a fully opaque splat through
    /// its centre (`T = exp(-kappa * rho)`).
    pub kappa: f32,
    /// Fixed sphelet radius of a LiDAR return (world units).
    pub lidar_radius: f32,
    /// Peak coverage of a LiDAR return (`c = opacity * (1 - b^2 / r^2)`).
    pub lidar_opacity: f32,
    /// Shadow rays stop once the running transmittance drops below this.
    pub transmittance_epsilon: f32,
    /// Splats/points occlude environment light only within this distance
    /// (terrain occludes at any distance); bounds the paged working set.
    pub ibl_occlusion_distance: f32,
    /// Angular radius of the sun disc the reservoir light sample covers.
    pub sun_angular_radius_deg: f32,
    /// Defensive floor of the visibility-aware ReSTIR target.
    pub restir_defensive: f32,
    /// Secondary rays leaving a splat skip the stretch in which they rise
    /// this many standard deviations (of the hit splat, along its normal),
    /// so a surface made of overlapping splats does not shadow itself.
    pub splat_self_bias_sigmas: f32,
    /// The same for LiDAR returns, in sphelet radii.
    pub lidar_self_bias_radii: f32,
    /// Render LiDAR returns on locally planar neighbourhoods as oriented
    /// discs (surfels) instead of spheres, so dense surface swaths do not
    /// shadow themselves at grazing sun angles.
    pub lidar_surfels: bool,
    /// Shade terrain with C0-continuous interpolated vertex normals instead
    /// of the per-cell bilinear-patch normal (removes cell-edge streaks on
    /// steep slopes; intersection stays the exact patch).
    pub terrain_smooth_normals: bool,
    pub shading: ShadingParams,
    pub media: MediaParams,
    pub page_capacity: u32,
    pub splat_slots: u32,
    pub point_slots: u32,
    pub policy: PagingPolicy,
    pub loader_threads: usize,
}

impl Default for FusionParams {
    fn default() -> Self {
        Self {
            kappa: 4.0,
            lidar_radius: 0.2,
            lidar_opacity: 1.0,
            transmittance_epsilon: 1e-3,
            ibl_occlusion_distance: 12.0,
            sun_angular_radius_deg: 0.2665,
            restir_defensive: 0.25,
            splat_self_bias_sigmas: 2.0,
            lidar_self_bias_radii: 1.0,
            terrain_smooth_normals: true,
            lidar_surfels: true,
            shading: ShadingParams::default(),
            media: MediaParams::default(),
            page_capacity: DEFAULT_PAGE_CAPACITY,
            splat_slots: 96,
            point_slots: 96,
            policy: PagingPolicy::Exact,
            loader_threads: 2,
        }
    }
}

impl FusionParams {
    pub fn validate(&self) -> Result<(), RenderError> {
        let fail = |message: String| Err(RenderError::Render(message));
        if !(self.kappa.is_finite() && self.kappa > 0.0) {
            return fail(format!("splat kappa must be finite and > 0, got {}", self.kappa));
        }
        if !(self.lidar_radius.is_finite() && self.lidar_radius > 0.0) {
            return fail(format!(
                "LiDAR sphelet radius must be finite and > 0, got {}",
                self.lidar_radius
            ));
        }
        if !(self.lidar_opacity > 0.0 && self.lidar_opacity <= 1.0) {
            return fail(format!(
                "LiDAR opacity must be in (0, 1], got {}",
                self.lidar_opacity
            ));
        }
        if !(self.transmittance_epsilon > 0.0 && self.transmittance_epsilon < 0.5) {
            return fail(format!(
                "transmittance epsilon must be in (0, 0.5), got {}",
                self.transmittance_epsilon
            ));
        }
        if !(self.ibl_occlusion_distance.is_finite() && self.ibl_occlusion_distance > 0.0) {
            return fail("environment occlusion distance must be finite and > 0".into());
        }
        if !(0.0..=5.0).contains(&self.sun_angular_radius_deg) {
            return fail(format!(
                "sun angular radius must be in [0, 5] degrees, got {}",
                self.sun_angular_radius_deg
            ));
        }
        if !(self.restir_defensive > 0.0 && self.restir_defensive <= 1.0) {
            return fail(format!(
                "ReSTIR defensive floor must be in (0, 1], got {}",
                self.restir_defensive
            ));
        }
        if !(self.splat_self_bias_sigmas >= 0.0 && self.splat_self_bias_sigmas.is_finite())
            || !(self.lidar_self_bias_radii >= 0.0 && self.lidar_self_bias_radii.is_finite())
        {
            return fail("self-shadow bias factors must be finite and >= 0".into());
        }
        if self.shading.brdf > 12 {
            return fail(format!("unknown BRDF model index {}", self.shading.brdf));
        }
        if !(self.media.density >= 0.0 && self.media.density.is_finite())
            || !self.media.height_falloff.is_finite()
        {
            return fail("fog density must be finite and >= 0 with a finite height falloff".into());
        }
        if self.page_capacity == 0 || self.page_capacity > 1 << 16 {
            return fail(format!(
                "page capacity must be in 1..=65536, got {}",
                self.page_capacity
            ));
        }
        Ok(())
    }
}

/// Uniform block mirrored by `FusionUniforms` in unified_occlusion.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct FusionUniforms {
    /// tlas_node_count, page_count, blas_base (node index), nodes_per_slot.
    pub counts: [u32; 4],
    /// splat_slots, point_slots, prims_per_slot, flags.
    pub pools: [u32; 4],
    /// kappa, lidar_opacity, transmittance_epsilon, ibl_occlusion_distance.
    pub optics: [f32; 4],
    /// cos(sun angular radius), restir_defensive, accept_threshold, ray_epsilon.
    pub sampling: [f32; 4],
    /// splat self-bias (sigmas), LiDAR self-bias (radii), unused, unused.
    pub surface: [f32; 4],
    pub shading: ShadingParams,
}

/// The fused scene description: every page of every source under one global
/// page-id space. Only this index is resident; primitives stay in their
/// sources until a traversal asks for them.
pub struct FusedScene {
    sources: Arc<Vec<Arc<dyn PageSource>>>,
    addresses: Vec<PageAddress>,
    metas: Vec<PageMeta>,
    params: FusionParams,
    logical: u64,
    _tracker: ResourceHandle,
}

impl FusedScene {
    /// Assemble the scene from its page sources. Every page must fit a
    /// residency slot; anything else is a diagnostic, never a truncation.
    pub fn new(
        sources: Vec<Arc<dyn PageSource>>,
        params: FusionParams,
    ) -> Result<Self, RenderError> {
        params.validate()?;
        let mut addresses = Vec::new();
        let mut metas = Vec::new();
        let mut logical = 0u64;
        for (source_index, source) in sources.iter().enumerate() {
            logical += source.logical_primitives();
            for local in 0..source.page_count() {
                let meta = source.meta(local);
                if meta.count == 0 || meta.count > params.page_capacity {
                    return Err(RenderError::Render(format!(
                        "{}: page {local} holds {} primitives but a residency slot holds \
                         1..={}; rebuild the source with a matching page capacity",
                        source.label(),
                        meta.count,
                        params.page_capacity
                    )));
                }
                if !meta.aabb.is_valid() {
                    return Err(RenderError::Render(format!(
                        "{}: page {local} has an invalid bounding box",
                        source.label()
                    )));
                }
                addresses.push((source_index as u32, local));
                metas.push(meta);
            }
        }
        if metas.len() as u64 * u64::from(params.page_capacity) > u64::from(u32::MAX) {
            return Err(RenderError::Render(format!(
                "{} pages x {} primitives exceed the 32-bit primitive id space",
                metas.len(),
                params.page_capacity
            )));
        }
        let has = |kind: PageKind| metas.iter().any(|meta| meta.kind == kind);
        if has(PageKind::Splat) && params.splat_slots == 0 {
            return Err(RenderError::Render(
                "the scene contains splat pages but the residency pool has no splat slots".into(),
            ));
        }
        if has(PageKind::Points) && params.point_slots == 0 {
            return Err(RenderError::Render(
                "the scene contains point pages but the residency pool has no point slots".into(),
            ));
        }
        let index_bytes = metas.len() * (std::mem::size_of::<PageMeta>() + 8);
        Ok(Self {
            sources: Arc::new(sources),
            addresses,
            metas,
            params,
            logical,
            _tracker: tracked_host_allocation(index_bytes as u64, "fusion-scene-index")?,
        })
    }

    pub fn params(&self) -> &FusionParams {
        &self.params
    }

    pub fn page_count(&self) -> u32 {
        self.metas.len() as u32
    }

    /// Primitives the scene describes, resident or not.
    pub fn logical_primitive_count(&self) -> u64 {
        self.logical
    }

    pub fn meta(&self, page: u32) -> PageMeta {
        self.metas[page as usize]
    }

    /// Traversal box of a page: point pages grow by the sphelet radius.
    pub fn page_aabb(&self, page: u32) -> Aabb {
        let meta = self.metas[page as usize];
        match meta.kind {
            PageKind::Splat => meta.aabb,
            PageKind::Points => {
                let r = self.params.lidar_radius;
                Aabb::new(meta.aabb.min.map(|v| v - r), meta.aabb.max.map(|v| v + r))
            }
        }
    }

    /// Leaves of the single top-level structure: (a) splat page proxies,
    /// (b) COPC octree node boxes, (c) terrain tiles.
    pub fn tlas_leaves(&self, terrain_tiles: &[Aabb]) -> Vec<TlasLeaf> {
        let mut leaves: Vec<TlasLeaf> = (0..self.page_count())
            .map(|page| TlasLeaf {
                aabb: self.page_aabb(page),
                kind: match self.metas[page as usize].kind {
                    PageKind::Splat => KIND_SPLAT,
                    PageKind::Points => KIND_LIDAR,
                },
                id: page,
            })
            .collect();
        leaves.extend(terrain_tiles.iter().enumerate().map(|(tile, aabb)| TlasLeaf {
            aabb: *aabb,
            kind: KIND_TERRAIN,
            id: tile as u32,
        }));
        leaves
    }

    /// Synchronously read one page's attributes (reference builders, tests).
    pub fn load_page(&self, page: u32) -> Result<PagePayload, RenderError> {
        let (source, local) = self.addresses[page as usize];
        self.sources[source as usize].load(local)
    }

    /// Synchronously decode one page into its GPU records.
    pub fn decode_page(&self, page: u32) -> Result<DecodedPage, RenderError> {
        DecodedPage::from_payload(
            page,
            &self.load_page(page)?,
            self.params.lidar_radius,
            self.params.lidar_surfels,
        )
    }

    /// Human-readable source list for diagnostics.
    pub fn describe(&self) -> String {
        self.sources
            .iter()
            .map(|source| source.label())
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// What one paging service call did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ServiceReport {
    /// Traversal miss events the GPU counted since the last service.
    pub miss_events: u32,
    /// Distinct pages requested.
    pub requested_pages: u32,
    /// Pages streamed into the pool by this call.
    pub loaded_pages: u32,
    /// Requested pages that could not be made resident (pool pinned).
    pub deferred_pages: u32,
}

/// Cumulative paging statistics of a render.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PagingStats {
    pub miss_events: u64,
    pub services: u64,
    pub residency: ResidencyStats,
    pub deferred_pages: u64,
    pub pool_bytes: u64,
    pub index_bytes: u64,
}

/// GPU half of the fused scene: the buffers behind bindings 5..=11, the
/// residency bookkeeping and the asynchronous loader.
pub struct FusedGpu {
    pub uniforms: TrackedBuffer,
    pub bvh: TrackedBuffer,
    pub pages: TrackedBuffer,
    pub splats: TrackedBuffer,
    pub inv_cov: TrackedBuffer,
    pub points: TrackedBuffer,
    pub media: TrackedBuffer,
    readback: TrackedBuffer,
    table: Vec<u32>,
    table_dirty: bool,
    residency: Residency,
    loader: PageLoader,
    addresses: Vec<PageAddress>,
    blas_base: u32,
    nodes_per_slot: u32,
    page_capacity: u32,
    splat_slots: u32,
    tlas_node_count: u32,
    pending: Vec<u32>,
    stats: PagingStats,
    policy: PagingPolicy,
}

fn storage(
    device: &wgpu::Device,
    label: &'static str,
    bytes: u64,
) -> Result<TrackedBuffer, RenderError> {
    tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )
}

impl FusedGpu {
    /// Allocate the fixed residency pool and upload the top-level structure.
    /// `terrain_tiles` become TLAS leaves next to the page proxies.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &FusedScene,
        terrain_tiles: &[Aabb],
        owner: Option<AllocationOwner>,
    ) -> Result<Self, RenderError> {
        let granted = device.limits().max_storage_buffers_per_shader_stage;
        if granted < REQUIRED_STORAGE_BUFFERS_PER_STAGE {
            return Err(RenderError::DegradedCapability(format!(
                "the fused splat/LiDAR/terrain path binds {REQUIRED_STORAGE_BUFFERS_PER_STAGE} \
                 storage buffers per compute stage but this device grants {granted} \
                 (deterministic mode pins the WebGPU default of 8)"
            )));
        }
        let params = scene.params();
        let leaves = scene.tlas_leaves(terrain_tiles);
        let tlas = build_tlas(&leaves);
        let tlas_node_count = tlas.len() as u32;
        let nodes_per_slot = blas_node_bound(params.page_capacity as usize) as u32;
        let total_slots = params.splat_slots + params.point_slots;
        let blas_base = tlas_node_count.max(1);
        let node_bytes = std::mem::size_of::<FusionBvhNode>() as u64;
        let bvh_bytes =
            (u64::from(blas_base) + u64::from(total_slots) * u64::from(nodes_per_slot)) * node_bytes;
        let bvh = storage(device, "fusion-bvh", bvh_bytes)?;
        if !tlas.is_empty() {
            queue.write_buffer(&bvh, 0, bytemuck::cast_slice(&tlas));
        }
        drop(tlas);

        let page_count = scene.page_count();
        let table_words =
            PAGE_TABLE_HEADER_WORDS + PAGE_TABLE_ENTRY_WORDS * page_count.max(1) as usize;
        let mut table = vec![0u32; table_words];
        for page in 0..page_count.max(1) as usize {
            table[PAGE_TABLE_HEADER_WORDS + PAGE_TABLE_ENTRY_WORDS * page] = NOT_RESIDENT;
        }
        let pages = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("fusion-page-table"),
                contents: bytemuck::cast_slice(&table),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            },
        )?;
        let readback = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("fusion-page-table-readback"),
                size: (table_words * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )?;

        // A pool with no slots still binds one element so the layout stays
        // valid; it is never addressed because no page of that kind exists.
        let pool = |label: &'static str, slots: u32, stride: usize| {
            let records = (u64::from(slots) * u64::from(params.page_capacity)).max(1);
            storage(device, label, records * stride as u64)
        };
        let splats = pool(
            "fusion-splat-soa",
            params.splat_slots,
            std::mem::size_of::<SplatGpu>(),
        )?;
        let inv_cov = pool(
            "fusion-splat-inverse-covariance",
            params.splat_slots,
            std::mem::size_of::<InvCovGpu>(),
        )?;
        let points = pool(
            "fusion-copc-point-pages",
            params.point_slots,
            std::mem::size_of::<PointGpu>(),
        )?;

        let uniforms_value = FusionUniforms {
            counts: [tlas_node_count, page_count, blas_base, nodes_per_slot],
            pools: [
                params.splat_slots,
                params.point_slots,
                params.page_capacity,
                u32::from(!terrain_tiles.is_empty())
                    | (u32::from(params.terrain_smooth_normals) << 2),
            ],
            optics: [
                params.kappa,
                params.lidar_opacity,
                params.transmittance_epsilon,
                params.ibl_occlusion_distance,
            ],
            sampling: [
                params.sun_angular_radius_deg.to_radians().cos(),
                params.restir_defensive,
                ACCEPT_THRESHOLD,
                1e-3,
            ],
            surface: [
                params.splat_self_bias_sigmas,
                params.lidar_self_bias_radii,
                0.0,
                0.0,
            ],
            shading: params.shading,
        };
        let uniforms = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("fusion-uniforms"),
                contents: bytemuck::bytes_of(&uniforms_value),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        let media_value = VolumetricUniformsStd140 {
            density: params.media.density,
            height_falloff: params.media.height_falloff,
            phase_g: params.media.phase_g,
            max_steps: 0,
            start_distance: 0.0,
            max_distance: 1e30,
            _pad_a0: 0.0,
            _pad_a1: 0.0,
            scattering_color: [1.0; 3],
            absorption: 1.0,
            sun_direction: [0.0, 1.0, 0.0],
            sun_intensity: 1.0,
            ambient_color: [0.0; 3],
            temporal_alpha: 0.0,
            use_shadows: 1,
            jitter_strength: 0.0,
            frame_index: 0,
            _pad0: 0,
        };
        let media = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("fusion-volumetric-params"),
                contents: bytemuck::bytes_of(&media_value),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;

        let kinds = (0..page_count).map(|page| scene.meta(page).kind).collect();
        let pool_bytes = bvh.size()
            + pages.size()
            + splats.size()
            + inv_cov.size()
            + points.size()
            + readback.size();
        Ok(Self {
            uniforms,
            bvh,
            pages,
            splats,
            inv_cov,
            points,
            media,
            readback,
            table,
            table_dirty: false,
            residency: Residency::new(kinds, params.splat_slots, params.point_slots),
            loader: PageLoader::new(
                scene.sources.clone(),
                params.lidar_radius,
                params.lidar_surfels,
                params.loader_threads,
                owner,
            ),
            addresses: scene.addresses.clone(),
            blas_base,
            nodes_per_slot,
            page_capacity: params.page_capacity,
            splat_slots: params.splat_slots,
            tlas_node_count,
            pending: Vec::new(),
            stats: PagingStats {
                pool_bytes,
                index_bytes: u64::from(page_count)
                    * (std::mem::size_of::<PageMeta>() as u64 + 8),
                ..PagingStats::default()
            },
            policy: params.policy,
        })
    }

    /// Bind-group entries for hybrid kernel group 1, bindings 5..=11.
    pub fn bind_entries(&self) -> Vec<wgpu::BindGroupEntry<'_>> {
        [
            (5, &self.uniforms),
            (6, &self.bvh),
            (7, &self.pages),
            (8, &self.splats),
            (9, &self.inv_cov),
            (10, &self.points),
            (11, &self.media),
        ]
        .into_iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        })
        .collect()
    }

    pub fn tlas_node_count(&self) -> u32 {
        self.tlas_node_count
    }

    pub fn stats(&self) -> PagingStats {
        PagingStats {
            residency: self.residency.stats,
            ..self.stats
        }
    }

    /// Bytes of the fixed GPU residency pool (BVH, page table, record pools
    /// and the table readback staging).
    pub fn pool_bytes(&self) -> u64 {
        self.stats.pool_bytes
    }

    fn entry(page: u32) -> usize {
        PAGE_TABLE_HEADER_WORDS + PAGE_TABLE_ENTRY_WORDS * page as usize
    }

    fn read_table(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: u64,
    ) -> Result<Vec<u32>, RenderError> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fusion-page-table-readback"),
        });
        encoder.copy_buffer_to_buffer(&self.pages, 0, &self.readback, 0, bytes);
        queue.submit([encoder.finish()]);
        let slice = self.readback.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| RenderError::Readback("page-table map channel closed".into()))?
            .map_err(|e| RenderError::Readback(format!("page-table map failed: {e:?}")))?;
        let words = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
        self.readback.unmap();
        Ok(words)
    }

    fn upload(&mut self, queue: &wgpu::Queue, decoded: &DecodedPage, slot: u32) {
        let node_slot = match decoded.kind {
            PageKind::Splat => slot,
            PageKind::Points => self.splat_slots + slot,
        };
        let node_offset = (u64::from(self.blas_base)
            + u64::from(node_slot) * u64::from(self.nodes_per_slot))
            * std::mem::size_of::<FusionBvhNode>() as u64;
        queue.write_buffer(&self.bvh, node_offset, bytemuck::cast_slice(&decoded.nodes));
        let first = u64::from(slot) * u64::from(self.page_capacity);
        match decoded.kind {
            PageKind::Splat => {
                queue.write_buffer(
                    &self.splats,
                    first * std::mem::size_of::<SplatGpu>() as u64,
                    bytemuck::cast_slice(&decoded.splats),
                );
                queue.write_buffer(
                    &self.inv_cov,
                    first * std::mem::size_of::<InvCovGpu>() as u64,
                    bytemuck::cast_slice(&decoded.inv_cov),
                );
            }
            PageKind::Points => {
                queue.write_buffer(
                    &self.points,
                    first * std::mem::size_of::<PointGpu>() as u64,
                    bytemuck::cast_slice(&decoded.points),
                );
            }
        }
    }

    fn admit(
        &mut self,
        queue: &wgpu::Queue,
        decoded: DecodedPage,
        frame: u64,
        report: &mut ServiceReport,
    ) -> Result<(), RenderError> {
        if decoded.nodes.len() as u32 > self.nodes_per_slot || decoded.count > self.page_capacity {
            return Err(RenderError::Render(format!(
                "page {} decoded to {} nodes / {} primitives, beyond the slot capacity",
                decoded.page,
                decoded.nodes.len(),
                decoded.count
            )));
        }
        match self.residency.admit(decoded.page, frame) {
            Some(admission) => {
                if let Some(evicted) = admission.evicted {
                    self.table[Self::entry(evicted)] = NOT_RESIDENT;
                }
                self.upload(queue, &decoded, admission.slot);
                self.table[Self::entry(decoded.page)] = admission.slot;
                self.table_dirty = true;
                report.loaded_pages += 1;
            }
            None => {
                report.deferred_pages += 1;
                self.stats.deferred_pages += 1;
            }
        }
        Ok(())
    }

    /// Service page feedback after the frame's dispatches were submitted.
    ///
    /// Reads the miss counter; on misses reads the page table, updates LRU
    /// recency from the touched flags, and queues the missing pages in
    /// traversal-priority order (most-requested first). Under
    /// `PagingPolicy::Exact` it then waits for every queued page and fails
    /// if the working set cannot fit the pool; under `Progressive` it only
    /// collects pages that have already finished loading.
    pub fn service(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: u64,
    ) -> Result<ServiceReport, RenderError> {
        let mut report = ServiceReport::default();
        self.stats.services += 1;
        let header = self.read_table(device, queue, (PAGE_TABLE_HEADER_WORDS * 4) as u64)?;
        report.miss_events = header[0];
        self.stats.miss_events += u64::from(header[0]);

        if report.miss_events > 0 {
            let words = self.read_table(device, queue, (self.table.len() * 4) as u64)?;
            let mut requests: Vec<(u32, u32)> = Vec::new();
            for page in 0..self.residency.page_count() as u32 {
                let base = Self::entry(page);
                if words[base + 2] != 0 {
                    self.residency.touch(page, frame);
                }
                if words[base + 1] != 0
                    && self.residency.slot(page).is_none()
                    && !self.pending.contains(&page)
                {
                    requests.push((words[base + 1], page));
                }
            }
            // Traversal priority: the pages most rays asked for come first.
            requests.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            report.requested_pages = requests.len() as u32;
            for (_, page) in requests {
                self.loader.request(page, self.addresses[page as usize]);
                self.pending.push(page);
            }
            self.table_dirty = true;
        }

        match self.policy {
            PagingPolicy::Exact => {
                while let Some((page, result)) = self.loader.recv() {
                    self.pending.retain(|pending| *pending != page);
                    self.admit(queue, result?, frame, &mut report)?;
                }
                if report.deferred_pages > 0 {
                    return Err(RenderError::Budget(format!(
                        "fused scene working set does not fit the residency pool: {} page(s) \
                         needed by the current frame could not be admitted ({} splat slots, {} \
                         point slots, all pinned); raise the pool or use the progressive policy",
                        report.deferred_pages,
                        self.residency.slots(PageKind::Splat),
                        self.residency.slots(PageKind::Points)
                    )));
                }
            }
            PagingPolicy::Progressive => {
                while let Some((page, result)) = self.loader.try_recv() {
                    self.pending.retain(|pending| *pending != page);
                    self.admit(queue, result?, frame, &mut report)?;
                }
            }
        }

        if self.table_dirty {
            self.publish_table(queue);
        }
        Ok(report)
    }

    /// Publish the residency mirror and clear the traversal feedback
    /// (miss counter, request counts, touched flags).
    fn publish_table(&mut self, queue: &wgpu::Queue) {
        self.table[0] = 0;
        for page in 0..self.residency.page_count() {
            let base = PAGE_TABLE_HEADER_WORDS + PAGE_TABLE_ENTRY_WORDS * page;
            self.table[base + 1] = 0;
            self.table[base + 2] = 0;
        }
        queue.write_buffer(&self.pages, 0, bytemuck::cast_slice(&self.table));
        self.table_dirty = false;
    }

    /// Start a new view on this pool. The touched flags accumulate until the
    /// next miss, so the previous view's pages would otherwise still count
    /// as "used by the current frame" and could never be evicted.
    pub fn begin_view(&mut self, queue: &wgpu::Queue) {
        self.publish_table(queue);
    }

    /// Pages requested from the loader and not yet resident.
    pub fn pending_pages(&self) -> usize {
        self.pending.len()
    }

    /// Slot of a resident page (tests and diagnostics).
    pub fn resident_slot(&self, page: u32) -> Option<u32> {
        self.residency.slot(page)
    }
}

#[cfg(test)]
mod tests {
    use super::super::stream::{hash_unit, SplatCloudSource};
    use super::super::GaussianSplatCloud;
    use super::*;

    fn cloud(n: usize) -> Arc<GaussianSplatCloud> {
        let u = |i: usize, k: u32| hash_unit((i as u32).wrapping_mul(97).wrapping_add(k));
        Arc::new(
            GaussianSplatCloud::from_parts(
                (0..n)
                    .map(|i| [u(i, 1) * 10.0, u(i, 2) * 2.0, u(i, 3) * 10.0])
                    .collect(),
                vec![[0.1, 0.2, 0.1]; n],
                vec![[1.0, 0.0, 0.0, 0.0]; n],
                vec![0.8; n],
                vec![[0.2; 3]; n],
                None,
            )
            .unwrap(),
        )
    }

    #[test]
    fn uniform_layout_matches_the_wgsl_block() {
        assert_eq!(std::mem::size_of::<FusionUniforms>(), 112);
        assert_eq!(std::mem::size_of::<ShadingParams>(), 32);
        assert_eq!(std::mem::size_of::<VolumetricUniformsStd140>(), 96);
        assert_eq!(std::mem::size_of::<SplatGpu>(), 80);
        assert_eq!(std::mem::size_of::<InvCovGpu>(), 32);
        assert_eq!(std::mem::size_of::<PointGpu>(), 32);
        assert_eq!(std::mem::size_of::<FusionBvhNode>(), 32);
    }

    #[test]
    fn scene_builds_one_tlas_over_pages_and_terrain_tiles() {
        let source: Arc<dyn PageSource> = Arc::new(SplatCloudSource::new(cloud(1000), 256));
        let params = FusionParams {
            page_capacity: 256,
            ..FusionParams::default()
        };
        let scene = FusedScene::new(vec![source], params).unwrap();
        assert_eq!(scene.logical_primitive_count(), 1000);
        assert_eq!(scene.page_count(), 4);
        let tiles = [
            Aabb::new([-5.0, 0.0, -5.0], [0.0, 1.0, 0.0]),
            Aabb::new([0.0, 0.0, 0.0], [5.0, 1.0, 5.0]),
        ];
        let leaves = scene.tlas_leaves(&tiles);
        assert_eq!(leaves.len(), 6);
        assert_eq!(leaves.iter().filter(|l| l.kind == KIND_TERRAIN).count(), 2);
        assert_eq!(leaves.iter().filter(|l| l.kind == KIND_SPLAT).count(), 4);
        let tlas = build_tlas(&leaves);
        assert_eq!(tlas.len(), 11);
        let decoded = scene.decode_page(2).unwrap();
        assert_eq!(decoded.page, 2);
        assert!(decoded.count <= 256);
    }

    #[test]
    fn oversized_pages_and_missing_pools_are_diagnosed() {
        let source: Arc<dyn PageSource> = Arc::new(SplatCloudSource::new(cloud(1000), 512));
        let too_small = FusedScene::new(
            vec![source.clone()],
            FusionParams {
                page_capacity: 256,
                ..FusionParams::default()
            },
        );
        assert!(format!("{}", too_small.err().unwrap()).contains("residency slot"));
        let no_pool = FusedScene::new(
            vec![source],
            FusionParams {
                page_capacity: 512,
                splat_slots: 0,
                ..FusionParams::default()
            },
        );
        assert!(format!("{}", no_pool.err().unwrap()).contains("no splat slots"));
        let bad = FusionParams {
            kappa: -1.0,
            ..FusionParams::default()
        };
        assert!(bad.validate().is_err());
    }
}
