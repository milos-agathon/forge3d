//! BOP-P2-02: runtime height-tile streaming for clipmap terrain.
//!
//! Connects camera-driven clipmap demand to the bounded asynchronous loader,
//! sparse physical height atlas, and GPU hash page table. The shader resolves
//! each requested `(lod, x, y)` through resident ancestors to the pinned root;
//! until that root arrives, the renderer keeps its ordinary overview height
//! binding. Uploads use a tracked, bounded staging ring with queue-completion
//! backpressure.
//!
//! Coarse CPU passes (height AO, sun visibility, shadows) intentionally keep
//! using the overview heightmap passed to each render call; streaming
//! refines the geometry/shading height source only.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::*;
use crate::core::resource_tracker::{
    owner_group_report, tracked_create_buffer, AllocationOwner, TrackedBuffer,
};
use crate::terrain::clipmap::{ClipmapConfig, ClipmapStreamer};
use crate::terrain::lod::LodConfig;
use crate::terrain::page_table::{
    AsyncTileLoader, CoalescePolicy, HeightReader, PageTable, TileLoadTerminal,
    MAX_LOADER_WORKERS,
};
use crate::terrain::stream::{HeightMosaic, MosaicConfig, PreparedHeightUpload};
use crate::terrain::tiling::{QuadTreeNode, TileBounds, TileId};
use crate::terrain::vt_family_residency::{FamilyResidencyTracker, TileKey as VtTileKey};
#[cfg(feature = "enable-globe")]
use glam::DVec3;
use glam::{Mat4, Vec2, Vec3};

/// ORBIS owns a deliberately smaller-than-512-MiB GPU-visible envelope. The
/// remaining byte makes the public limit strict even after integer rounding.
const ORBIS_GPU_VISIBLE_CAP_BYTES: u64 = 512 * 1024 * 1024 - 1;
const ORBIS_UPLOAD_RING_COUNT: usize = 3;
const ORBIS_MAX_IN_FLIGHT: usize = 1024;

fn validate_max_in_flight(max_in_flight: usize) -> Result<usize> {
    if max_in_flight == 0 || max_in_flight > ORBIS_MAX_IN_FLIGHT {
        return Err(anyhow!(
            "height streaming max_in_flight must be in 1..={ORBIS_MAX_IN_FLIGHT}"
        ));
    }
    Ok(max_in_flight)
}

fn validate_pool_size(pool_size: usize) -> Result<usize> {
    if pool_size > MAX_LOADER_WORKERS {
        return Err(anyhow!(
            "height streaming pool_size must be in 0..={MAX_LOADER_WORKERS} (0 selects one worker)"
        ));
    }
    Ok(pool_size.max(1))
}

struct HeightUploadFlight {
    _buffer: TrackedBuffer,
    complete: Arc<AtomicBool>,
}

/// Bounded mapped-at-creation uploads. No `Queue::write_*` call is used: each
/// admitted batch owns one tracked COPY_SRC buffer until the queue callback
/// proves the submission complete. Ring exhaustion is ordinary backpressure.
struct HeightUploadRing {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    owner: AllocationOwner,
    buffer_size: u64,
    flights: Vec<Option<HeightUploadFlight>>,
}

struct HeightUploadLayout {
    tile_offsets: Vec<(u64, u64)>,
    page_offset: u64,
    required_bytes: u64,
}

impl HeightUploadRing {
    fn align_up(value: u64, alignment: u64) -> Result<u64> {
        value
            .checked_add(alignment - 1)
            .map(|aligned| aligned / alignment * alignment)
            .ok_or_else(|| anyhow!("height upload layout overflow"))
    }

    /// Exact COPY_SRC layout shared by prebudgeting and submission.
    fn plan_layout(tile_lengths: &[(u64, u64)], page_bytes: u64) -> Result<HeightUploadLayout> {
        let mut cursor = 0u64;
        let mut tile_offsets = Vec::with_capacity(tile_lengths.len());
        for &(height_bytes, coverage_bytes) in tile_lengths {
            cursor = Self::align_up(cursor, 256)?;
            let height_offset = cursor;
            cursor = cursor
                .checked_add(height_bytes)
                .ok_or_else(|| anyhow!("height upload layout overflow"))?;
            cursor = Self::align_up(cursor, 256)?;
            let coverage_offset = cursor;
            cursor = cursor
                .checked_add(coverage_bytes)
                .ok_or_else(|| anyhow!("height upload layout overflow"))?;
            tile_offsets.push((height_offset, coverage_offset));
        }
        let page_offset = Self::align_up(cursor, 4)?;
        let required_bytes = page_offset
            .checked_add(page_bytes)
            .ok_or_else(|| anyhow!("height upload layout overflow"))?;
        Ok(HeightUploadLayout {
            tile_offsets,
            page_offset,
            required_bytes,
        })
    }

    fn required_capacity(
        max_in_flight: usize,
        height_bytes: u64,
        coverage_bytes: u64,
        page_bytes: u64,
    ) -> Result<u64> {
        let mut cursor = 0u64;
        for _ in 0..max_in_flight {
            cursor = Self::align_up(cursor, 256)?;
            cursor = cursor
                .checked_add(height_bytes)
                .ok_or_else(|| anyhow!("height upload layout overflow"))?;
            cursor = Self::align_up(cursor, 256)?;
            cursor = cursor
                .checked_add(coverage_bytes)
                .ok_or_else(|| anyhow!("height upload layout overflow"))?;
        }
        Self::align_up(cursor, 4)?
            .checked_add(page_bytes)
            .ok_or_else(|| anyhow!("height upload layout overflow"))
    }

    fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        owner: AllocationOwner,
        buffer_size: u64,
    ) -> Self {
        Self {
            device,
            queue,
            owner,
            buffer_size,
            flights: (0..ORBIS_UPLOAD_RING_COUNT).map(|_| None).collect(),
        }
    }

    fn prebudget_bytes(buffer_size: u64) -> u64 {
        buffer_size.saturating_mul(ORBIS_UPLOAD_RING_COUNT as u64)
    }

    fn reclaim(&mut self) {
        self.device.poll(wgpu::Maintain::Poll);
        for flight in &mut self.flights {
            if flight
                .as_ref()
                .is_some_and(|flight| flight.complete.load(Ordering::Acquire))
            {
                *flight = None;
            }
        }
    }

    fn has_capacity(&mut self) -> bool {
        self.reclaim();
        self.flights.iter().any(Option::is_none)
    }

    fn submit(
        &mut self,
        mosaic: &HeightMosaic,
        page_table: &PageTable,
        tiles: &[PreparedHeightUpload],
        page_bytes: &[u8],
    ) -> Result<bool> {
        self.reclaim();
        let Some(slot) = self.flights.iter().position(Option::is_none) else {
            return Ok(false);
        };
        let tile_lengths: Vec<_> = tiles
            .iter()
            .map(|tile| (tile.bytes.len() as u64, tile.coverage_bytes.len() as u64))
            .collect();
        let layout = Self::plan_layout(&tile_lengths, page_bytes.len() as u64)?;
        if layout.required_bytes > self.buffer_size {
            return Err(anyhow!(
                "height upload batch requires {} bytes, ring buffer holds {}",
                layout.required_bytes,
                self.buffer_size
            ));
        }
        let buffer = {
            let _scope = self.owner.activate_group("orbis.height");
            tracked_create_buffer(
                &self.device,
                &wgpu::BufferDescriptor {
                    label: Some("orbis.height.upload-staging"),
                    size: self.buffer_size,
                    usage: wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                },
            )?
        };
        {
            let mut mapped = buffer.slice(..).get_mapped_range_mut();
            for (tile, &(height_offset, coverage_offset)) in
                tiles.iter().zip(&layout.tile_offsets)
            {
                let height_offset = height_offset as usize;
                let coverage_offset = coverage_offset as usize;
                mapped[height_offset..height_offset + tile.bytes.len()]
                    .copy_from_slice(&tile.bytes);
                mapped[coverage_offset..coverage_offset + tile.coverage_bytes.len()]
                    .copy_from_slice(&tile.coverage_bytes);
            }
            let page_offset = layout.page_offset as usize;
            mapped[page_offset..page_offset + page_bytes.len()].copy_from_slice(page_bytes);
        }
        buffer.unmap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("orbis.height.upload-batch"),
            });
        for (tile, &(height_offset, coverage_offset)) in
            tiles.iter().zip(&layout.tile_offsets)
        {
            mosaic.encode_prepared_upload(
                &mut encoder,
                &buffer,
                height_offset,
                coverage_offset,
                tile,
            );
        }
        encoder.copy_buffer_to_buffer(
            &buffer,
            layout.page_offset,
            &page_table.buffer,
            0,
            page_bytes.len() as u64,
        );
        self.queue.submit([encoder.finish()]);
        let complete = Arc::new(AtomicBool::new(false));
        let callback_complete = complete.clone();
        self.queue.on_submitted_work_done(move || {
            callback_complete.store(true, Ordering::Release);
        });
        self.flights[slot] = Some(HeightUploadFlight {
            _buffer: buffer,
            complete,
        });
        Ok(true)
    }
}

/// Slices height tiles out of a caller-provided DEM by bilinear sampling.
/// Worker threads of `AsyncTileLoader` invoke `read` off the render thread.
pub(in crate::terrain::renderer) struct DemSliceHeightReader {
    dem: Vec<f32>,
    width: usize,
    height: usize,
}

impl DemSliceHeightReader {
    pub(in crate::terrain::renderer) fn new(dem: Vec<f32>, width: usize, height: usize) -> Self {
        Self { dem, width, height }
    }

    fn sample_bilinear(&self, nx: f32, ny: f32) -> f32 {
        let fx = (nx.clamp(0.0, 1.0)) * (self.width - 1) as f32;
        let fy = (ny.clamp(0.0, 1.0)) * (self.height - 1) as f32;
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let h00 = self.dem[y0 * self.width + x0];
        let h10 = self.dem[y0 * self.width + x1];
        let h01 = self.dem[y1 * self.width + x0];
        let h11 = self.dem[y1 * self.width + x1];
        let top = h00 * (1.0 - tx) + h10 * tx;
        let bottom = h01 * (1.0 - tx) + h11 * tx;
        top * (1.0 - ty) + bottom * ty
    }
}

impl HeightReader for DemSliceHeightReader {
    fn read_result(
        &self,
        root_bounds: &TileBounds,
        tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> std::result::Result<Vec<f32>, String> {
        let bounds = QuadTreeNode::calculate_bounds(root_bounds, tile_id, tile_size);
        let root_size = root_bounds.max - root_bounds.min;
        let mut out = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let u = x as f32 / (width - 1).max(1) as f32;
                let v = y as f32 / (height - 1).max(1) as f32;
                let world = bounds.min + Vec2::new(u, v) * (bounds.max - bounds.min);
                let nx = (world.x - root_bounds.min.x) / root_size.x.max(1e-6);
                let ny = (world.y - root_bounds.min.y) / root_size.y.max(1e-6);
                out.push(self.sample_bilinear(nx, ny));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
fn upsample_bilinear(src: &[f32], src_res: u32, dst_res: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((dst_res * dst_res) as usize);
    for y in 0..dst_res {
        for x in 0..dst_res {
            let fx = x as f32 / (dst_res - 1).max(1) as f32 * (src_res - 1) as f32;
            let fy = y as f32 / (dst_res - 1).max(1) as f32 * (src_res - 1) as f32;
            let x0 = fx.floor() as usize;
            let y0 = fy.floor() as usize;
            let x1 = (x0 + 1).min((src_res - 1) as usize);
            let y1 = (y0 + 1).min((src_res - 1) as usize);
            let tx = fx - x0 as f32;
            let ty = fy - y0 as f32;
            let s = src_res as usize;
            let top = src[y0 * s + x0] * (1.0 - tx) + src[y0 * s + x1] * tx;
            let bottom = src[y1 * s + x0] * (1.0 - tx) + src[y1 * s + x1] * tx;
            out.push(top * (1.0 - ty) + bottom * ty);
        }
    }
    out
}

/// Map a tile at an arbitrary LOD onto a fixed mosaic LOD: same LOD passes
/// through, finer tiles collapse to their ancestor, coarser tiles expand to
/// all covered descendants. Coordinates outside the mosaic axis are dropped.
fn map_tile_to_fixed_lod(tile: TileId, fixed_lod: u32, tiles_axis: u32, out: &mut HashSet<TileId>) {
    use std::cmp::Ordering;
    let clamp = |x: u32, y: u32| -> Option<(u32, u32)> {
        (x < tiles_axis && y < tiles_axis).then_some((x, y))
    };
    match tile.lod.cmp(&fixed_lod) {
        Ordering::Equal => {
            if let Some((x, y)) = clamp(tile.x, tile.y) {
                out.insert(TileId::new(fixed_lod, x, y));
            }
        }
        Ordering::Greater => {
            let shift = tile.lod - fixed_lod;
            if let Some((x, y)) = clamp(tile.x >> shift, tile.y >> shift) {
                out.insert(TileId::new(fixed_lod, x, y));
            }
        }
        Ordering::Less => {
            let shift = fixed_lod - tile.lod;
            let scale = 1u32 << shift;
            for dy in 0..scale {
                for dx in 0..scale {
                    if let Some((x, y)) = clamp(tile.x * scale + dx, tile.y * scale + dy) {
                        out.insert(TileId::new(fixed_lod, x, y));
                    }
                }
            }
        }
    }
}

pub(in crate::terrain::renderer) struct HeightStreamingStats {
    pub center: Vec2,
    pub pending_ring_tiles: usize,
    pub loaded_ring_tiles: usize,
    pub resident_fine_tiles: usize,
    pub total_tiles: usize,
    pub tiles_requested: usize,
    pub tiles_uploaded: usize,
    pub coarse_prefilled: usize,
    pub resident_height_bytes: u64,
    pub gpu_visible_current_bytes: u64,
    pub gpu_visible_high_water_bytes: u64,
    pub ancestor_fallbacks: usize,
    pub converged: bool,
    pub loader_pending: usize,
    pub loader_completed: usize,
}

/// Fourth-family VT runtime. Height uses an R32Float physical mosaic, but its
/// demand retention and residency accounting are the same feedback-driven
/// `TileKey`/family policy used by albedo, normal, and mask.
pub(in crate::terrain::renderer) struct HeightVtFamilyRuntime {
    residency_owner_id: u64,
    allocation_owner: AllocationOwner,
    pub(in crate::terrain::renderer) streamer: ClipmapStreamer,
    pub(in crate::terrain::renderer) mosaic: HeightMosaic,
    page_table: PageTable,
    upload_ring: HeightUploadRing,
    loader: AsyncTileLoader,
    lod: u32,
    tiles_axis: u32,
    tile_resolution: u32,
    gpu_visible_budget_bytes: u64,
    resident_fine: HashSet<TileId>,
    feedback_requests: crate::terrain::vt::requests::RetainedRequestSet,
    family_residency: FamilyResidencyTracker,
    /// ring tile -> fine tiles still missing before it counts as loaded
    ring_waiting: HashMap<TileId, HashSet<TileId>>,
    tiles_requested: usize,
    tiles_uploaded: usize,
    coarse_prefilled: usize,
    ancestor_fallbacks: usize,
    stream_epoch: u64,
    retry: HashMap<TileId, RetryState>,
    globe_mode: bool,
}

pub(in crate::terrain::renderer) enum HeightStreamingCamera {
    Flat(Vec3),
    #[cfg(feature = "enable-globe")]
    Globe(DVec3),
}

#[derive(Clone, Copy)]
struct RetryState {
    failures: u8,
    next_epoch: u64,
}

fn validate_prepared_final_mappings(
    prepared: Vec<PreparedHeightUpload>,
    mut final_slot: impl FnMut(TileId) -> Option<(u32, u32)>,
) -> (Vec<PreparedHeightUpload>, Vec<TileId>) {
    let mut retained = Vec::with_capacity(prepared.len());
    let mut displaced = Vec::new();
    for upload in prepared {
        if final_slot(upload.id) == Some((upload.sx, upload.sy)) {
            retained.push(upload);
        } else {
            displaced.push(upload.id);
        }
    }
    (retained, displaced)
}

impl HeightVtFamilyRuntime {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::terrain::renderer) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        allocation_owner: AllocationOwner,
        terrain_extent: f32,
        ring_count: u32,
        ring_resolution: u32,
        lod: u32,
        tile_resolution: u32,
        max_in_flight: usize,
        pool_size: usize,
        reader: Arc<dyn HeightReader>,
        coarse_prefill: bool,
        max_resident_bytes: Option<u64>,
        globe_mode: bool,
    ) -> Result<Self> {
        let tiles_axis = 1u32 << lod;
        let virtual_tiles = u64::from(tiles_axis) * u64::from(tiles_axis);
        let tile_bytes = u64::from(tile_resolution)
            * u64::from(tile_resolution)
            * (std::mem::size_of::<f32>() as u64 + 1);
        let gpu_visible_budget_bytes = max_resident_bytes
            .unwrap_or(ORBIS_GPU_VISIBLE_CAP_BYTES)
            .min(ORBIS_GPU_VISIBLE_CAP_BYTES);
        let max_in_flight = validate_max_in_flight(max_in_flight)?;
        let pool_size = validate_pool_size(pool_size)?;
        let padded_tile_row = (u64::from(tile_resolution) * 4).div_ceil(256) * 256;
        let padded_coverage_row = u64::from(tile_resolution).div_ceil(256) * 256;
        let staging_height_bytes = padded_tile_row.saturating_mul(u64::from(tile_resolution));
        let staging_coverage_bytes =
            padded_coverage_row.saturating_mul(u64::from(tile_resolution));
        let existing_group_bytes = owner_group_report(&allocation_owner, "orbis.height")
            .current_total_bytes();
        // The pinned root is fallback coverage, not one of the target-LOD
        // leaves. Budget a distinct slot so a comfortably sized cache can
        // converge all leaves without ever evicting that root.
        let mut slots = virtual_tiles.saturating_add(1).max(1) as u32;
        let (mosaic_tiles_x, mosaic_tiles_y, upload_buffer_size, prebudget_total) = loop {
            let tiles_x = (slots as f64).sqrt().ceil().max(1.0) as u32;
            let tiles_y = slots.div_ceil(tiles_x);
            let page_bytes = PageTable::allocation_bytes_for_capacity(slots as usize);
            let upload_buffer_size = HeightUploadRing::required_capacity(
                max_in_flight,
                staging_height_bytes,
                staging_coverage_bytes,
                page_bytes,
            )?;
            let actual_bytes = existing_group_bytes
                .saturating_add(
                    u64::from(tiles_x)
                        .saturating_mul(u64::from(tiles_y))
                        .saturating_mul(tile_bytes),
                )
                .saturating_add(page_bytes)
                .saturating_add(HeightUploadRing::prebudget_bytes(upload_buffer_size));
            if actual_bytes <= gpu_visible_budget_bytes {
                break (tiles_x, tiles_y, upload_buffer_size, actual_bytes);
            }
            if slots == 1 {
                return Err(anyhow!(
                    "height streaming prebudget {actual_bytes} exceeds strict GPU-visible budget {gpu_visible_budget_bytes}"
                ));
            }
            slots -= 1;
        };

        let half = terrain_extent * 0.5;
        let root_bounds = TileBounds::new(Vec2::new(-half, -half), Vec2::new(half, half));
        let tile_world_size = Vec2::splat(terrain_extent);

        let (mosaic, page_table) = {
            let _owner = allocation_owner.activate_group("orbis.height");
            let mosaic = HeightMosaic::new(
                &device,
                MosaicConfig {
                    tile_size_px: tile_resolution,
                    tiles_x: mosaic_tiles_x,
                    tiles_y: mosaic_tiles_y,
                    // A virtual region can be much larger than the physical
                    // atlas. The page table supplies slot indirection and
                    // ancestor fallback, so never allocate one slot per tile.
                    fixed_lod: None,
                },
                false,
            )?;
            let page_table = PageTable::new_sparse(
                &device,
                slots as usize,
                lod,
                tile_resolution,
                mosaic.config.texture_size(),
            )?;
            (mosaic, page_table)
        };
        let tracked_after_construction = owner_group_report(&allocation_owner, "orbis.height");
        if tracked_after_construction.current_total_bytes() > gpu_visible_budget_bytes {
            return Err(anyhow!(
                "tracked ORBIS height allocation {} exceeds GPU-visible budget {}",
                tracked_after_construction.current_total_bytes(),
                gpu_visible_budget_bytes
            ));
        }
        debug_assert!(prebudget_total <= gpu_visible_budget_bytes);
        let loader = AsyncTileLoader::new_with_reader(
            root_bounds.clone(),
            tile_world_size,
            tile_resolution,
            max_in_flight,
            pool_size,
            reader,
            CoalescePolicy::PreferFine,
        );
        let clipmap_config =
            ClipmapConfig::new(ring_count.clamp(1, 8), ring_resolution.clamp(4, 256));
        let streamer = if globe_mode {
            #[cfg(feature = "enable-globe")]
            {
                let radius = crate::terrain::clipmap::globe::GlobeFrame::WGS84_MEAN_RADIUS_M;
                let camera_anchor = DVec3::X * (radius + f64::from(terrain_extent.max(1.0)));
                let frame = crate::terrain::clipmap::globe::GlobeFrame::globe(radius, camera_anchor)
                    .map_err(|error| anyhow!("cannot create ORBIS globe frame: {error}"))?;
                ClipmapStreamer::new_globe(
                    clipmap_config,
                    DVec3::X * radius,
                    frame,
                    terrain_extent,
                )
                .ok_or_else(|| anyhow!("cannot create ORBIS globe clipmap streamer"))?
            }
            #[cfg(not(feature = "enable-globe"))]
            {
                return Err(anyhow!(
                    "COG globe height streaming requires the enable-globe feature"
                ));
            }
        } else {
            ClipmapStreamer::new(clipmap_config, Vec2::ZERO, terrain_extent)
        };

        let mut state = Self {
            residency_owner_id: crate::core::memory_tracker::global_tracker()
                .allocate_resident_owner(),
            allocation_owner: allocation_owner.clone(),
            streamer,
            mosaic,
            page_table,
            upload_ring: HeightUploadRing::new(
                device,
                queue,
                allocation_owner.clone(),
                upload_buffer_size,
            ),
            loader,
            lod,
            tiles_axis,
            tile_resolution,
            gpu_visible_budget_bytes,
            resident_fine: HashSet::new(),
            feedback_requests: Default::default(),
            family_residency: FamilyResidencyTracker::new(
                gpu_visible_budget_bytes,
                1u32 << crate::terrain::vt::HEIGHT_FAMILY,
                u64::from(tile_resolution) * u64::from(tile_resolution) * 5,
            ),
            ring_waiting: HashMap::new(),
            tiles_requested: 0,
            tiles_uploaded: 0,
            coarse_prefilled: 0,
            ancestor_fallbacks: 0,
            stream_epoch: 0,
            retry: HashMap::new(),
            globe_mode,
        };
        // Compatibility accepts the old flag, but startup is always
        // asynchronous. Until this request completes, all render paths keep
        // using the caller-provided overview texture and the disabled table.
        let _ = coarse_prefill;
        if state.loader.request(TileId::new(0, 0, 0)) {
            state.tiles_requested = 1;
        }
        Ok(state)
    }

    fn schedule_retry(&mut self, tile: TileId) {
        let failures = self
            .retry
            .get(&tile)
            .map_or(1, |state| state.failures.saturating_add(1));
        let delay = 1u64 << failures.min(6);
        self.retry.insert(
            tile,
            RetryState {
                failures,
                next_epoch: self.stream_epoch.saturating_add(delay),
            },
        );
    }

    fn request_due_retries(&mut self) {
        let due: Vec<_> = self
            .retry
            .iter()
            .filter_map(|(tile, state)| (state.next_epoch <= self.stream_epoch).then_some(*tile))
            .collect();
        for tile in due {
            if self.loader.request(tile) {
                self.tiles_requested += 1;
                // An actual terminal failure schedules the next delay. While
                // the ticket is active, dedup/backpressure bounds retries.
                self.retry.remove(&tile);
            }
        }
    }

    /// Map a tile at an arbitrary LOD onto the mosaic's fixed LOD.
    fn tiles_at_fixed_lod(&self, tile: TileId, out: &mut HashSet<TileId>) {
        map_tile_to_fixed_lod(tile, self.lod, self.tiles_axis, out);
    }

    fn touch_resident_or_ancestor(&mut self, requested: TileId) {
        if let Some(resident) = self.mosaic.touch_nearest_resident_ancestor(requested) {
            if resident != requested {
                self.ancestor_fallbacks += 1;
            }
        }
    }

    fn ring_tile_is_resident(&self, ring_tile: TileId) -> bool {
        let mut fine = HashSet::new();
        self.tiles_at_fixed_lod(ring_tile, &mut fine);
        !fine.is_empty() && fine.iter().all(|tile| self.resident_fine.contains(tile))
    }

    fn invalidate_evicted_leaf(&mut self, victim: TileId) {
        let affected: Vec<_> = self
            .streamer
            .required_tiles()
            .iter()
            .copied()
            .filter(|ring_tile| {
                let mut fine = HashSet::new();
                self.tiles_at_fixed_lod(*ring_tile, &mut fine);
                fine.contains(&victim)
            })
            .collect();
        self.streamer.invalidate_loaded(&affected);
        for ring_tile in affected {
            self.ring_waiting
                .entry(ring_tile)
                .or_default()
                .insert(victim);
        }
    }

    fn evict_before_upload(&mut self, incoming: TileId) -> bool {
        // Physical resources are allocated once under `allocation_owner`; an
        // upload is admitted only while those tracked bytes fit the strict
        // ORBIS envelope, and logical leaf residency is LRU-evicted first.
        if owner_group_report(&self.allocation_owner, "orbis.height").current_total_bytes()
            > self.gpu_visible_budget_bytes
        {
            return false;
        }
        while self.mosaic.slot_of(&incoming).is_none()
            && self.mosaic.resident_tile_count() >= self.page_table.capacity
        {
            let Some(victim) = self.mosaic.evict_least_recently_used_leaf() else {
                return false;
            };
            self.resident_fine.remove(&victim);
            self.invalidate_evicted_leaf(victim);
            self.family_residency.on_evict(&VtTileKey {
                family_slot: crate::terrain::vt::HEIGHT_FAMILY as u32,
                material_index: 0,
                x: victim.x,
                y: victim.y,
                mip_level: victim.lod,
            });
        }
        true
    }

    /// One streaming step. Visibility feedback is the primary demand signal;
    /// camera-derived ring tiles are lower-priority predictive prefetch.
    pub(in crate::terrain::renderer) fn stream_step(
        &mut self,
        _queue: &wgpu::Queue,
        camera: HeightStreamingCamera,
        feedback_uvs: &[[f32; 2]],
        max_uploads: usize,
    ) -> Result<HeightStreamingStats> {
        self.stream_epoch = self.stream_epoch.saturating_add(1);
        let mut active_fine_demand = HashSet::new();
        for uv in feedback_uvs {
            let x = (uv[0].clamp(0.0, 1.0 - f32::EPSILON) * self.tiles_axis as f32) as u32;
            let y = (uv[1].clamp(0.0, 1.0 - f32::EPSILON) * self.tiles_axis as f32) as u32;
            let tile = TileId::new(self.lod, x, y);
            active_fine_demand.insert(tile);
            self.touch_resident_or_ancestor(tile);
            if self.resident_fine.contains(&tile) {
                continue;
            }
            let key = VtTileKey {
                family_slot: crate::terrain::vt::HEIGHT_FAMILY as u32,
                material_index: 0,
                x,
                y,
                mip_level: self.lod,
            };
            self.feedback_requests[crate::terrain::vt::HEIGHT_FAMILY as usize].insert(key);
            if self.loader.request(tile) {
                self.tiles_requested += 1;
            }
        }

        let _new_ring_tiles = match camera {
            HeightStreamingCamera::Flat(camera_pos) if !self.globe_mode => {
                let lod_config = LodConfig::new(2.0, 1024, 768, 45.0f32.to_radians());
                self.streamer.update(
                    camera_pos,
                    Mat4::IDENTITY,
                    Mat4::IDENTITY,
                    &lod_config,
                )
            }
            #[cfg(feature = "enable-globe")]
            HeightStreamingCamera::Globe(camera_anchor) if self.globe_mode => {
                self.streamer.update_globe(camera_anchor)
            }
            _ => Vec::new(),
        };

        let required_ring_tiles = self.streamer.required_tiles().to_vec();
        let resident_ring_tiles: HashSet<_> = required_ring_tiles
            .iter()
            .copied()
            .filter(|ring_tile| self.ring_tile_is_resident(*ring_tile))
            .collect();
        let ring_tiles = self.streamer.reconcile_residency(&resident_ring_tiles);
        let required_ring_set: HashSet<_> = required_ring_tiles.iter().copied().collect();
        self.ring_waiting
            .retain(|ring_tile, _| required_ring_set.contains(ring_tile));
        for ring_tile in &required_ring_tiles {
            self.tiles_at_fixed_lod(*ring_tile, &mut active_fine_demand);
        }
        self.retry.retain(|tile, _| {
            *tile == TileId::new(0, 0, 0) || active_fine_demand.contains(tile)
        });
        self.request_due_retries();

        for ring_tile in &ring_tiles {
            let mut fine = HashSet::new();
            self.tiles_at_fixed_lod(*ring_tile, &mut fine);
            for tile in &fine {
                self.touch_resident_or_ancestor(*tile);
            }
            fine.retain(|t| !self.resident_fine.contains(t));
            if fine.is_empty() {
                self.streamer.mark_loaded(std::slice::from_ref(ring_tile));
                continue;
            }
            for t in &fine {
                self.feedback_requests[crate::terrain::vt::HEIGHT_FAMILY as usize].insert(
                    VtTileKey {
                        family_slot: crate::terrain::vt::HEIGHT_FAMILY as u32,
                        material_index: 0,
                        x: t.x,
                        y: t.y,
                        mip_level: t.lod,
                    },
                );
                if self.loader.request(*t) {
                    self.tiles_requested += 1;
                }
            }
            self.ring_waiting
                .entry(*ring_tile)
                .or_default()
                .extend(fine);
        }

        // `ClipmapStreamer::update` reports newly demanded ring keys; a tile
        // rejected by bounded-loader backpressure must still be retried on a
        // later frame. Keep the retry source bounded by the already-bounded
        // ring_waiting sets rather than adding another queue.
        let retry_tiles: HashSet<TileId> = self
            .ring_waiting
            .values()
            .flat_map(|missing| missing.iter().copied())
            .filter(|tile| !self.resident_fine.contains(tile))
            .collect();
        for tile in retry_tiles {
            self.touch_resident_or_ancestor(tile);
            if self.loader.request(tile) {
                self.tiles_requested += 1;
            }
        }

        let terminals = if self.upload_ring.has_capacity() {
            self.loader.drain_terminals(max_uploads.max(1))
        } else {
            Vec::new()
        };
        let mut prepared = Vec::new();
        for terminal in terminals {
            let td = match terminal {
                TileLoadTerminal::Complete(tile) => tile,
                TileLoadTerminal::Error(ticket) | TileLoadTerminal::Cancelled(ticket) => {
                    self.schedule_retry(ticket.tile_id);
                    continue;
                }
            };
            if td.width != self.tile_resolution
                || td.height != self.tile_resolution
                || (td.width * td.height) as usize != td.height_data.len()
                || td.coverage_data.len() != td.height_data.len()
            {
                self.schedule_retry(td.tile_id);
                continue;
            }
            if self.evict_before_upload(td.tile_id) {
                if let Ok(upload) = self.mosaic.prepare_covered_tile_upload(
                    td.tile_id,
                    &td.height_data,
                    &td.coverage_data,
                )
                {
                    prepared.push(upload);
                } else {
                    self.schedule_retry(td.tile_id);
                }
            } else {
                self.schedule_retry(td.tile_id);
            }
        }
        let (finalized, displaced) =
            validate_prepared_final_mappings(prepared, |id| self.mosaic.slot_of(&id));
        prepared = finalized;
        for tile in displaced {
            self.schedule_retry(tile);
        }
        if !prepared.is_empty() {
            let page_bytes = self
                .page_table
                .serialize(&self.mosaic)
                .map(|table| table.bytes())
                .map_err(|error| anyhow!("height page-table serialization failed: {error}"));
            let submission = page_bytes.and_then(|bytes| {
                self.upload_ring
                    .submit(&self.mosaic, &self.page_table, &prepared, &bytes)
                    .map_err(|error| anyhow!("height upload submission failed: {error}"))
            });
            let submitted = match submission {
                Ok(submitted) => submitted,
                Err(error) => {
                    for upload in prepared {
                        self.mosaic.remove_mapping(upload.id);
                    }
                    return Err(error);
                }
            };
            if submitted {
                for upload in &prepared {
                    let tile_id = upload.id;
                    if tile_id == TileId::new(0, 0, 0) {
                        self.retry.remove(&tile_id);
                        self.coarse_prefilled = 1;
                        self.tiles_uploaded += 1;
                        continue;
                    }
                    self.resident_fine.insert(tile_id);
                    self.retry.remove(&tile_id);
                    let key = VtTileKey {
                        family_slot: crate::terrain::vt::HEIGHT_FAMILY as u32,
                        material_index: 0,
                        x: tile_id.x,
                        y: tile_id.y,
                        mip_level: tile_id.lod,
                    };
                    self.feedback_requests[crate::terrain::vt::HEIGHT_FAMILY as usize].remove(&key);
                    self.family_residency.on_insert(key);
                    self.tiles_uploaded += 1;
                }
            } else {
                for upload in prepared {
                    self.mosaic.remove_mapping(upload.id);
                    self.schedule_retry(upload.id);
                }
            }
        }

        let mut satisfied: Vec<TileId> = Vec::new();
        for (ring_tile, missing) in self.ring_waiting.iter_mut() {
            missing.retain(|t| !self.resident_fine.contains(t));
            if missing.is_empty() {
                satisfied.push(*ring_tile);
            }
        }
        for ring_tile in &satisfied {
            self.ring_waiting.remove(ring_tile);
        }
        if !satisfied.is_empty() {
            self.streamer.mark_loaded(&satisfied);
        }

        let stats = self.stats();
        let height = self
            .family_residency
            .family(crate::terrain::vt::HEIGHT_FAMILY as u32);
        super::virtual_texture::publish_height_family_stats(
            self.residency_owner_id,
            height.resident_tiles,
            height.resident_bytes,
            height.budget_bytes,
            self.feedback_requests
                .iter()
                .map(|bucket| bucket.len() as u32)
                .sum(),
        );
        Ok(stats)
    }

    pub(in crate::terrain::renderer) fn stats(&self) -> HeightStreamingStats {
        let total_tiles = (self.tiles_axis * self.tiles_axis) as usize;
        let (loader_pending, _, _) = self.loader.stats();
        let (_, _, _, _, _, loader_completed) = self.loader.counters();
        let group_report = owner_group_report(&self.allocation_owner, "orbis.height");
        HeightStreamingStats {
            center: self.streamer.center(),
            pending_ring_tiles: self.streamer.pending_count(),
            loaded_ring_tiles: self.streamer.loaded_count(),
            resident_fine_tiles: self.resident_fine.len(),
            total_tiles,
            tiles_requested: self.tiles_requested,
            tiles_uploaded: self.tiles_uploaded,
            coarse_prefilled: self.coarse_prefilled,
            // Logical residency fluctuates with LRU eviction. The physical
            // allocation envelope is reported separately below.
            resident_height_bytes: self.resident_fine.len() as u64
                * u64::from(self.tile_resolution)
                * u64::from(self.tile_resolution)
                * 5,
            gpu_visible_current_bytes: group_report.current_total_bytes(),
            gpu_visible_high_water_bytes: group_report.peak_total_bytes,
            ancestor_fallbacks: self.ancestor_fallbacks,
            converged: loader_pending == 0 && self.ring_waiting.is_empty(),
            loader_pending,
            loader_completed,
        }
    }

    pub(in crate::terrain::renderer) fn is_globe(&self) -> bool {
        self.globe_mode
    }
}

impl Drop for HeightVtFamilyRuntime {
    fn drop(&mut self) {
        super::virtual_texture::clear_height_family_stats(self.residency_owner_id);
    }
}

impl TerrainScene {
    /// Clipmap mesh center: follows the streaming center when height
    /// streaming is active, otherwise stays at the region origin.
    pub(in crate::terrain::renderer) fn height_streaming_center(&self) -> Vec2 {
        self.height_streaming
            .as_ref()
            .map(|s| s.streamer.center())
            .unwrap_or(Vec2::ZERO)
    }

    /// The caller-provided overview stays bound in every render path. Dynamic
    /// pages use the separate atlas binding, so a regional COG miss can fall
    /// back without pretending that it owns the whole planet.
    pub(in crate::terrain::renderer) fn main_pass_height_view<'a>(
        &'a self,
        uploaded: &'a wgpu::TextureView,
    ) -> &'a wgpu::TextureView {
        uploaded
    }

    pub(in crate::terrain::renderer) fn main_pass_height_atlas_view<'a>(
        &'a self,
        uploaded: &'a wgpu::TextureView,
    ) -> &'a wgpu::TextureView {
        self.height_streaming
            .as_ref()
            .filter(|runtime| runtime.mosaic.resident_tile_count() > 0)
            .map(|runtime| &runtime.mosaic.view)
            .unwrap_or(uploaded)
    }

    pub(in crate::terrain::renderer) fn main_pass_height_coverage_view<'a>(
        &'a self,
        uploaded: &'a wgpu::TextureView,
    ) -> &'a wgpu::TextureView {
        self.height_streaming
            .as_ref()
            .filter(|runtime| runtime.mosaic.resident_tile_count() > 0)
            .map(|runtime| &runtime.mosaic.coverage_view)
            .unwrap_or(uploaded)
    }

    /// Sparse dynamic-height indirection shared by beauty, AOV and offline
    /// terrain passes. Until the asynchronous root is resident, the disabled
    /// fallback preserves the caller's ordinary overview rendering exactly.
    pub(in crate::terrain::renderer) fn main_pass_height_page_table(&self) -> &wgpu::Buffer {
        self.height_streaming
            .as_ref()
            .filter(|runtime| runtime.mosaic.resident_tile_count() > 0)
            .map(|runtime| runtime.page_table.buffer.inner())
            .unwrap_or(self.height_page_table_fallback_buffer.inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn height_upload_capacity_admits_configured_max_batch_exactly() {
        let resolution = 65u64;
        let max_in_flight = 8usize;
        let height_bytes = (resolution * 4).div_ceil(256) * 256 * resolution;
        let coverage_bytes = resolution.div_ceil(256) * 256 * resolution;
        let page_bytes = PageTable::allocation_bytes_for_capacity(max_in_flight + 1);
        let shapes = vec![(height_bytes, coverage_bytes); max_in_flight];
        let layout = HeightUploadRing::plan_layout(&shapes, page_bytes).unwrap();
        let reserved = HeightUploadRing::required_capacity(
            max_in_flight,
            height_bytes,
            coverage_bytes,
            page_bytes,
        )
        .unwrap();
        assert_eq!(reserved, layout.required_bytes);
        assert!(layout.required_bytes <= reserved);
    }

    #[test]
    fn height_upload_rejects_oversize_in_flight_before_layout_allocation() {
        assert!(validate_max_in_flight(0).is_err());
        assert!(validate_max_in_flight(ORBIS_MAX_IN_FLIGHT + 1).is_err());
        assert_eq!(validate_max_in_flight(ORBIS_MAX_IN_FLIGHT).unwrap(), 1024);
    }

    #[test]
    fn loader_worker_count_is_bounded_before_thread_construction() {
        assert_eq!(validate_pool_size(0).unwrap(), 1);
        assert_eq!(validate_pool_size(MAX_LOADER_WORKERS).unwrap(), 16);
        assert!(validate_pool_size(MAX_LOADER_WORKERS + 1).is_err());
    }

    fn prepared(id: TileId, sx: u32, sy: u32) -> PreparedHeightUpload {
        PreparedHeightUpload {
            id,
            sx,
            sy,
            bytes: Vec::new(),
            coverage_bytes: Vec::new(),
            padded_bytes_per_row: 0,
            coverage_padded_bytes_per_row: 0,
            rows: 0,
        }
    }

    #[test]
    fn capacity_limited_batch_never_marks_a_displaced_preparation_resident() {
        let first = TileId::new(2, 0, 0);
        let second = TileId::new(2, 1, 0);
        // Both completions prepared the only physical slot in sequence. The
        // final atomic mapping belongs solely to the second completion.
        let final_mapping = HashMap::from([(second, (0, 0))]);
        let (retained, displaced) = validate_prepared_final_mappings(
            vec![prepared(first, 0, 0), prepared(second, 0, 0)],
            |id| final_mapping.get(&id).copied(),
        );
        assert_eq!(retained.iter().map(|item| item.id).collect::<Vec<_>>(), vec![second]);
        assert_eq!(displaced, vec![first]);
    }

    #[test]
    fn dem_slice_reader_reproduces_linear_ramp() {
        // A DEM that is linear in x should bilinear-sample exactly.
        let (w, h) = (9usize, 5usize);
        let dem: Vec<f32> = (0..h)
            .flat_map(|_| (0..w).map(|x| x as f32 / (w - 1) as f32))
            .collect();
        let reader = DemSliceHeightReader::new(dem, w, h);
        let root = TileBounds::new(Vec2::new(-50.0, -50.0), Vec2::new(50.0, 50.0));
        let tile_size = Vec2::splat(100.0);

        // LOD 0 tile spans the whole region: corners must match DEM corners.
        let full = reader
            .read_result(&root, tile_size, TileId::new(0, 0, 0), 9, 5)
            .unwrap();
        assert!((full[0] - 0.0).abs() < 1e-6);
        assert!((full[8] - 1.0).abs() < 1e-6);

        // LOD 1 tile (1, 0) covers the right half in x: values in [0.5, 1.0].
        let right = reader
            .read_result(&root, tile_size, TileId::new(1, 1, 0), 5, 3)
            .unwrap();
        for v in &right {
            assert!(
                *v >= 0.5 - 1e-6 && *v <= 1.0 + 1e-6,
                "value {} out of right-half range",
                v
            );
        }
    }

    #[test]
    fn upsample_bilinear_preserves_corners_and_range() {
        let src = vec![0.0f32, 1.0, 2.0, 3.0];
        let up = upsample_bilinear(&src, 2, 5);
        assert_eq!(up.len(), 25);
        assert!((up[0] - 0.0).abs() < 1e-6);
        assert!((up[4] - 1.0).abs() < 1e-6);
        assert!((up[20] - 2.0).abs() < 1e-6);
        assert!((up[24] - 3.0).abs() < 1e-6);
        for v in &up {
            assert!(*v >= 0.0 && *v <= 3.0);
        }
    }

    #[test]
    fn map_tile_to_fixed_lod_covers_all_cases() {
        let mut out = HashSet::new();

        // Same LOD passes through.
        map_tile_to_fixed_lod(TileId::new(2, 1, 3), 2, 4, &mut out);
        assert_eq!(out.len(), 1);
        assert!(out.contains(&TileId::new(2, 1, 3)));

        // Finer collapses to ancestor.
        out.clear();
        map_tile_to_fixed_lod(TileId::new(4, 13, 6), 2, 4, &mut out);
        assert_eq!(out.len(), 1);
        assert!(out.contains(&TileId::new(2, 3, 1)));

        // Coarser expands to all covered descendants.
        out.clear();
        map_tile_to_fixed_lod(TileId::new(0, 0, 0), 2, 4, &mut out);
        assert_eq!(out.len(), 16);

        // Out-of-axis coordinates are dropped.
        out.clear();
        map_tile_to_fixed_lod(TileId::new(2, 9, 0), 2, 4, &mut out);
        assert!(out.is_empty());
    }
}
