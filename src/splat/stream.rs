// src/splat/stream.rs
// Out-of-core paging for the fused scene. Primitives live in *pages* (a chunk
// of splat SoA records or one COPC octree node's points) that are streamed
// from disk on demand into a fixed GPU residency pool with LRU eviction.
// This module owns the page abstraction (`PageSource`), the on-disk page
// store, the in-memory/COPC sources, the GPU-ready decode (records + a small
// per-page BVH), the residency bookkeeping and the asynchronous loader.
// Billions of primitives are described by a compact index; only the pages a
// traversal actually reaches ever become resident.
// RELEVANT FILES: src/splat/fusion.rs, src/splat/bvh.rs,
//                 src/shaders/fusion/unified_occlusion.wgsl

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use glam::DVec3;

use super::bvh::{build_bvh, Aabb, FusionBvhNode, BLAS_LEAF_SIZE};
use super::load::{SplatChunk, SplatPlyReader};
use super::num::f32_from_u32;
use super::{inverse_covariance, sh_rest_count, sigma_extent, GaussianSplatCloud};
use crate::core::error::RenderError;
use crate::core::resource_tracker::{tracked_host_allocation, AllocationOwner, ResourceHandle};
use crate::pointcloud::{CopcDataset, OctreeKey};

/// Page-table value for a page that is not in the residency pool.
pub const NOT_RESIDENT: u32 = u32::MAX;

/// What a page holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PageKind {
    Splat = 0,
    Points = 1,
}

/// GPU record of one splat: position + opacity, DC colour + shading SH
/// degree, and the three first-band coefficients.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct SplatGpu {
    pub pos_opacity: [f32; 4],
    pub sh0_degree: [f32; 4],
    pub sh1: [[f32; 4]; 3],
}

/// GPU record of one splat's packed inverse covariance:
/// `m0 = (xx, xy, xz, yy)`, `m1 = (yz, zz, 0, 0)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct InvCovGpu {
    pub m0: [f32; 4],
    pub m1: [f32; 4],
}

/// GPU record of one LiDAR return: position + sphelet radius, linear colour
/// and the octahedral surfel normal (`SURFEL_SPHERE` = isotropic sphelet).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct PointGpu {
    pub pos_radius: [f32; 4],
    pub color: [f32; 3],
    pub normal_oct: u32,
}

/// Highest SH band the GPU shading path evaluates. Higher bands stay on the
/// CPU cloud (`GaussianSplatCloud::color`); this is reported, not hidden.
pub const GPU_SH_DEGREE: u32 = 1;

/// Index-level description of a page.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageMeta {
    pub kind: PageKind,
    pub count: u32,
    /// World bounds. Splat pages include the 3-sigma proxies; point pages are
    /// the raw point bounds (the fused scene expands them by the sphelet
    /// radius).
    pub aabb: Aabb,
}

/// Decoded attributes of one page in world coordinates.
#[derive(Clone, Debug)]
pub enum PagePayload {
    Splats(SplatChunk),
    Points {
        positions: Vec<[f32; 3]>,
        /// Linear RGB.
        colors: Vec<[f32; 3]>,
    },
}

impl PagePayload {
    pub fn len(&self) -> usize {
        match self {
            Self::Splats(chunk) => chunk.len(),
            Self::Points { positions, .. } => positions.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A set of pages that can be loaded independently (and concurrently).
pub trait PageSource: Send + Sync {
    fn kind(&self) -> PageKind;
    fn page_count(&self) -> u32;
    fn meta(&self, page: u32) -> PageMeta;
    /// Blocking read + decode of one page.
    fn load(&self, page: u32) -> Result<PagePayload, RenderError>;
    /// Primitives this source describes (the logical scene size).
    fn logical_primitives(&self) -> u64;
    fn label(&self) -> String;
}

// ---------------------------------------------------------------------------
// Morton ordering
// ---------------------------------------------------------------------------

fn spread_bits_21(value: u64) -> u64 {
    let mut x = value & 0x1f_ffff;
    x = (x | (x << 32)) & 0x001f_0000_0000_ffff;
    x = (x | (x << 16)) & 0x001f_0000_ff00_00ff;
    x = (x | (x << 8)) & 0x100f_00f0_0f00_f00f;
    x = (x | (x << 4)) & 0x10c3_0c30_c30c_30c3;
    x = (x | (x << 2)) & 0x1249_2492_4924_9249;
    x
}

/// 63-bit Morton code of `p` quantised to 21 bits per axis inside `bounds`.
pub fn morton_code(p: [f32; 3], bounds: Aabb) -> u64 {
    let mut code = 0u64;
    for axis in 0..3 {
        let extent = (bounds.max[axis] - bounds.min[axis]).max(1e-20);
        let unit = ((p[axis] - bounds.min[axis]) / extent).clamp(0.0, 1.0);
        let cell = (f64::from(unit) * 2_097_151.0) as u64;
        code |= spread_bits_21(cell) << axis;
    }
    code
}

/// Split `positions` into spatially compact pages of at most `capacity`
/// primitives. A k-d style median split along the widest axis cuts at whole
/// multiples of the capacity, so every page except possibly one is full and
/// page boxes stay tight (a Morton run can straddle distant octants).
pub fn paginate(positions: &[[f32; 3]], capacity: usize) -> Vec<Vec<u32>> {
    fn split(
        indices: &mut [u32],
        positions: &[[f32; 3]],
        capacity: usize,
        out: &mut Vec<Vec<u32>>,
    ) {
        if indices.len() <= capacity {
            out.push(indices.to_vec());
            return;
        }
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for &i in indices.iter() {
            let p = positions[i as usize];
            for axis in 0..3 {
                lo[axis] = lo[axis].min(p[axis]);
                hi[axis] = hi[axis].max(p[axis]);
            }
        }
        let axis = (0..3)
            .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
            .unwrap();
        let left = (indices.len().div_ceil(capacity) / 2) * capacity;
        indices.select_nth_unstable_by(left, |&a, &b| {
            positions[a as usize][axis]
                .total_cmp(&positions[b as usize][axis])
                .then(a.cmp(&b))
        });
        let (head, tail) = indices.split_at_mut(left);
        split(head, positions, capacity, out);
        split(tail, positions, capacity, out);
    }
    let mut indices: Vec<u32> = (0..positions.len() as u32).collect();
    let mut pages = Vec::new();
    if !indices.is_empty() {
        split(&mut indices, positions, capacity.max(1), &mut pages);
    }
    pages
}

fn splat_page_aabb(chunk: &SplatChunk) -> Aabb {
    let mut aabb = Aabb::EMPTY;
    for i in 0..chunk.len() {
        let half = sigma_extent(chunk.scales[i], chunk.rotations[i], super::kernel::SIGMA_CUTOFF);
        let p = chunk.positions[i];
        aabb = aabb.union(Aabb::new(
            [p[0] - half[0], p[1] - half[1], p[2] - half[2]],
            [p[0] + half[0], p[1] + half[1], p[2] + half[2]],
        ));
    }
    aabb
}

// ---------------------------------------------------------------------------
// In-memory splat cloud as a page source
// ---------------------------------------------------------------------------

/// Pages over a resident `GaussianSplatCloud` (small clouds loaded through
/// `load_gaussian_splats`). The cloud itself is tracked host memory; only the
/// GPU residency is paged.
pub struct SplatCloudSource {
    cloud: Arc<GaussianSplatCloud>,
    pages: Vec<Vec<u32>>,
    metas: Vec<PageMeta>,
}

impl SplatCloudSource {
    pub fn new(cloud: Arc<GaussianSplatCloud>, page_capacity: usize) -> Self {
        let pages = paginate(&cloud.positions, page_capacity);
        let metas = pages
            .iter()
            .map(|page| {
                let aabb = page.iter().fold(Aabb::EMPTY, |acc, &i| {
                    let (lo, hi) = cloud.splat_aabb(i as usize);
                    acc.union(Aabb::new(lo, hi))
                });
                PageMeta {
                    kind: PageKind::Splat,
                    count: page.len() as u32,
                    aabb,
                }
            })
            .collect();
        Self {
            cloud,
            pages,
            metas,
        }
    }
}

impl PageSource for SplatCloudSource {
    fn kind(&self) -> PageKind {
        PageKind::Splat
    }

    fn page_count(&self) -> u32 {
        self.pages.len() as u32
    }

    fn meta(&self, page: u32) -> PageMeta {
        self.metas[page as usize]
    }

    fn load(&self, page: u32) -> Result<PagePayload, RenderError> {
        let indices = &self.pages[page as usize];
        let cloud = &self.cloud;
        let per_rest = cloud.sh_rest.as_ref().map_or(0, |rest| rest.per_splat());
        let mut chunk = SplatChunk::default();
        for &i in indices {
            let i = i as usize;
            chunk.positions.push(cloud.positions[i]);
            chunk.scales.push(cloud.scales[i]);
            chunk.rotations.push(cloud.rotations[i]);
            chunk.opacities.push(cloud.opacities[i]);
            chunk.sh0.push(cloud.sh0[i]);
            if let Some(rest) = &cloud.sh_rest {
                chunk
                    .sh_rest
                    .extend_from_slice(&rest.coeffs[i * per_rest..(i + 1) * per_rest]);
            }
        }
        Ok(PagePayload::Splats(chunk))
    }

    fn logical_primitives(&self) -> u64 {
        self.cloud.len() as u64
    }

    fn label(&self) -> String {
        format!("splat-cloud[{} splats]", self.cloud.len())
    }
}

// ---------------------------------------------------------------------------
// On-disk page store
// ---------------------------------------------------------------------------

const STORE_MAGIC: &[u8; 8] = b"F3DPGST1";
const STORE_VERSION: u32 = 1;
const STORE_HEADER_BYTES: u64 = 96;
const INDEX_ENTRY_BYTES: usize = 56;
const POINT_RECORD_BYTES: usize = 16;

/// One index entry of a page store. Several entries may reference the same
/// payload byte range with different origins: payload positions are stored
/// relative to the page origin.
#[derive(Clone, Copy, Debug, PartialEq)]
struct IndexEntry {
    offset: u64,
    byte_len: u32,
    count: u32,
    origin: [f32; 3],
    aabb: Aabb,
}

impl IndexEntry {
    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.offset.to_le_bytes());
        out.extend_from_slice(&self.byte_len.to_le_bytes());
        out.extend_from_slice(&self.count.to_le_bytes());
        for value in self.origin.iter().chain(&self.aabb.min).chain(&self.aabb.max) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes());
    }

    fn read(bytes: &[u8]) -> Self {
        let f = |at: usize| f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        Self {
            offset: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            byte_len: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            count: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            origin: [f(16), f(20), f(24)],
            aabb: Aabb::new([f(28), f(32), f(36)], [f(40), f(44), f(48)]),
        }
    }
}

fn splat_record_floats(sh_degree: u32) -> usize {
    14 + 3 * sh_rest_count(sh_degree)
}

/// Totals reported when a page store is finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageStoreSummary {
    pub page_count: u64,
    pub logical_primitives: u64,
    pub payload_bytes: u64,
    pub file_bytes: u64,
}

/// Sequential writer for a page store.
pub struct PageStoreWriter {
    path: PathBuf,
    out: BufWriter<File>,
    kind: PageKind,
    page_capacity: u32,
    sh_degree: u32,
    entries: Vec<IndexEntry>,
    cursor: u64,
    logical: u64,
    bounds: Aabb,
}

impl PageStoreWriter {
    pub fn create(
        path: impl AsRef<Path>,
        kind: PageKind,
        page_capacity: u32,
        sh_degree: u32,
    ) -> Result<Self, RenderError> {
        if page_capacity == 0 {
            return Err(RenderError::Upload(
                "page store capacity must be at least one primitive".into(),
            ));
        }
        if sh_degree > 3 {
            return Err(RenderError::Upload(format!(
                "page store SH degree must be 0..=3, got {sh_degree}"
            )));
        }
        let path = path.as_ref().to_path_buf();
        let mut out = BufWriter::with_capacity(1 << 20, File::create(&path)?);
        out.write_all(&[0u8; STORE_HEADER_BYTES as usize])?;
        Ok(Self {
            path,
            out,
            kind,
            page_capacity,
            sh_degree,
            entries: Vec::new(),
            cursor: STORE_HEADER_BYTES,
            logical: 0,
            bounds: Aabb::EMPTY,
        })
    }

    fn push_entry(&mut self, entry: IndexEntry) -> u32 {
        self.logical += u64::from(entry.count);
        self.bounds = self.bounds.union(entry.aabb);
        self.entries.push(entry);
        (self.entries.len() - 1) as u32
    }

    fn check_count(&self, count: usize) -> Result<(), RenderError> {
        if count == 0 || count > self.page_capacity as usize {
            return Err(RenderError::Upload(format!(
                "page of {count} primitives violates the store capacity 1..={}",
                self.page_capacity
            )));
        }
        Ok(())
    }

    /// Append a page of splats given in world coordinates. `chunk.sh_rest`
    /// must carry `sh_rest_count(sh_degree)` coefficients per splat.
    pub fn push_splat_page(&mut self, chunk: &SplatChunk) -> Result<u32, RenderError> {
        if self.kind != PageKind::Splat {
            return Err(RenderError::Upload(
                "cannot push splats into a point page store".into(),
            ));
        }
        self.check_count(chunk.len())?;
        let per_rest = sh_rest_count(self.sh_degree);
        if chunk.sh_rest.len() != chunk.len() * per_rest {
            return Err(RenderError::Upload(format!(
                "splat page carries {} SH coefficients, expected {} for degree {}",
                chunk.sh_rest.len(),
                chunk.len() * per_rest,
                self.sh_degree
            )));
        }
        let aabb = splat_page_aabb(chunk);
        let origin = aabb.min;
        let mut bytes = Vec::with_capacity(chunk.len() * splat_record_floats(self.sh_degree) * 4);
        let mut put = |value: f32| bytes.extend_from_slice(&value.to_le_bytes());
        for i in 0..chunk.len() {
            for axis in 0..3 {
                put(chunk.positions[i][axis] - origin[axis]);
            }
            chunk.scales[i].into_iter().for_each(&mut put);
            chunk.rotations[i].into_iter().for_each(&mut put);
            put(chunk.opacities[i]);
            chunk.sh0[i].into_iter().for_each(&mut put);
            for coeff in &chunk.sh_rest[i * per_rest..(i + 1) * per_rest] {
                coeff.iter().copied().for_each(&mut put);
            }
        }
        self.out.write_all(&bytes)?;
        let entry = IndexEntry {
            offset: self.cursor,
            byte_len: bytes.len() as u32,
            count: chunk.len() as u32,
            origin,
            aabb,
        };
        self.cursor += bytes.len() as u64;
        Ok(self.push_entry(entry))
    }

    /// Append a page of points (world positions, sRGB bytes + classification).
    pub fn push_point_page(
        &mut self,
        positions: &[[f32; 3]],
        rgba: &[[u8; 4]],
    ) -> Result<u32, RenderError> {
        if self.kind != PageKind::Points {
            return Err(RenderError::Upload(
                "cannot push points into a splat page store".into(),
            ));
        }
        self.check_count(positions.len())?;
        if rgba.len() != positions.len() {
            return Err(RenderError::Upload(
                "point page colour count does not match its positions".into(),
            ));
        }
        let aabb = positions
            .iter()
            .fold(Aabb::EMPTY, |acc, p| acc.union(Aabb::new(*p, *p)));
        let origin = aabb.min;
        let mut bytes = Vec::with_capacity(positions.len() * POINT_RECORD_BYTES);
        for (p, c) in positions.iter().zip(rgba) {
            for axis in 0..3 {
                bytes.extend_from_slice(&(p[axis] - origin[axis]).to_le_bytes());
            }
            bytes.extend_from_slice(c);
        }
        self.out.write_all(&bytes)?;
        let entry = IndexEntry {
            offset: self.cursor,
            byte_len: bytes.len() as u32,
            count: positions.len() as u32,
            origin,
            aabb,
        };
        self.cursor += bytes.len() as u64;
        Ok(self.push_entry(entry))
    }

    /// Append an index entry that reuses the payload of `source_page` at a
    /// different origin (instanced tiles; also how a synthetic billion-
    /// primitive index is laid over a handful of on-disk payloads).
    pub fn push_alias(&mut self, source_page: u32, origin: [f32; 3]) -> Result<u32, RenderError> {
        let source = *self.entries.get(source_page as usize).ok_or_else(|| {
            RenderError::Upload(format!("alias source page {source_page} does not exist"))
        })?;
        let mut entry = source;
        for axis in 0..3 {
            let shift = origin[axis] - source.origin[axis];
            entry.aabb.min[axis] += shift;
            entry.aabb.max[axis] += shift;
        }
        entry.origin = origin;
        Ok(self.push_entry(entry))
    }

    pub fn page_count(&self) -> u32 {
        self.entries.len() as u32
    }

    /// Write the index and header and close the file.
    pub fn finish(mut self) -> Result<PageStoreSummary, RenderError> {
        let index_offset = self.cursor;
        let mut index = Vec::with_capacity(self.entries.len() * INDEX_ENTRY_BYTES);
        for entry in &self.entries {
            entry.write(&mut index);
        }
        self.out.write_all(&index)?;
        self.out.flush()?;
        let mut file = self.out.into_inner().map_err(|e| {
            RenderError::Upload(format!(
                "page store {}: flush failed: {e}",
                self.path.display()
            ))
        })?;
        let mut header = Vec::with_capacity(STORE_HEADER_BYTES as usize);
        header.extend_from_slice(STORE_MAGIC);
        header.extend_from_slice(&STORE_VERSION.to_le_bytes());
        header.extend_from_slice(&(self.kind as u32).to_le_bytes());
        header.extend_from_slice(&self.page_capacity.to_le_bytes());
        header.extend_from_slice(&self.sh_degree.to_le_bytes());
        header.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        header.extend_from_slice(&self.logical.to_le_bytes());
        header.extend_from_slice(&index_offset.to_le_bytes());
        header.extend_from_slice(&STORE_HEADER_BYTES.to_le_bytes());
        for value in self.bounds.min.iter().chain(&self.bounds.max) {
            header.extend_from_slice(&value.to_le_bytes());
        }
        header.resize(STORE_HEADER_BYTES as usize, 0);
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header)?;
        file.sync_all()?;
        let file_bytes = index_offset + index.len() as u64;
        Ok(PageStoreSummary {
            page_count: self.entries.len() as u64,
            logical_primitives: self.logical,
            payload_bytes: index_offset - STORE_HEADER_BYTES,
            file_bytes,
        })
    }
}

/// Read side of a page store: the index is resident (tracked host memory),
/// payloads are read from disk one page at a time.
pub struct PageStoreFile {
    path: PathBuf,
    kind: PageKind,
    page_capacity: u32,
    sh_degree: u32,
    logical: u64,
    entries: Vec<IndexEntry>,
    file: Mutex<File>,
    _index_tracker: ResourceHandle,
}

impl PageStoreFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RenderError> {
        let path = path.as_ref().to_path_buf();
        let describe = |message: String| {
            RenderError::Upload(format!("page store {}: {message}", path.display()))
        };
        let mut file = File::open(&path)
            .map_err(|e| describe(format!("cannot open: {e}")))?;
        let file_len = file.metadata()?.len();
        let mut header = [0u8; STORE_HEADER_BYTES as usize];
        file.read_exact(&mut header)
            .map_err(|e| describe(format!("truncated header: {e}")))?;
        if &header[0..8] != STORE_MAGIC {
            return Err(describe("bad magic (not a forge3d page store)".into()));
        }
        let u32_at = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().unwrap());
        let u64_at = |at: usize| u64::from_le_bytes(header[at..at + 8].try_into().unwrap());
        if u32_at(8) != STORE_VERSION {
            return Err(describe(format!("unsupported version {}", u32_at(8))));
        }
        let kind = match u32_at(12) {
            0 => PageKind::Splat,
            1 => PageKind::Points,
            other => return Err(describe(format!("unknown page kind {other}"))),
        };
        let page_capacity = u32_at(16);
        let sh_degree = u32_at(20);
        let page_count = u64_at(24);
        let logical = u64_at(32);
        let index_offset = u64_at(40);
        let index_bytes = page_count
            .checked_mul(INDEX_ENTRY_BYTES as u64)
            .filter(|bytes| index_offset.checked_add(*bytes) == Some(file_len))
            .ok_or_else(|| describe("index does not match the file length".into()))?;
        if page_count > u64::from(u32::MAX) || sh_degree > 3 || page_capacity == 0 {
            return Err(describe("header fields are out of range".into()));
        }
        let tracker = tracked_host_allocation(index_bytes, "fusion-page-store-index")?;
        let mut raw = vec![0u8; index_bytes as usize];
        file.seek(SeekFrom::Start(index_offset))?;
        file.read_exact(&mut raw)
            .map_err(|e| describe(format!("truncated index: {e}")))?;
        let record_bytes = match kind {
            PageKind::Splat => splat_record_floats(sh_degree) * 4,
            PageKind::Points => POINT_RECORD_BYTES,
        };
        let mut entries = Vec::with_capacity(page_count as usize);
        let mut total = 0u64;
        for chunk in raw.chunks_exact(INDEX_ENTRY_BYTES) {
            let entry = IndexEntry::read(chunk);
            let end = entry.offset.checked_add(u64::from(entry.byte_len));
            if entry.count == 0
                || entry.count > page_capacity
                || entry.byte_len as usize != entry.count as usize * record_bytes
                || entry.offset < STORE_HEADER_BYTES
                || end.is_none_or(|end| end > index_offset)
                || !entry.aabb.is_valid()
            {
                return Err(describe(format!(
                    "index entry {} is inconsistent with the store layout",
                    entries.len()
                )));
            }
            total += u64::from(entry.count);
            entries.push(entry);
        }
        if total != logical {
            return Err(describe(format!(
                "index describes {total} primitives but the header claims {logical}"
            )));
        }
        Ok(Self {
            path,
            kind,
            page_capacity,
            sh_degree,
            logical,
            entries,
            file: Mutex::new(file),
            _index_tracker: tracker,
        })
    }

    pub fn page_capacity(&self) -> u32 {
        self.page_capacity
    }

    pub fn sh_degree(&self) -> u32 {
        self.sh_degree
    }
}

impl PageSource for PageStoreFile {
    fn kind(&self) -> PageKind {
        self.kind
    }

    fn page_count(&self) -> u32 {
        self.entries.len() as u32
    }

    fn meta(&self, page: u32) -> PageMeta {
        let entry = &self.entries[page as usize];
        PageMeta {
            kind: self.kind,
            count: entry.count,
            aabb: entry.aabb,
        }
    }

    fn load(&self, page: u32) -> Result<PagePayload, RenderError> {
        let entry = self.entries[page as usize];
        let mut bytes = vec![0u8; entry.byte_len as usize];
        {
            let mut file = self
                .file
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            file.seek(SeekFrom::Start(entry.offset))?;
            file.read_exact(&mut bytes).map_err(|e| {
                RenderError::Upload(format!(
                    "page store {}: page {page} read failed: {e}",
                    self.path.display()
                ))
            })?;
        }
        let n = entry.count as usize;
        let origin = entry.origin;
        match self.kind {
            PageKind::Splat => {
                let per_rest = sh_rest_count(self.sh_degree);
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let stride = splat_record_floats(self.sh_degree);
                let mut chunk = SplatChunk::default();
                for record in floats.chunks_exact(stride) {
                    chunk.positions.push([
                        record[0] + origin[0],
                        record[1] + origin[1],
                        record[2] + origin[2],
                    ]);
                    chunk.scales.push([record[3], record[4], record[5]]);
                    chunk
                        .rotations
                        .push([record[6], record[7], record[8], record[9]]);
                    chunk.opacities.push(record[10]);
                    chunk.sh0.push([record[11], record[12], record[13]]);
                    for k in 0..per_rest {
                        let at = 14 + 3 * k;
                        chunk
                            .sh_rest
                            .push([record[at], record[at + 1], record[at + 2]]);
                    }
                }
                if chunk.len() != n {
                    return Err(RenderError::Upload(format!(
                        "page store {}: page {page} decoded {} splats, expected {n}",
                        self.path.display(),
                        chunk.len()
                    )));
                }
                Ok(PagePayload::Splats(chunk))
            }
            PageKind::Points => {
                let mut positions = Vec::with_capacity(n);
                let mut colors = Vec::with_capacity(n);
                for record in bytes.chunks_exact(POINT_RECORD_BYTES) {
                    let f = |at: usize| f32::from_le_bytes(record[at..at + 4].try_into().unwrap());
                    positions.push([f(0) + origin[0], f(4) + origin[1], f(8) + origin[2]]);
                    colors.push([
                        srgb_byte_to_linear(record[12]),
                        srgb_byte_to_linear(record[13]),
                        srgb_byte_to_linear(record[14]),
                    ]);
                }
                Ok(PagePayload::Points { positions, colors })
            }
        }
    }

    fn logical_primitives(&self) -> u64 {
        self.logical
    }

    fn label(&self) -> String {
        format!(
            "page-store[{} pages, {} primitives] {}",
            self.entries.len(),
            self.logical,
            self.path.display()
        )
    }
}

/// sRGB byte -> linear float.
pub fn srgb_byte_to_linear(value: u8) -> f32 {
    let c = f32::from(value) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Write a resident cloud as a page store (Morton-ordered pages).
pub fn write_splat_page_store(
    path: impl AsRef<Path>,
    cloud: &GaussianSplatCloud,
    page_capacity: u32,
) -> Result<PageStoreSummary, RenderError> {
    let degree = cloud.sh_degree();
    let per_rest = sh_rest_count(degree);
    let mut writer = PageStoreWriter::create(path, PageKind::Splat, page_capacity, degree)?;
    for page in paginate(&cloud.positions, page_capacity as usize) {
        let mut chunk = SplatChunk::default();
        for i in page {
            let i = i as usize;
            chunk.positions.push(cloud.positions[i]);
            chunk.scales.push(cloud.scales[i]);
            chunk.rotations.push(cloud.rotations[i]);
            chunk.opacities.push(cloud.opacities[i]);
            chunk.sh0.push(cloud.sh0[i]);
            if let Some(rest) = &cloud.sh_rest {
                chunk
                    .sh_rest
                    .extend_from_slice(&rest.coeffs[i * per_rest..(i + 1) * per_rest]);
            }
        }
        writer.push_splat_page(&chunk)?;
    }
    writer.finish()
}

/// Build a splat page store from a 3DGS `.ply` without ever holding the file
/// in memory: an external bucketed Morton sort.
///
/// Pass 1 streams the file for its bounds and a coarse Morton histogram.
/// Pass 2 streams it again, appending each record to the temp bucket that
/// owns its coarse cell (buckets are contiguous Morton ranges sized to
/// `memory_budget_bytes`). Pass 3 paginates one bucket at a time in memory
/// and emits pages. A single coarse cell that alone exceeds the budget is a
/// diagnostic error, never a silent overrun.
pub fn build_splat_page_store_from_ply(
    ply: impl AsRef<Path>,
    out: impl AsRef<Path>,
    page_capacity: u32,
    memory_budget_bytes: u64,
) -> Result<PageStoreSummary, RenderError> {
    const COARSE_BITS: u32 = 5; // 32^3 cells
    const COARSE_CELLS: usize = 1 << (3 * COARSE_BITS);
    const CHUNK: usize = 1 << 16;
    let ply = ply.as_ref();
    let out = out.as_ref();

    let mut reader = SplatPlyReader::open(ply)?;
    let degree = reader.header().sh_degree;
    let per_rest = sh_rest_count(degree);
    let floats = splat_record_floats(degree);
    let record_bytes = (floats * 4) as u64;
    let bucket_capacity = (memory_budget_bytes / (record_bytes * 3)).max(1);

    // Pass 1: bounds.
    let mut bounds = Aabb::EMPTY;
    let mut total = 0u64;
    loop {
        let chunk = reader.read_chunk(CHUNK)?;
        if chunk.is_empty() {
            break;
        }
        total += chunk.len() as u64;
        for p in &chunk.positions {
            bounds = bounds.union(Aabb::new(*p, *p));
        }
    }
    if total == 0 {
        return Err(RenderError::Upload(format!(
            "splat PLY {} contains no splats",
            ply.display()
        )));
    }
    let coarse = |p: [f32; 3]| (morton_code(p, bounds) >> (3 * (21 - COARSE_BITS))) as usize;

    // Pass 1b: coarse histogram -> contiguous Morton buckets.
    let mut histogram = vec![0u64; COARSE_CELLS];
    let mut reader = SplatPlyReader::open(ply)?;
    loop {
        let chunk = reader.read_chunk(CHUNK)?;
        if chunk.is_empty() {
            break;
        }
        for p in &chunk.positions {
            histogram[coarse(*p)] += 1;
        }
    }
    let mut bucket_of = vec![0u32; COARSE_CELLS];
    let mut bucket_sizes = vec![0u64];
    for (cell, count) in histogram.iter().enumerate() {
        if *count > bucket_capacity {
            return Err(RenderError::Budget(format!(
                "splat PLY {}: one coarse cell holds {count} splats, more than the {} that fit \
                 the {memory_budget_bytes}-byte build budget; raise the budget",
                ply.display(),
                bucket_capacity
            )));
        }
        let last = bucket_sizes.len() - 1;
        if bucket_sizes[last] + count > bucket_capacity && bucket_sizes[last] > 0 {
            bucket_sizes.push(0);
        }
        let last = bucket_sizes.len() - 1;
        bucket_sizes[last] += count;
        bucket_of[cell] = last as u32;
    }
    if bucket_sizes.len() > 2048 {
        return Err(RenderError::Budget(format!(
            "splat PLY {}: {} sort buckets exceed the 2048 open-file ceiling; raise the \
             build budget",
            ply.display(),
            bucket_sizes.len()
        )));
    }

    // Pass 2: distribute records to bucket files.
    let stem = out
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "pages".into());
    let bucket_path = |index: usize| out.with_file_name(format!("{stem}.bucket{index}.tmp"));
    let mut buckets = (0..bucket_sizes.len())
        .map(|index| File::create(bucket_path(index)).map(BufWriter::new))
        .collect::<Result<Vec<_>, _>>()?;
    let mut reader = SplatPlyReader::open(ply)?;
    loop {
        let chunk = reader.read_chunk(CHUNK)?;
        if chunk.is_empty() {
            break;
        }
        for i in 0..chunk.len() {
            let file = &mut buckets[bucket_of[coarse(chunk.positions[i])] as usize];
            let mut put = |value: f32| file.write_all(&value.to_le_bytes());
            for value in chunk.positions[i]
                .into_iter()
                .chain(chunk.scales[i])
                .chain(chunk.rotations[i])
                .chain([chunk.opacities[i]])
                .chain(chunk.sh0[i])
            {
                put(value)?;
            }
            for coeff in &chunk.sh_rest[i * per_rest..(i + 1) * per_rest] {
                for value in coeff {
                    put(*value)?;
                }
            }
        }
    }
    for bucket in &mut buckets {
        bucket.flush()?;
    }
    drop(buckets);

    // Pass 3: paginate each bucket in memory and emit pages.
    let mut writer = PageStoreWriter::create(out, PageKind::Splat, page_capacity, degree)?;
    for index in 0..bucket_sizes.len() {
        let path = bucket_path(index);
        let bytes_on_disk = std::fs::metadata(&path)?.len();
        let _tracker = tracked_host_allocation(bytes_on_disk, "fusion-page-store-sort-bucket")?;
        let mut raw = Vec::with_capacity(bytes_on_disk as usize);
        BufReader::new(File::open(&path)?).read_to_end(&mut raw)?;
        std::fs::remove_file(&path)?;
        let values: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        drop(raw);
        let count = values.len() / floats;
        let bucket_positions: Vec<[f32; 3]> = (0..count)
            .map(|i| {
                let r = &values[i * floats..];
                [r[0], r[1], r[2]]
            })
            .collect();
        for run in paginate(&bucket_positions, page_capacity as usize) {
            let mut chunk = SplatChunk::default();
            for i in &run {
                let r = &values[*i as usize * floats..(*i as usize + 1) * floats];
                chunk.positions.push([r[0], r[1], r[2]]);
                chunk.scales.push([r[3], r[4], r[5]]);
                chunk.rotations.push([r[6], r[7], r[8], r[9]]);
                chunk.opacities.push(r[10]);
                chunk.sh0.push([r[11], r[12], r[13]]);
                for k in 0..per_rest {
                    chunk
                        .sh_rest
                        .push([r[14 + 3 * k], r[15 + 3 * k], r[16 + 3 * k]]);
                }
            }
            writer.push_splat_page(&chunk)?;
        }
    }
    let summary = writer.finish()?;
    if summary.logical_primitives != total {
        return Err(RenderError::Upload(format!(
            "page store build lost primitives: wrote {} of {total}",
            summary.logical_primitives
        )));
    }
    Ok(summary)
}

// ---------------------------------------------------------------------------
// Synthetic large-scale index
// ---------------------------------------------------------------------------

/// Description of a tiled point field whose index aliases a few on-disk
/// template pages across a large grid — a billion-primitive *index* over real
/// disk-backed payloads, used to exercise out-of-core residency at scale
/// without shipping a multi-gigabyte dataset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SyntheticFieldDesc {
    pub page_capacity: u32,
    /// Distinct payload pages written to disk.
    pub template_pages: u32,
    /// Grid of tiles along x and z.
    pub grid: (u32, u32),
    /// World footprint of one tile along x and z.
    pub cell_size: f32,
    /// World position of the grid's minimum corner (x, base height, z).
    pub origin: [f32; 3],
    /// Vertical extent of the point slab inside a tile.
    pub thickness: f32,
    /// Axis-aligned xz rectangle (min_x, min_z, max_x, max_z) left empty.
    pub hole: [f32; 4],
    pub seed: u32,
}

/// Avalanche hash used for deterministic procedural content.
pub fn hash_u32(value: u32) -> u32 {
    let mut x = value;
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb_352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// Uniform float in `[0, 1)` from a hash state.
pub fn hash_unit(value: u32) -> f32 {
    f32_from_u32(hash_u32(value) >> 8) * (1.0 / 16_777_216.0)
}

/// Write the synthetic tiled point field described by `desc`.
pub fn write_synthetic_point_field(
    path: impl AsRef<Path>,
    desc: &SyntheticFieldDesc,
) -> Result<PageStoreSummary, RenderError> {
    if desc.template_pages == 0 || desc.grid.0 == 0 || desc.grid.1 == 0 {
        return Err(RenderError::Upload(
            "synthetic point field needs at least one template page and a non-empty grid".into(),
        ));
    }
    if !(desc.cell_size.is_finite() && desc.cell_size > 0.0 && desc.thickness >= 0.0) {
        return Err(RenderError::Upload(
            "synthetic point field cell size must be finite and > 0".into(),
        ));
    }
    let mut writer = PageStoreWriter::create(path, PageKind::Points, desc.page_capacity, 0)?;
    // The first `template_pages` tiles carry real payloads; every later tile
    // is an index entry that aliases one of them at its own origin.
    let mut templates: Vec<(u32, u32, u32)> = Vec::new(); // (page, gx, gz)
    let mut k = 0u32;
    for gz in 0..desc.grid.1 {
        for gx in 0..desc.grid.0 {
            let x0 = desc.origin[0] + f32_from_u32(gx) * desc.cell_size;
            let z0 = desc.origin[2] + f32_from_u32(gz) * desc.cell_size;
            let overlaps_hole = x0 < desc.hole[2]
                && x0 + desc.cell_size > desc.hole[0]
                && z0 < desc.hole[3]
                && z0 + desc.cell_size > desc.hole[1];
            if overlaps_hole {
                continue;
            }
            if (templates.len() as u32) < desc.template_pages {
                let template = templates.len() as u32;
                let mut positions = Vec::with_capacity(desc.page_capacity as usize);
                let mut rgba = Vec::with_capacity(desc.page_capacity as usize);
                for i in 0..desc.page_capacity {
                    let state = desc
                        .seed
                        .wrapping_add(template.wrapping_mul(0x9e37_79b9))
                        .wrapping_add(i.wrapping_mul(0x85eb_ca6b));
                    positions.push([
                        x0 + hash_unit(state) * desc.cell_size,
                        desc.origin[1] + hash_unit(state ^ 0x02e5_be93) * desc.thickness,
                        z0 + hash_unit(state ^ 0x68bc_21eb) * desc.cell_size,
                    ]);
                    let shade = 96 + (hash_u32(state ^ 0x1b87_3593) & 0x3f) as u8;
                    rgba.push([shade / 2, shade, shade / 3, 5]);
                }
                let page = writer.push_point_page(&positions, &rgba)?;
                templates.push((page, gx, gz));
                continue;
            }
            let (page, tx, tz) = templates[(hash_u32(desc.seed ^ k) % desc.template_pages) as usize];
            k = k.wrapping_add(1);
            let base = writer.entries[page as usize].origin;
            writer.push_alias(
                page,
                [
                    base[0] + (f32_from_u32(gx) - f32_from_u32(tx)) * desc.cell_size,
                    base[1],
                    base[2] + (f32_from_u32(gz) - f32_from_u32(tz)) * desc.cell_size,
                ],
            )?;
        }
    }
    writer.finish()
}

// ---------------------------------------------------------------------------
// COPC octree as a page source
// ---------------------------------------------------------------------------

/// Maps georeferenced LAS coordinates into the fused scene frame: positions
/// are measured from `origin` in f64 and, for z-up data, re-axed to the
/// tracer's y-up frame as `(east, height, -north)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointCloudFrame {
    pub origin: [f64; 3],
    pub z_up: bool,
}

impl Default for PointCloudFrame {
    fn default() -> Self {
        Self {
            origin: [0.0; 3],
            z_up: true,
        }
    }
}

impl PointCloudFrame {
    fn anchor(&self) -> crate::camera::Anchor {
        let mut anchor = crate::camera::Anchor::with_epsilon(f64::MIN_POSITIVE)
            .expect("positive rebase threshold");
        anchor.rebase_if_needed(DVec3::from_array(self.origin));
        anchor
    }

    /// Scene-frame f32 position of an f64 LAS coordinate. The f64 -> f32
    /// crossing is the typed Anchor exit; this only re-axes its result.
    pub fn to_scene(&self, anchor: &crate::camera::Anchor, p: DVec3) -> [f32; 3] {
        let local = anchor.to_render_f32(crate::geo::units::SceneCoord::scene(p));
        if self.z_up {
            [local.x, local.z, -local.y]
        } else {
            local.to_array()
        }
    }
}

/// A run of points `[first, first + count)` of one octree node.
struct CopcPart {
    key: OctreeKey,
    first: u32,
    count: u32,
}

/// One page: a run of a large node, or several small nodes of one depth.
struct CopcPage {
    parts: Vec<CopcPart>,
    count: u32,
    aabb: Aabb,
}

/// Morton code of an octree node's cell coordinates (21 bits per axis).
fn node_morton(key: &OctreeKey) -> u64 {
    spread_bits_21(u64::from(key.x))
        | (spread_bits_21(u64::from(key.y)) << 1)
        | (spread_bits_21(u64::from(key.z)) << 2)
}

/// Page layout of a COPC hierarchy (`(key, point count, scene-frame cell)` per
/// non-empty node). Nodes holding at least half a page keep standalone pages:
/// runs of at most `capacity` points sharing the node's cell. Smaller nodes
/// are packed per depth in Morton order into shared pages whose box is the
/// union of the member cells. Standalone pages come first, ordered by
/// `(depth, x, y, z, first)`; packed pages follow in creation order.
fn pack_copc_nodes(nodes: Vec<(OctreeKey, u32, Aabb)>, capacity: u32) -> Vec<CopcPage> {
    let mut pages = Vec::new();
    let mut small: Vec<(OctreeKey, u32, Aabb)> = Vec::new();
    let mut big: Vec<(OctreeKey, u32, Aabb)> = Vec::new();
    for node in nodes {
        if node.1 >= capacity / 2 {
            big.push(node)
        } else if node.1 > 0 {
            small.push(node)
        }
    }
    big.sort_by_key(|a| (a.0.depth, a.0.x, a.0.y, a.0.z));
    for (key, count, aabb) in big {
        let mut first = 0;
        while first < count {
            let run = (count - first).min(capacity);
            pages.push(CopcPage {
                parts: vec![CopcPart {
                    key: key.clone(),
                    first,
                    count: run,
                }],
                count: run,
                aabb,
            });
            first += run;
        }
    }
    small.sort_by(|a, b| {
        (a.0.depth, node_morton(&a.0), a.0.x, a.0.y, a.0.z).cmp(&(
            b.0.depth,
            node_morton(&b.0),
            b.0.x,
            b.0.y,
            b.0.z,
        ))
    });
    let mut open: Option<CopcPage> = None;
    let mut open_depth = u32::MAX;
    for (key, count, aabb) in small {
        let fits = open
            .as_ref()
            .is_some_and(|p| open_depth == key.depth && p.count + count <= capacity);
        if !fits {
            pages.extend(open.take());
            open = Some(CopcPage {
                parts: Vec::new(),
                count: 0,
                aabb: Aabb::EMPTY,
            });
            open_depth = key.depth;
        }
        let page = open.as_mut().unwrap();
        page.parts.push(CopcPart {
            key,
            first: 0,
            count,
        });
        page.count += count;
        page.aabb = page.aabb.union(aabb);
    }
    pages.extend(open);
    pages
}

/// COPC octree nodes as pages. A node holding at least half a page is split
/// into consecutive runs that share the node's bounds; smaller nodes are
/// packed together (see `pack_copc_nodes`).
pub struct CopcPageSource {
    dataset: CopcDataset,
    frame: PointCloudFrame,
    pages: Vec<CopcPage>,
    total: u64,
    path: PathBuf,
    /// Last decoded node, reused by the runs of an oversized node.
    cache: Mutex<Option<(OctreeKey, Arc<PagePayload>)>>,
}

/// ASPRS classification palette (linear RGB) for colourless returns.
fn classification_color(class: u8) -> [f32; 3] {
    match class {
        2 => [0.35, 0.26, 0.17],     // ground
        3 => [0.25, 0.42, 0.16],     // low vegetation
        4 => [0.17, 0.38, 0.12],     // medium vegetation
        5 => [0.10, 0.30, 0.08],     // high vegetation
        6 => [0.55, 0.52, 0.50],     // building
        9 => [0.10, 0.22, 0.40],     // water
        17 => [0.40, 0.40, 0.42],    // bridge deck
        _ => [0.50, 0.50, 0.50],
    }
}

impl CopcPageSource {
    pub fn open(
        path: impl AsRef<Path>,
        frame: PointCloudFrame,
        page_capacity: u32,
    ) -> Result<Self, RenderError> {
        let path = path.as_ref().to_path_buf();
        if frame.origin.iter().any(|v| !v.is_finite()) {
            return Err(RenderError::Upload(
                "point cloud frame origin must be finite".into(),
            ));
        }
        let dataset = CopcDataset::open(&path).map_err(|e| {
            RenderError::Upload(format!("COPC {}: {e}", path.display()))
        })?;
        let anchor = frame.anchor();
        let mut nodes = Vec::new();
        let mut total = 0u64;
        for node in dataset.nodes() {
            if node.point_count == 0 {
                continue;
            }
            // Octree cell mapped to the scene frame (the re-axing can swap
            // min/max, so rebuild them). Every point of a node lies inside
            // its cell, so the cell is a conservative page box.
            let a = frame.to_scene(&anchor, node.bounds.min);
            let b = frame.to_scene(&anchor, node.bounds.max);
            let mut aabb = Aabb::EMPTY;
            aabb = aabb.union(Aabb::new(a, a)).union(Aabb::new(b, b));
            let count = u32::try_from(node.point_count).map_err(|_| {
                RenderError::Upload(format!(
                    "COPC {}: node {} holds more points than a page index can address",
                    path.display(),
                    node.key.to_string()
                ))
            })?;
            total += u64::from(count);
            nodes.push((node.key.clone(), count, aabb));
        }
        if page_capacity == 0 {
            return Err(RenderError::Upload(
                "COPC page capacity must be > 0".into(),
            ));
        }
        // Deterministic layout regardless of hash-map iteration.
        let pages = pack_copc_nodes(nodes, page_capacity);
        if pages.is_empty() {
            return Err(RenderError::Upload(format!(
                "COPC {}: the hierarchy contains no points",
                path.display()
            )));
        }
        Ok(Self {
            dataset,
            frame,
            pages,
            total,
            path,
            cache: Mutex::new(None),
        })
    }

    fn decode_node(&self, key: &OctreeKey) -> Result<Arc<PagePayload>, RenderError> {
        {
            let cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
            if let Some((cached, payload)) = cache.as_ref() {
                if cached == key {
                    return Ok(payload.clone());
                }
            }
        }
        let data = self.dataset.read_points(key).map_err(|e| {
            RenderError::Upload(format!(
                "COPC {}: node {}: {e}",
                self.path.display(),
                key.to_string()
            ))
        })?;
        let anchor = self.frame.anchor();
        let n = data.positions.len() / 3;
        let mut positions = Vec::with_capacity(n);
        let mut colors = Vec::with_capacity(n);
        for i in 0..n {
            let p = DVec3::new(
                data.positions[3 * i],
                data.positions[3 * i + 1],
                data.positions[3 * i + 2],
            );
            positions.push(self.frame.to_scene(&anchor, p));
            let color = if let Some(rgb) = data.colors.as_ref().filter(|c| c.len() >= 3 * n) {
                [
                    srgb_byte_to_linear(rgb[3 * i]),
                    srgb_byte_to_linear(rgb[3 * i + 1]),
                    srgb_byte_to_linear(rgb[3 * i + 2]),
                ]
            } else if let Some(class) = data.classifications.as_ref().and_then(|c| c.get(i)) {
                classification_color(*class)
            } else if let Some(intensity) = data.intensities.as_ref().and_then(|v| v.get(i)) {
                [f32::from(*intensity) / 65_535.0; 3]
            } else {
                [0.5; 3]
            };
            colors.push(color);
        }
        let payload = Arc::new(PagePayload::Points { positions, colors });
        *self.cache.lock().unwrap_or_else(|p| p.into_inner()) =
            Some((key.clone(), payload.clone()));
        Ok(payload)
    }
}

impl PageSource for CopcPageSource {
    fn kind(&self) -> PageKind {
        PageKind::Points
    }

    fn page_count(&self) -> u32 {
        self.pages.len() as u32
    }

    fn meta(&self, page: u32) -> PageMeta {
        let page = &self.pages[page as usize];
        PageMeta {
            kind: PageKind::Points,
            count: page.count,
            aabb: page.aabb,
        }
    }

    fn load(&self, page: u32) -> Result<PagePayload, RenderError> {
        let page = &self.pages[page as usize];
        let mut out_positions = Vec::with_capacity(page.count as usize);
        let mut out_colors = Vec::with_capacity(page.count as usize);
        for part in &page.parts {
            let node = self.decode_node(&part.key)?;
            let PagePayload::Points { positions, colors } = node.as_ref() else {
                unreachable!("COPC nodes decode to point payloads");
            };
            let (first, end) = (part.first as usize, (part.first + part.count) as usize);
            if end > positions.len() {
                return Err(RenderError::Upload(format!(
                    "COPC {}: node {} decoded {} points but the hierarchy promised {end}",
                    self.path.display(),
                    part.key.to_string(),
                    positions.len()
                )));
            }
            out_positions.extend_from_slice(&positions[first..end]);
            out_colors.extend_from_slice(&colors[first..end]);
        }
        Ok(PagePayload::Points {
            positions: out_positions,
            colors: out_colors,
        })
    }

    fn logical_primitives(&self) -> u64 {
        self.total
    }

    fn label(&self) -> String {
        format!(
            "copc[{} nodes, {} points] {}",
            self.dataset.node_count(),
            self.total,
            self.path.display()
        )
    }
}

// ---------------------------------------------------------------------------
// GPU-ready decode
// ---------------------------------------------------------------------------

/// A page decoded into the exact bytes the residency pool stores: primitive
/// records in BVH-leaf order plus the per-page tree. The host bytes are
/// registered with the memory tracker for as long as the page is in flight.
pub struct DecodedPage {
    pub page: u32,
    pub kind: PageKind,
    pub count: u32,
    pub nodes: Vec<FusionBvhNode>,
    pub splats: Vec<SplatGpu>,
    pub inv_cov: Vec<InvCovGpu>,
    pub points: Vec<PointGpu>,
    _tracker: ResourceHandle,
}

impl DecodedPage {
    /// Host bytes held by this decoded page.
    pub fn byte_size(&self) -> u64 {
        (self.nodes.len() * std::mem::size_of::<FusionBvhNode>()
            + self.splats.len() * std::mem::size_of::<SplatGpu>()
            + self.inv_cov.len() * std::mem::size_of::<InvCovGpu>()
            + self.points.len() * std::mem::size_of::<PointGpu>()) as u64
    }

    /// Decode a payload: compute each primitive's bounding proxy (3-sigma
    /// ellipsoid AABB for splats, sphelet box for points), build the per-page
    /// tree over the proxies, and emit records in leaf order with the packed
    /// inverse covariance precomputed.
    /// `surfels` estimates an oriented disc per LiDAR return on locally
    /// planar neighbourhoods of this page (see `surfel::estimate_point_surfels`);
    /// otherwise every return is an isotropic sphelet.
    pub fn from_payload(
        page: u32,
        payload: &PagePayload,
        lidar_radius: f32,
        surfels: bool,
    ) -> Result<Self, RenderError> {
        let count = payload.len();
        if count == 0 {
            return Err(RenderError::Upload(format!("page {page} decoded to zero primitives")));
        }
        match payload {
            PagePayload::Splats(chunk) => {
                let per_rest = if chunk.is_empty() {
                    0
                } else {
                    chunk.sh_rest.len() / chunk.len()
                };
                let boxes: Vec<Aabb> = (0..count)
                    .map(|i| {
                        let half = sigma_extent(
                            chunk.scales[i],
                            chunk.rotations[i],
                            super::kernel::SIGMA_CUTOFF,
                        );
                        let p = chunk.positions[i];
                        Aabb::new(
                            [p[0] - half[0], p[1] - half[1], p[2] - half[2]],
                            [p[0] + half[0], p[1] + half[1], p[2] + half[2]],
                        )
                    })
                    .collect();
                if boxes.iter().any(|b| !b.is_valid()) {
                    return Err(RenderError::Upload(format!(
                        "page {page} contains a splat with a non-finite bounding proxy"
                    )));
                }
                let bvh = build_bvh(&boxes, BLAS_LEAF_SIZE);
                let degree = if per_rest >= 3 { GPU_SH_DEGREE } else { 0 };
                let mut splats = Vec::with_capacity(count);
                let mut inv_cov = Vec::with_capacity(count);
                for &i in &bvh.order {
                    let i = i as usize;
                    let p = chunk.positions[i];
                    let c = chunk.sh0[i];
                    let mut sh1 = [[0.0f32; 4]; 3];
                    if degree >= 1 {
                        for (k, slot) in sh1.iter_mut().enumerate() {
                            let coeff = chunk.sh_rest[i * per_rest + k];
                            *slot = [coeff[0], coeff[1], coeff[2], 0.0];
                        }
                    }
                    splats.push(SplatGpu {
                        pos_opacity: [p[0], p[1], p[2], chunk.opacities[i]],
                        sh0_degree: [c[0], c[1], c[2], f32_from_u32(degree)],
                        sh1,
                    });
                    let m = inverse_covariance(chunk.scales[i], chunk.rotations[i]);
                    inv_cov.push(InvCovGpu {
                        m0: [m[0], m[1], m[2], m[3]],
                        m1: [m[4], m[5], 0.0, 0.0],
                    });
                }
                let bytes = bvh.nodes.len() * 32 + count * (80 + 32);
                Ok(Self {
                    page,
                    kind: PageKind::Splat,
                    count: count as u32,
                    nodes: bvh.nodes,
                    splats,
                    inv_cov,
                    points: Vec::new(),
                    _tracker: tracked_host_allocation(bytes as u64, "fusion-decoded-page")?,
                })
            }
            PagePayload::Points { positions, colors } => {
                if !(lidar_radius.is_finite() && lidar_radius > 0.0) {
                    return Err(RenderError::Upload(format!(
                        "LiDAR sphelet radius must be finite and > 0, got {lidar_radius}"
                    )));
                }
                let boxes: Vec<Aabb> = positions
                    .iter()
                    .map(|p| {
                        Aabb::new(
                            [p[0] - lidar_radius, p[1] - lidar_radius, p[2] - lidar_radius],
                            [p[0] + lidar_radius, p[1] + lidar_radius, p[2] + lidar_radius],
                        )
                    })
                    .collect();
                if boxes.iter().any(|b| !b.is_valid()) {
                    return Err(RenderError::Upload(format!(
                        "page {page} contains a point with a non-finite position"
                    )));
                }
                let bvh = build_bvh(&boxes, BLAS_LEAF_SIZE);
                let normals = if surfels {
                    super::surfel::estimate_point_surfels(positions, lidar_radius)
                } else {
                    vec![super::surfel::SURFEL_SPHERE; positions.len()]
                };
                let points = bvh
                    .order
                    .iter()
                    .map(|&i| {
                        let p = positions[i as usize];
                        let c = colors[i as usize];
                        PointGpu {
                            pos_radius: [p[0], p[1], p[2], lidar_radius],
                            color: c,
                            normal_oct: normals[i as usize],
                        }
                    })
                    .collect();
                let bytes = bvh.nodes.len() * 32 + count * 32;
                Ok(Self {
                    page,
                    kind: PageKind::Points,
                    count: count as u32,
                    nodes: bvh.nodes,
                    splats: Vec::new(),
                    inv_cov: Vec::new(),
                    points,
                    _tracker: tracked_host_allocation(bytes as u64, "fusion-decoded-page")?,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Residency pool bookkeeping
// ---------------------------------------------------------------------------

/// Counters describing how the residency pool behaved during a render.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidencyStats {
    pub loads: u64,
    pub evictions: u64,
    pub peak_resident_pages: u32,
    pub resident_pages: u32,
}

struct SlotPool {
    page_in_slot: Vec<u32>,
    last_used: Vec<u64>,
}

/// Fixed-size residency pool with least-recently-used eviction. One slot
/// array per page kind (splat records and point records have different
/// strides); a page's slot index is what the GPU page table publishes.
pub struct Residency {
    slot_of: Vec<u32>,
    kind_of: Vec<PageKind>,
    pools: [SlotPool; 2],
    pub stats: ResidencyStats,
}

/// Outcome of admitting a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Admission {
    pub slot: u32,
    pub evicted: Option<u32>,
}

impl Residency {
    pub fn new(kinds: Vec<PageKind>, splat_slots: u32, point_slots: u32) -> Self {
        let pool = |slots: u32| SlotPool {
            page_in_slot: vec![NOT_RESIDENT; slots as usize],
            last_used: vec![0; slots as usize],
        };
        Self {
            slot_of: vec![NOT_RESIDENT; kinds.len()],
            kind_of: kinds,
            pools: [pool(splat_slots), pool(point_slots)],
            stats: ResidencyStats::default(),
        }
    }

    pub fn page_count(&self) -> usize {
        self.slot_of.len()
    }

    pub fn kind(&self, page: u32) -> PageKind {
        self.kind_of[page as usize]
    }

    /// Slot of a resident page.
    pub fn slot(&self, page: u32) -> Option<u32> {
        let slot = self.slot_of[page as usize];
        (slot != NOT_RESIDENT).then_some(slot)
    }

    pub fn slots(&self, kind: PageKind) -> u32 {
        self.pools[kind as usize].page_in_slot.len() as u32
    }

    /// Record that `page` was traversed during `frame` (LRU recency).
    pub fn touch(&mut self, page: u32, frame: u64) {
        if let Some(slot) = self.slot(page) {
            let pool = &mut self.pools[self.kind_of[page as usize] as usize];
            pool.last_used[slot as usize] = pool.last_used[slot as usize].max(frame);
        }
    }

    /// Give `page` a slot, evicting the least recently used page of the same
    /// kind when the pool is full. Pages used during `frame` are never
    /// evicted: when every slot is pinned by the current frame the working
    /// set does not fit and the caller gets `None`.
    pub fn admit(&mut self, page: u32, frame: u64) -> Option<Admission> {
        if let Some(slot) = self.slot(page) {
            self.touch(page, frame);
            return Some(Admission {
                slot,
                evicted: None,
            });
        }
        let pool = &mut self.pools[self.kind_of[page as usize] as usize];
        let mut victim: Option<usize> = None;
        for slot in 0..pool.page_in_slot.len() {
            if pool.page_in_slot[slot] == NOT_RESIDENT {
                victim = Some(slot);
                break;
            }
            if pool.last_used[slot] < frame
                && victim.is_none_or(|v| pool.last_used[slot] < pool.last_used[v])
            {
                victim = Some(slot);
            }
        }
        let slot = victim?;
        let previous = pool.page_in_slot[slot];
        pool.page_in_slot[slot] = page;
        pool.last_used[slot] = frame;
        let evicted = (previous != NOT_RESIDENT).then_some(previous);
        if let Some(previous) = evicted {
            self.slot_of[previous as usize] = NOT_RESIDENT;
            self.stats.evictions += 1;
        } else {
            self.stats.resident_pages += 1;
        }
        self.slot_of[page as usize] = slot as u32;
        self.stats.loads += 1;
        self.stats.peak_resident_pages = self.stats.peak_resident_pages.max(self.stats.resident_pages);
        Some(Admission {
            slot: slot as u32,
            evicted,
        })
    }
}

// ---------------------------------------------------------------------------
// Asynchronous loader
// ---------------------------------------------------------------------------

/// Where a global page id lives: (source index, page within the source).
pub type PageAddress = (u32, u32);

struct LoadJob {
    page: u32,
    address: PageAddress,
}

/// Background page loader: requests are queued to worker threads that read
/// and decode pages off the render thread; completed pages are collected
/// without blocking (`try_recv`) or with a blocking wait (`recv`).
pub struct PageLoader {
    jobs: Option<Sender<LoadJob>>,
    done: Receiver<(u32, Result<DecodedPage, RenderError>)>,
    workers: Vec<std::thread::JoinHandle<()>>,
    in_flight: usize,
}

impl PageLoader {
    /// `owner` attributes the workers' decoded-page host allocations to the
    /// render that issued the requests (per-render peak accounting).
    pub fn new(
        sources: Arc<Vec<Arc<dyn PageSource>>>,
        lidar_radius: f32,
        lidar_surfels: bool,
        workers: usize,
        owner: Option<AllocationOwner>,
    ) -> Self {
        let (job_tx, job_rx) = channel::<LoadJob>();
        let (done_tx, done_rx) = channel();
        let job_rx = Arc::new(Mutex::new(job_rx));
        let workers = (0..workers.max(1))
            .map(|index| {
                let job_rx = job_rx.clone();
                let done_tx = done_tx.clone();
                let sources = sources.clone();
                let owner = owner.clone();
                std::thread::Builder::new()
                    .name(format!("fusion-page-loader-{index}"))
                    .spawn(move || loop {
                        let _owner_guard = owner.as_ref().map(AllocationOwner::activate);
                        let job = {
                            let guard = job_rx.lock().unwrap_or_else(|p| p.into_inner());
                            guard.recv()
                        };
                        let Ok(job) = job else { break };
                        let (source, local) = job.address;
                        let result = sources[source as usize]
                            .load(local)
                            .and_then(|payload| {
                                DecodedPage::from_payload(
                                    job.page,
                                    &payload,
                                    lidar_radius,
                                    lidar_surfels,
                                )
                            });
                        if done_tx.send((job.page, result)).is_err() {
                            break;
                        }
                    })
                    .expect("spawn fusion page loader")
            })
            .collect();
        Self {
            jobs: Some(job_tx),
            done: done_rx,
            workers,
            in_flight: 0,
        }
    }

    /// Queue an asynchronous page request.
    pub fn request(&mut self, page: u32, address: PageAddress) {
        if let Some(jobs) = &self.jobs {
            if jobs.send(LoadJob { page, address }).is_ok() {
                self.in_flight += 1;
            }
        }
    }

    /// Requests issued but not yet collected.
    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    /// Collect one finished page without blocking.
    pub fn try_recv(&mut self) -> Option<(u32, Result<DecodedPage, RenderError>)> {
        match self.done.try_recv() {
            Ok(done) => {
                self.in_flight -= 1;
                Some(done)
            }
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
        }
    }

    /// Wait for one finished page; `None` when nothing is in flight.
    pub fn recv(&mut self) -> Option<(u32, Result<DecodedPage, RenderError>)> {
        if self.in_flight == 0 {
            return None;
        }
        let done = self.done.recv().ok()?;
        self.in_flight -= 1;
        Some(done)
    }
}

impl Drop for PageLoader {
    fn drop(&mut self) {
        self.jobs = None;
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Convenience used by tests and tools: number of pages needed for `count`
/// primitives at `capacity` per page.
pub fn pages_for(count: u64, capacity: u32) -> u64 {
    count.div_ceil(u64::from(capacity.max(1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_copc_nodes_are_packed_without_losing_or_duplicating_points() {
        use crate::splat::fixture::{write_copc, LasPoint, COPC_HALFSIZE, COPC_ORIGIN};
        let dir = std::env::temp_dir().join(format!("forge3d-copc-pack-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("many_nodes.copc.laz");
        let n = 3000u32;
        let points: Vec<LasPoint> = (0..n)
            .map(|i| {
                let u = |k: u32| f64::from(hash_unit(i.wrapping_mul(3).wrapping_add(k))) * 1.98 - 0.99;
                LasPoint {
                    xyz: [
                        COPC_ORIGIN[0] + u(1) * COPC_HALFSIZE,
                        COPC_ORIGIN[1] + u(2) * COPC_HALFSIZE,
                        COPC_ORIGIN[2] + u(3) * COPC_HALFSIZE,
                    ],
                    rgb: [100, 120, 90],
                    classification: 2,
                }
            })
            .collect();
        write_copc(&path, &points).unwrap();
        let capacity = 512u32;
        let source = CopcPageSource::open(
            &path,
            PointCloudFrame {
                origin: COPC_ORIGIN,
                z_up: true,
            },
            capacity,
        )
        .unwrap();
        let nodes = crate::pointcloud::CopcDataset::open(&path)
            .unwrap()
            .nodes()
            .iter()
            .filter(|node| node.point_count > 0)
            .count() as u32;
        let pages = source.page_count();
        let floor = n.div_ceil(capacity);
        println!("{n} points in {nodes} non-empty nodes -> {pages} pages (floor {floor})");
        assert!(
            pages >= floor && pages <= 2 * floor,
            "pages {pages}, floor {floor}"
        );
        assert!(pages < nodes, "pages {pages} must be fewer than nodes {nodes}");
        let mut loaded: Vec<[u32; 3]> = Vec::new();
        for page in 0..pages {
            let meta = source.meta(page);
            assert!(meta.count <= capacity);
            let PagePayload::Points { positions, .. } = source.load(page).unwrap() else {
                panic!()
            };
            assert_eq!(positions.len() as u32, meta.count);
            for p in &positions {
                for a in 0..3 {
                    assert!(p[a] >= meta.aabb.min[a] - 1e-3 && p[a] <= meta.aabb.max[a] + 1e-3);
                }
                loaded.push(p.map(f32::to_bits));
            }
        }
        assert_eq!(loaded.len() as u32, n);
        loaded.sort_unstable();
        loaded.dedup();
        assert_eq!(loaded.len() as u32, n, "a point was duplicated or lost");
        std::fs::remove_dir_all(&dir).ok();
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("forge3d-fusion-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn test_cloud(n: usize, degree: u32) -> GaussianSplatCloud {
        let per = sh_rest_count(degree);
        let u = |i: usize, k: u32| hash_unit((i as u32).wrapping_mul(31).wrapping_add(k));
        GaussianSplatCloud::from_parts(
            (0..n)
                .map(|i| [u(i, 1) * 40.0 - 20.0, u(i, 2) * 4.0, u(i, 3) * 40.0 - 20.0])
                .collect(),
            (0..n)
                .map(|i| [0.05 + 0.2 * u(i, 4), 0.05 + 0.2 * u(i, 5), 0.05 + 0.2 * u(i, 6)])
                .collect(),
            (0..n)
                .map(|i| [1.0, u(i, 7) - 0.5, u(i, 8) - 0.5, u(i, 9) - 0.5])
                .collect(),
            (0..n).map(|i| 0.2 + 0.7 * u(i, 10)).collect(),
            (0..n).map(|i| [u(i, 11), u(i, 12), u(i, 13)]).collect(),
            (degree > 0).then(|| super::super::ShRest {
                degree,
                coeffs: (0..n * per)
                    .map(|j| [u(j, 14) - 0.5, u(j, 15) - 0.5, u(j, 16) - 0.5])
                    .collect(),
            }),
        )
        .unwrap()
    }

    fn collect_splats(source: &dyn PageSource) -> Vec<([f32; 3], [f32; 3], f32)> {
        let mut out = Vec::new();
        for page in 0..source.page_count() {
            let PagePayload::Splats(chunk) = source.load(page).unwrap() else {
                panic!("expected splats");
            };
            assert_eq!(chunk.len(), source.meta(page).count as usize);
            for i in 0..chunk.len() {
                out.push((chunk.positions[i], chunk.scales[i], chunk.opacities[i]));
            }
        }
        out.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out
    }

    #[test]
    fn page_store_round_trips_a_cloud_and_pages_are_spatially_coherent() {
        let dir = temp_dir("store");
        let cloud = test_cloud(5000, 1);
        let path = dir.join("cloud.f3dpages");
        let summary = write_splat_page_store(&path, &cloud, 512).unwrap();
        assert_eq!(summary.logical_primitives, 5000);
        assert_eq!(summary.page_count, 10);
        let store = PageStoreFile::open(&path).unwrap();
        assert_eq!(store.kind(), PageKind::Splat);
        assert_eq!(store.logical_primitives(), 5000);
        let mut expected: Vec<_> = (0..cloud.len())
            .map(|i| (cloud.positions[i], cloud.scales[i], cloud.opacities[i]))
            .collect();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let got = collect_splats(&store);
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(&expected) {
            for axis in 0..3 {
                // Positions are stored relative to the page origin.
                assert!((a.0[axis] - b.0[axis]).abs() < 1e-4);
            }
            assert_eq!((a.1, a.2), (b.1, b.2));
        }
        // Pages are compact: the mean page box is far smaller than the cloud.
        let cloud_extent = 40.0f32;
        let mean_extent: f32 = (0..store.page_count())
            .map(|p| {
                let aabb = store.meta(p).aabb;
                (aabb.max[0] - aabb.min[0]).max(aabb.max[2] - aabb.min[2])
            })
            .sum::<f32>()
            / f32_from_u32(store.page_count());
        assert!(mean_extent < 0.5 * cloud_extent, "{mean_extent}");
        // Every splat's 3-sigma proxy lies inside its page box.
        let in_memory = SplatCloudSource::new(Arc::new(test_cloud(5000, 1)), 512);
        assert_eq!(collect_splats(&in_memory).len(), 5000);
        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn external_sort_build_from_ply_preserves_every_splat() {
        let dir = temp_dir("ooc");
        let cloud = test_cloud(3000, 1);
        let ply = dir.join("cloud.ply");
        super::super::save_gaussian_splats(&ply, &cloud).unwrap();
        let out = dir.join("cloud.f3dpages");
        // A budget small enough to force several sort buckets.
        let summary = build_splat_page_store_from_ply(&ply, &out, 256, 96 * 1024).unwrap();
        assert_eq!(summary.logical_primitives, 3000);
        let store = PageStoreFile::open(&out).unwrap();
        let got = collect_splats(&store);
        assert_eq!(got.len(), 3000);
        let leftovers = std::fs::read_dir(&dir)
            .unwrap()
            .filter(|e| e.as_ref().unwrap().path().extension().is_some_and(|x| x == "tmp"))
            .count();
        assert_eq!(leftovers, 0, "bucket temp files must be cleaned up");
        // An impossible budget is a diagnostic, not a silent overrun.
        let tiny = build_splat_page_store_from_ply(&ply, dir.join("x.f3dpages"), 256, 64);
        assert!(format!("{}", tiny.unwrap_err()).contains("budget"));
        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn synthetic_field_indexes_a_billion_points_over_a_few_megabytes() {
        let dir = temp_dir("field");
        let path = dir.join("field.f3dpages");
        let desc = SyntheticFieldDesc {
            page_capacity: 4096,
            template_pages: 8,
            grid: (496, 496),
            cell_size: 16.0,
            origin: [-3968.0, 0.0, -3968.0],
            thickness: 2.0,
            hole: [-64.0, -64.0, 64.0, 64.0],
            seed: 7,
        };
        let summary = write_synthetic_point_field(&path, &desc).unwrap();
        assert!(summary.logical_primitives > 1_000_000_000, "{summary:?}");
        assert!(summary.payload_bytes < 1 << 20, "{summary:?}");
        let store = PageStoreFile::open(&path).unwrap();
        assert_eq!(store.logical_primitives(), summary.logical_primitives);
        // Aliased pages land inside their own tile and never in the hole.
        for page in [8u32, 9, 1000, store.page_count() - 1] {
            let meta = store.meta(page);
            let PagePayload::Points { positions, .. } = store.load(page).unwrap() else {
                panic!("expected points");
            };
            assert_eq!(positions.len(), 4096);
            for p in &positions {
                for axis in 0..3 {
                    assert!(p[axis] >= meta.aabb.min[axis] - 1e-2);
                    assert!(p[axis] <= meta.aabb.max[axis] + 1e-2);
                }
            }
            let overlaps = meta.aabb.min[0] < 64.0
                && meta.aabb.max[0] > -64.0
                && meta.aabb.min[2] < 64.0
                && meta.aabb.max[2] > -64.0;
            assert!(!overlaps, "page {page} intrudes into the hole: {meta:?}");
        }
        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn corrupt_page_stores_are_rejected() {
        let dir = temp_dir("corrupt");
        let path = dir.join("cloud.f3dpages");
        write_splat_page_store(&path, &test_cloud(100, 0), 64).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let bad_magic = dir.join("magic.f3dpages");
        let mut copy = bytes.clone();
        copy[0] = b'X';
        std::fs::write(&bad_magic, &copy).unwrap();
        assert!(format!("{}", PageStoreFile::open(&bad_magic).err().unwrap()).contains("magic"));
        let truncated = dir.join("short.f3dpages");
        std::fs::write(&truncated, &bytes[..bytes.len() - 8]).unwrap();
        assert!(format!("{}", PageStoreFile::open(&truncated).err().unwrap()).contains("index"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn decoded_pages_reorder_records_into_leaf_runs_with_inverse_covariance() {
        let cloud = Arc::new(test_cloud(700, 1));
        let source = SplatCloudSource::new(cloud.clone(), 512);
        let payload = source.load(0).unwrap();
        let decoded = DecodedPage::from_payload(0, &payload, 0.1, true).unwrap();
        assert_eq!(decoded.count, 512);
        assert_eq!(decoded.splats.len(), 512);
        assert_eq!(decoded.inv_cov.len(), 512);
        assert!(decoded.nodes.len() <= super::super::bvh::blas_node_bound(512));
        for node in decoded.nodes.iter().filter(|node| node.is_leaf()) {
            for k in 0..node.leaf_count() {
                let splat = decoded.splats[(node.a + k) as usize];
                // The centre of every record lies inside its leaf box.
                for axis in 0..3 {
                    assert!(splat.pos_opacity[axis] >= node.aabb_min[axis]);
                    assert!(splat.pos_opacity[axis] <= node.aabb_max[axis]);
                }
                assert_eq!(splat.sh0_degree[3], 1.0);
            }
        }
        // The packed inverse covariance matches the splat it travels with.
        let PagePayload::Splats(chunk) = &payload else { unreachable!() };
        let first = decoded.splats[0];
        let source_index = (0..chunk.len())
            .find(|&i| chunk.positions[i] == [first.pos_opacity[0], first.pos_opacity[1], first.pos_opacity[2]])
            .unwrap();
        let m = inverse_covariance(chunk.scales[source_index], chunk.rotations[source_index]);
        assert_eq!(decoded.inv_cov[0].m0, [m[0], m[1], m[2], m[3]]);
        assert_eq!(decoded.inv_cov[0].m1, [m[4], m[5], 0.0, 0.0]);
        assert!(DecodedPage::from_payload(
            0,
            &PagePayload::Points {
                positions: vec![[0.0; 3]],
                colors: vec![[1.0; 3]]
            },
            0.0,
            true
        )
        .is_err());
    }

    #[test]
    fn residency_pool_evicts_least_recently_used_and_never_the_current_frame() {
        let kinds = vec![PageKind::Splat; 6];
        let mut residency = Residency::new(kinds, 3, 0);
        for (frame, page) in [(1u64, 0u32), (2, 1), (3, 2)] {
            let admission = residency.admit(page, frame).unwrap();
            assert_eq!(admission.evicted, None);
        }
        assert_eq!(residency.stats.resident_pages, 3);
        // Page 0 is the oldest; touching it protects it and page 1 goes.
        residency.touch(0, 4);
        let admission = residency.admit(3, 5).unwrap();
        assert_eq!(admission.evicted, Some(1));
        assert_eq!(residency.slot(1), None);
        assert_eq!(residency.slot(3), Some(admission.slot));
        // Pages used in the current frame are pinned: a fourth page in the
        // same frame cannot be admitted once all three slots are pinned.
        residency.touch(0, 6);
        residency.touch(2, 6);
        residency.touch(3, 6);
        assert_eq!(residency.admit(4, 6), None);
        assert_eq!(residency.stats.evictions, 1);
        assert_eq!(residency.stats.peak_resident_pages, 3);
        // A later frame can evict again.
        assert!(residency.admit(4, 7).is_some());
        // Point pages have no slots in this pool.
        let mut points = Residency::new(vec![PageKind::Points], 1, 0);
        assert_eq!(points.admit(0, 1), None);
    }

    #[test]
    fn loader_streams_pages_from_worker_threads() {
        let cloud = Arc::new(test_cloud(2000, 0));
        let source: Arc<dyn PageSource> = Arc::new(SplatCloudSource::new(cloud, 256));
        let pages = source.page_count();
        let mut loader = PageLoader::new(Arc::new(vec![source]), 0.1, true, 3, None);
        for page in 0..pages {
            loader.request(page, (0, page));
        }
        assert_eq!(loader.in_flight(), pages as usize);
        let mut seen = vec![false; pages as usize];
        let mut total = 0u32;
        while let Some((page, result)) = loader.recv() {
            let decoded = result.unwrap();
            assert_eq!(decoded.page, page);
            assert!(!seen[page as usize]);
            seen[page as usize] = true;
            total += decoded.count;
        }
        assert_eq!(total, 2000);
        assert!(seen.iter().all(|s| *s));
        assert_eq!(loader.in_flight(), 0);
        assert!(loader.try_recv().is_none());
    }

    #[test]
    fn pagination_covers_every_index_once_with_full_pages() {
        let positions: Vec<[f32; 3]> = (0..1000u32)
            .map(|i| [hash_unit(i), hash_unit(i ^ 0xabcd), hash_unit(i ^ 0x1234)])
            .collect();
        let pages = paginate(&positions, 128);
        assert_eq!(pages.len(), 8);
        assert_eq!(pages.iter().filter(|page| page.len() == 128).count(), 7);
        let mut seen = vec![false; 1000];
        for page in &pages {
            assert!(page.len() <= 128);
            for &i in page {
                assert!(!seen[i as usize]);
                seen[i as usize] = true;
            }
        }
        assert!(seen.iter().all(|s| *s));
        assert_eq!(pages_for(1_000_000_000, 4096), 244_141);
    }
}
