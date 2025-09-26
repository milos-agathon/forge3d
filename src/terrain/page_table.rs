// E1: Page table & async IO scaffolding
// Lightweight GPU buffer storing tile->slot mappings for the current height mosaic.
// Also includes a simple async tile request queue scaffold (main-thread drained).

use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Queue};

use crate::terrain::stream::HeightMosaic;
use crate::terrain::tiling::{TileBounds, TileData, TileId};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PageTableEntry {
    // tile id
    pub lod: u32,
    pub x: u32,
    pub y: u32,
    pub _pad0: u32,
    // atlas slot coordinates and linearized index
    pub sx: u32,
    pub sy: u32,
    pub slot: u32,
    pub _pad1: u32,
}

// E1c: Background async tile loader (request -> TileData)
use glam::Vec2;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread;

/// Return true if `child` is a strict descendant of `ancestor` in the quadtree.
fn is_descendant_of(child: TileId, ancestor: TileId) -> bool {
    if child.lod <= ancestor.lod {
        return false;
    }
    let shift = child.lod - ancestor.lod;
    (child.x >> shift) == ancestor.x && (child.y >> shift) == ancestor.y
}

#[derive(Clone, Copy, Debug)]
pub enum CoalescePolicy {
    PreferCoarse,
    PreferFine,
}

impl Default for CoalescePolicy {
    fn default() -> Self {
        Self::PreferCoarse
    }
}

pub struct AsyncTileLoader {
    req_tx: SyncSender<TileId>,
    done_rx: Receiver<TileData>,
    dispatcher: thread::JoinHandle<()>,
    workers: Vec<thread::JoinHandle<()>>,
    worker_txs: Vec<Sender<TileId>>,
    pool_size: usize,
    // E1e: dedup + backpressure
    pending: Mutex<HashSet<TileId>>, // TileIds currently in-flight
    max_in_flight: usize,            // Backpressure limit
    // E1e/E1f: cancellation
    cancelled: Mutex<HashSet<TileId>>, // Requests canceled by the main thread; results will be dropped
    policy: CoalescePolicy,
    // E1m: counters
    c_requests: AtomicUsize,          // attempts
    c_enqueued: AtomicUsize,          // successfully sent to dispatcher
    c_dropped_by_policy: AtomicUsize, // not enqueued due to coalescing
    c_canceled: AtomicUsize,          // cancel() marked
    c_send_fail: AtomicUsize,         // channel try_send failed
    c_completed: AtomicUsize,         // results delivered by drain_completed
}

impl AsyncTileLoader {
    pub fn new(
        root_bounds: TileBounds,
        tile_size: Vec2,
        tile_resolution: u32,
        max_in_flight: usize,
        pool_size: usize,
    ) -> Self {
        Self::new_with_reader(
            root_bounds,
            tile_size,
            tile_resolution,
            max_in_flight,
            pool_size,
            Arc::new(SyntheticHeightReader),
            CoalescePolicy::PreferCoarse,
        )
    }

    pub fn new_with_reader(
        root_bounds: TileBounds,
        tile_size: Vec2,
        tile_resolution: u32,
        max_in_flight: usize,
        pool_size: usize,
        reader: Arc<dyn HeightReader>,
        policy: CoalescePolicy,
    ) -> Self {
        let (req_tx, req_rx) = mpsc::sync_channel::<TileId>(max_in_flight.max(1));
        let (done_tx, done_rx) = mpsc::channel::<TileData>();

        // Worker pool
        let mut worker_txs = Vec::new();
        let mut workers = Vec::new();
        let n = pool_size.max(1);
        for _ in 0..n {
            let (wtx, wrx) = mpsc::channel::<TileId>();
            worker_txs.push(wtx.clone());
            let done_tx_clone = done_tx.clone();
            let rb = root_bounds.clone();
            let ts = tile_size;
            let reader = reader.clone();
            let handle = thread::spawn(move || {
                while let Ok(id) = wrx.recv() {
                    let w = tile_resolution;
                    let h = tile_resolution;
                    let heights = reader.read(&rb, ts, id, w, h);
                    let td = TileData::new(id, heights, w, h);
                    let _ = done_tx_clone.send(td);
                }
            });
            workers.push(handle);
        }

        // Dispatcher
        let worker_txs_for_dispatcher: Vec<Sender<TileId>> = worker_txs.iter().cloned().collect();
        let dispatcher = thread::spawn(move || {
            let mut idx: usize = 0;
            while let Ok(id) = req_rx.recv() {
                if worker_txs_for_dispatcher.is_empty() {
                    break;
                }
                let _ = worker_txs_for_dispatcher[idx % worker_txs_for_dispatcher.len()].send(id);
                idx = idx.wrapping_add(1);
            }
        });

        Self {
            req_tx,
            done_rx,
            dispatcher,
            workers,
            worker_txs,
            pool_size: n,
            pending: Mutex::new(HashSet::new()),
            max_in_flight: max_in_flight.max(1),
            cancelled: Mutex::new(HashSet::new()),
            policy,
            c_requests: AtomicUsize::new(0),
            c_enqueued: AtomicUsize::new(0),
            c_dropped_by_policy: AtomicUsize::new(0),
            c_canceled: AtomicUsize::new(0),
            c_send_fail: AtomicUsize::new(0),
            c_completed: AtomicUsize::new(0),
        }
    }

    pub fn request(&self, id: TileId) -> bool {
        // E1e: deduplicate and apply backpressure + E1k: LOD coalescing (policy-configurable)
        if let Ok(mut pend) = self.pending.lock() {
            self.c_requests.fetch_add(1, Ordering::Relaxed);
            // If previously canceled, clear that state when re-requested
            if let Ok(mut can) = self.cancelled.lock() {
                can.remove(&id);
            }
            match self.policy {
                CoalescePolicy::PreferCoarse => {
                    // Drop child if any ancestor pending
                    let mut p = id;
                    while let Some(parent) = p.parent() {
                        if pend.contains(&parent) {
                            self.c_dropped_by_policy.fetch_add(1, Ordering::Relaxed);
                            return false;
                        }
                        p = parent;
                    }
                    if pend.contains(&id) {
                        return false;
                    }
                    if pend.len() >= self.max_in_flight {
                        return false;
                    }
                    // Insert and cancel descendants
                    pend.insert(id);
                    let mut to_cancel: Vec<TileId> = Vec::new();
                    for &d in pend.iter() {
                        if d != id && is_descendant_of(d, id) {
                            to_cancel.push(d);
                        }
                    }
                    if self.req_tx.try_send(id).is_err() {
                        pend.remove(&id);
                        self.c_send_fail.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                    self.c_enqueued.fetch_add(1, Ordering::Relaxed);
                    if !to_cancel.is_empty() {
                        if let Ok(mut can) = self.cancelled.lock() {
                            for d in to_cancel.iter() {
                                pend.remove(d);
                                can.insert(*d);
                            }
                            self.c_canceled
                                .fetch_add(to_cancel.len(), Ordering::Relaxed);
                        }
                    }
                    true
                }
                CoalescePolicy::PreferFine => {
                    // If any descendant pending, skip this coarser request
                    for &d in pend.iter() {
                        if is_descendant_of(d, id) {
                            self.c_dropped_by_policy.fetch_add(1, Ordering::Relaxed);
                            return false;
                        }
                    }
                    if pend.contains(&id) {
                        return false;
                    }
                    if pend.len() >= self.max_in_flight {
                        return false;
                    }
                    // Tentatively insert
                    pend.insert(id);
                    // Try-send; if fails, roll back and keep ancestors
                    if self.req_tx.try_send(id).is_err() {
                        pend.remove(&id);
                        self.c_send_fail.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                    self.c_enqueued.fetch_add(1, Ordering::Relaxed);
                    // Cancel ancestors (post-send)
                    if let Ok(mut can) = self.cancelled.lock() {
                        let mut p = id;
                        while let Some(parent) = p.parent() {
                            if pend.remove(&parent) {
                                can.insert(parent);
                            }
                            p = parent;
                        }
                        self.c_canceled.fetch_add(1, Ordering::Relaxed);
                    }
                    true
                }
            }
        } else {
            false
        }
    }

    pub fn drain_completed(&self, limit: usize) -> Vec<TileData> {
        let mut out = Vec::new();
        while out.len() < limit {
            match self.done_rx.try_recv() {
                Ok(td) => {
                    // E1e: mark request as no longer pending
                    if let Ok(mut pend) = self.pending.lock() {
                        pend.remove(&td.tile_id);
                    }
                    // If canceled, drop the result silently
                    if let Ok(mut can) = self.cancelled.lock() {
                        if can.remove(&td.tile_id) {
                            continue;
                        }
                    }
                    self.c_completed.fetch_add(1, Ordering::Relaxed);
                    out.push(td)
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        out
    }

    pub fn stats(&self) -> (usize, usize, usize) {
        let pending_len = self.pending.lock().map(|p| p.len()).unwrap_or(0);
        (pending_len, self.max_in_flight, self.pool_size)
    }

    pub fn counters(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            self.c_requests.load(Ordering::Relaxed),
            self.c_enqueued.load(Ordering::Relaxed),
            self.c_dropped_by_policy.load(Ordering::Relaxed),
            self.c_canceled.load(Ordering::Relaxed),
            self.c_send_fail.load(Ordering::Relaxed),
            self.c_completed.load(Ordering::Relaxed),
        )
    }

    /// Cancel a list of pending/in-flight requests. Returns number of IDs marked canceled.
    pub fn cancel(&self, ids: &[TileId]) -> usize {
        let mut n = 0usize;
        if let Ok(mut pend) = self.pending.lock() {
            if let Ok(mut can) = self.cancelled.lock() {
                for id in ids.iter() {
                    if pend.remove(id) {
                        can.insert(*id);
                        n += 1;
                    }
                }
                if n > 0 {
                    self.c_canceled.fetch_add(n, Ordering::Relaxed);
                }
            }
        }
        n
    }
}

/// Simple file-backed overlay reader that expands a template like
/// "/data/tiles/{lod}/{x}/{y}.png" and returns RGBA8 bytes.
pub struct FileOverlayReader {
    template: String,
}

impl FileOverlayReader {
    pub fn new(template: String) -> Self {
        Self { template }
    }
    fn expand(&self, id: TileId) -> String {
        self.template
            .replace("{lod}", &id.lod.to_string())
            .replace("{x}", &id.x.to_string())
            .replace("{y}", &id.y.to_string())
    }
}

impl OverlayReader for FileOverlayReader {
    fn read(
        &self,
        _root_bounds: &TileBounds,
        _tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let path = self.expand(tile_id);
        match image::open(&path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                // If dimensions mismatch, rescale to expected size
                if rgba.width() != width || rgba.height() != height {
                    let resized = image::imageops::resize(
                        &rgba,
                        width,
                        height,
                        image::imageops::FilterType::Triangle,
                    );
                    resized.into_raw()
                } else {
                    rgba.into_raw()
                }
            }
            Err(_) => {
                // Fallback to solid transparent tile on failure
                vec![0u8; (width * height * 4) as usize]
            }
        }
    }
}

/// Simple file-backed height reader using PNG grayscale (8/16-bit) to f32 with scale/offset
pub struct FileHeightReader {
    template: String,
    scale: f32,
    offset: f32,
}

impl FileHeightReader {
    pub fn new(template: String, scale: f32, offset: f32) -> Self {
        Self {
            template,
            scale,
            offset,
        }
    }
    fn expand(&self, id: TileId) -> String {
        self.template
            .replace("{lod}", &id.lod.to_string())
            .replace("{x}", &id.x.to_string())
            .replace("{y}", &id.y.to_string())
    }
}

impl HeightReader for FileHeightReader {
    fn read(
        &self,
        _root_bounds: &TileBounds,
        _tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<f32> {
        let path = self.expand(tile_id);
        let expected = (width * height) as usize;
        match image::open(&path) {
            Ok(img) => {
                let gray = img.to_luma16();
                let (w, h) = gray.dimensions();
                if w != width || h != height {
                    let resized = image::imageops::resize(
                        &gray,
                        width,
                        height,
                        image::imageops::FilterType::Triangle,
                    );
                    let mut out = Vec::with_capacity(expected);
                    for &v16 in resized.as_raw().iter() {
                        let v = (v16 as f32) / 65535.0;
                        out.push(v * self.scale + self.offset);
                    }
                    out
                } else {
                    let mut out = Vec::with_capacity(expected);
                    for &v16 in gray.as_raw().iter() {
                        let v = (v16 as f32) / 65535.0;
                        out.push(v * self.scale + self.offset);
                    }
                    out
                }
            }
            Err(_) => vec![0.0f32; expected],
        }
    }
}

// ----- Pluggable Readers -----

pub trait HeightReader: Send + Sync + 'static {
    fn read(
        &self,
        root_bounds: &TileBounds,
        tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<f32>;
}

pub struct SyntheticHeightReader;
impl HeightReader for SyntheticHeightReader {
    fn read(
        &self,
        root_bounds: &TileBounds,
        tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<f32> {
        let mut heights = Vec::with_capacity((width * height) as usize);
        let bounds =
            crate::terrain::tiling::QuadTreeNode::calculate_bounds(root_bounds, tile_id, tile_size);
        for y in 0..height {
            for x in 0..width {
                let u = x as f32 / (width - 1) as f32;
                let v = y as f32 / (height - 1) as f32;
                let world_x = bounds.min.x + u * bounds.size().x;
                let world_y = bounds.min.y + v * bounds.size().y;
                let h = (world_x * 0.1).sin() * 10.0 + (world_y * 0.1).cos() * 10.0;
                heights.push(h);
            }
        }
        heights
    }
}

pub trait OverlayReader: Send + Sync + 'static {
    fn read(
        &self,
        root_bounds: &TileBounds,
        tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<u8>;
}

pub struct SyntheticOverlayReader;
impl OverlayReader for SyntheticOverlayReader {
    fn read(
        &self,
        root_bounds: &TileBounds,
        tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let mut px = Vec::with_capacity((width * height * 4) as usize);
        let bounds =
            crate::terrain::tiling::QuadTreeNode::calculate_bounds(root_bounds, tile_id, tile_size);
        for y in 0..height {
            for x in 0..width {
                let u = x as f32 / (width - 1) as f32;
                let v = y as f32 / (height - 1) as f32;
                let wx = bounds.min.x + u * bounds.size().x;
                let wy = bounds.min.y + v * bounds.size().y;
                let r = (((wx * 0.01).sin() * 0.5 + 0.5) * 255.0) as u8;
                let g = (((wy * 0.01).cos() * 0.5 + 0.5) * 255.0) as u8;
                let b = (((wx * 0.02 + wy * 0.02).sin() * 0.5 + 0.5) * 255.0) as u8;
                px.extend_from_slice(&[r, g, b, 255]);
            }
        }
        px
    }
}

// ---------------- Overlay async loader (RGBA8) ----------------

#[derive(Debug)]
pub struct OverlayTileData {
    pub tile_id: TileId,
    pub rgba_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub host_memory_size: u64,
}

impl OverlayTileData {
    pub fn new(tile_id: TileId, rgba_data: Vec<u8>, width: u32, height: u32) -> Self {
        let host_memory_size = rgba_data.len() as u64;
        Self {
            tile_id,
            rgba_data,
            width,
            height,
            host_memory_size,
        }
    }
}

pub struct AsyncOverlayLoader {
    req_tx: SyncSender<TileId>,
    done_rx: Receiver<OverlayTileData>,
    dispatcher: thread::JoinHandle<()>,
    workers: Vec<thread::JoinHandle<()>>,
    worker_txs: Vec<Sender<TileId>>,
    pool_size: usize,
    pending: Mutex<HashSet<TileId>>,
    cancelled: Mutex<HashSet<TileId>>,
    max_in_flight: usize,
    policy: CoalescePolicy,
    // E1m: counters
    c_requests: AtomicUsize,
    c_enqueued: AtomicUsize,
    c_dropped_by_policy: AtomicUsize,
    c_canceled: AtomicUsize,
    c_send_fail: AtomicUsize,
    c_completed: AtomicUsize,
}

impl AsyncOverlayLoader {
    pub fn new(
        root_bounds: TileBounds,
        tile_size: Vec2,
        tile_resolution: u32,
        max_in_flight: usize,
        pool_size: usize,
    ) -> Self {
        Self::new_with_reader(
            root_bounds,
            tile_size,
            tile_resolution,
            max_in_flight,
            pool_size,
            Arc::new(SyntheticOverlayReader),
            CoalescePolicy::PreferCoarse,
        )
    }

    pub fn new_with_reader(
        root_bounds: TileBounds,
        tile_size: Vec2,
        tile_resolution: u32,
        max_in_flight: usize,
        pool_size: usize,
        reader: Arc<dyn OverlayReader>,
        policy: CoalescePolicy,
    ) -> Self {
        let (req_tx, req_rx) = mpsc::sync_channel::<TileId>(max_in_flight.max(1));
        let (done_tx, done_rx) = mpsc::channel::<OverlayTileData>();

        // Workers
        let mut worker_txs = Vec::new();
        let mut workers = Vec::new();
        let n = pool_size.max(1);
        for _ in 0..n {
            let (wtx, wrx) = mpsc::channel::<TileId>();
            worker_txs.push(wtx.clone());
            let done_tx_clone = done_tx.clone();
            let rb = root_bounds.clone();
            let ts = tile_size;
            let reader = reader.clone();
            let handle = thread::spawn(move || {
                while let Ok(id) = wrx.recv() {
                    let w = tile_resolution;
                    let h = tile_resolution;
                    let rgba = reader.read(&rb, ts, id, w, h);
                    let td = OverlayTileData::new(id, rgba, w, h);
                    let _ = done_tx_clone.send(td);
                }
            });
            workers.push(handle);
        }

        // Dispatcher
        let worker_txs_for_dispatcher: Vec<Sender<TileId>> = worker_txs.iter().cloned().collect();
        let dispatcher = thread::spawn(move || {
            let mut idx: usize = 0;
            while let Ok(id) = req_rx.recv() {
                if worker_txs_for_dispatcher.is_empty() {
                    break;
                }
                let _ = worker_txs_for_dispatcher[idx % worker_txs_for_dispatcher.len()].send(id);
                idx = idx.wrapping_add(1);
            }
        });

        Self {
            req_tx,
            done_rx,
            dispatcher,
            workers,
            worker_txs,
            pool_size: n,
            pending: Mutex::new(HashSet::new()),
            cancelled: Mutex::new(HashSet::new()),
            max_in_flight: max_in_flight.max(1),
            policy,
            c_requests: AtomicUsize::new(0),
            c_enqueued: AtomicUsize::new(0),
            c_dropped_by_policy: AtomicUsize::new(0),
            c_canceled: AtomicUsize::new(0),
            c_send_fail: AtomicUsize::new(0),
            c_completed: AtomicUsize::new(0),
        }
    }

    pub fn request(&self, id: TileId) -> bool {
        if let Ok(mut pend) = self.pending.lock() {
            self.c_requests.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut can) = self.cancelled.lock() {
                can.remove(&id);
            }
            match self.policy {
                CoalescePolicy::PreferCoarse => {
                    let mut p = id;
                    while let Some(parent) = p.parent() {
                        if pend.contains(&parent) {
                            self.c_dropped_by_policy.fetch_add(1, Ordering::Relaxed);
                            return false;
                        }
                        p = parent;
                    }
                    if pend.contains(&id) {
                        return false;
                    }
                    if pend.len() >= self.max_in_flight {
                        return false;
                    }
                    pend.insert(id);
                    let mut to_cancel: Vec<TileId> = Vec::new();
                    for &d in pend.iter() {
                        if d != id && is_descendant_of(d, id) {
                            to_cancel.push(d);
                        }
                    }
                    if self.req_tx.try_send(id).is_err() {
                        pend.remove(&id);
                        self.c_send_fail.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                    self.c_enqueued.fetch_add(1, Ordering::Relaxed);
                    if !to_cancel.is_empty() {
                        if let Ok(mut can) = self.cancelled.lock() {
                            for d in to_cancel.iter() {
                                pend.remove(d);
                                can.insert(*d);
                            }
                            self.c_canceled
                                .fetch_add(to_cancel.len(), Ordering::Relaxed);
                        }
                    }
                    true
                }
                CoalescePolicy::PreferFine => {
                    for &d in pend.iter() {
                        if is_descendant_of(d, id) {
                            self.c_dropped_by_policy.fetch_add(1, Ordering::Relaxed);
                            return false;
                        }
                    }
                    if pend.contains(&id) {
                        return false;
                    }
                    if pend.len() >= self.max_in_flight {
                        return false;
                    }
                    pend.insert(id);
                    if self.req_tx.try_send(id).is_err() {
                        pend.remove(&id);
                        self.c_send_fail.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                    self.c_enqueued.fetch_add(1, Ordering::Relaxed);
                    if let Ok(mut can) = self.cancelled.lock() {
                        let mut p = id;
                        while let Some(parent) = p.parent() {
                            if pend.remove(&parent) {
                                can.insert(parent);
                            }
                            p = parent;
                        }
                        self.c_canceled.fetch_add(1, Ordering::Relaxed);
                    }
                    true
                }
            }
        } else {
            false
        }
    }

    pub fn cancel(&self, ids: &[TileId]) -> usize {
        let mut n = 0usize;
        if let Ok(mut pend) = self.pending.lock() {
            if let Ok(mut can) = self.cancelled.lock() {
                for id in ids.iter() {
                    if pend.remove(id) {
                        can.insert(*id);
                        n += 1;
                    }
                }
            }
        }
        n
    }

    pub fn drain_completed(&self, limit: usize) -> Vec<OverlayTileData> {
        let mut out = Vec::new();
        while out.len() < limit {
            match self.done_rx.try_recv() {
                Ok(td) => {
                    if let Ok(mut pend) = self.pending.lock() {
                        pend.remove(&td.tile_id);
                    }
                    if let Ok(mut can) = self.cancelled.lock() {
                        if can.remove(&td.tile_id) {
                            continue;
                        }
                    }
                    self.c_completed.fetch_add(1, Ordering::Relaxed);
                    out.push(td)
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        out
    }

    pub fn stats(&self) -> (usize, usize, usize) {
        let pending_len = self.pending.lock().map(|p| p.len()).unwrap_or(0);
        (pending_len, self.max_in_flight, self.pool_size)
    }
    pub fn counters(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            self.c_requests.load(Ordering::Relaxed),
            self.c_enqueued.load(Ordering::Relaxed),
            self.c_dropped_by_policy.load(Ordering::Relaxed),
            self.c_canceled.load(Ordering::Relaxed),
            self.c_send_fail.load(Ordering::Relaxed),
            self.c_completed.load(Ordering::Relaxed),
        )
    }
}

pub struct PageTable {
    pub buffer: Buffer,
    pub capacity: usize,
}

impl PageTable {
    pub fn new(device: &wgpu::Device, capacity: usize) -> Self {
        let entry_size = std::mem::size_of::<PageTableEntry>() as u64;
        let size = (capacity as u64).saturating_mul(entry_size).max(entry_size);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("terrain-page-table"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer, capacity }
    }

    pub fn sync_from_mosaic(&mut self, queue: &Queue, mosaic: &HeightMosaic) {
        // Build compact list (truncate if needed to fit capacity)
        let mut out: Vec<PageTableEntry> = Vec::with_capacity(self.capacity);
        let tiles_x = mosaic.config.tiles_x;
        for (id, (sx, sy)) in mosaic.entries().into_iter() {
            if out.len() >= self.capacity {
                break;
            }
            let slot = sy * tiles_x + sx;
            out.push(PageTableEntry {
                lod: id.lod,
                x: id.x,
                y: id.y,
                _pad0: 0,
                sx,
                sy,
                slot,
                _pad1: 0,
            });
        }
        if out.is_empty() {
            return;
        }
        let bytes = bytemuck::cast_slice(&out);
        queue.write_buffer(&self.buffer, 0, bytes);
    }
}

// --- Async IO scaffold ---------------------------------------------------------
use std::sync::mpsc::{Receiver, Sender};

pub struct AsyncTileQueue {
    tx: Sender<TileId>,
    rx: Receiver<TileId>,
}

impl AsyncTileQueue {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self { tx, rx }
    }
    pub fn sender(&self) -> Sender<TileId> {
        self.tx.clone()
    }
    pub fn drain(&self) -> Vec<TileId> {
        let mut v = Vec::new();
        while let Ok(id) = self.rx.try_recv() {
            v.push(id);
        }
        v
    }
}
