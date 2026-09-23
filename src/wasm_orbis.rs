//! Dependency-free raw wasm ABI for browser-owned asynchronous height I/O.
//!
//! Handles and payload allocations are opaque numeric registry keys. Browser
//! code may inspect a registered payload's linear-memory view, but can never
//! turn an arbitrary pointer into Rust ownership.

use crate::terrain::page_table::SerializedPageTable;
use crate::terrain::page_table::{AsyncTileLoader, CoalescePolicy, HeightReader, RequestTicket};
use crate::terrain::stream::{EvictionPolicy, HeightPageResidency};
use crate::terrain::tiling::{TileBounds, TileData, TileId};
use glam::Vec2;
use std::cell::RefCell;
use std::collections::HashMap;

const MAX_TILE_RESOLUTION: u32 = 2048;
const MAX_IN_FLIGHT: u32 = 1024;
const PAYLOAD_BYTES_PER_SAMPLE: u64 = 5;
const MAX_PAYLOAD_BYTES: u64 = 20 * 1024 * 1024;
const MAX_LIVE_PAYLOAD_BYTES: u64 = 64 * 1024 * 1024;
const MAX_LIVE_PAYLOADS: usize = 1024;
const MAX_LOADERS: usize = 64;

struct BrowserReader;
impl HeightReader for BrowserReader {
    fn read_result(
        &self,
        _root_bounds: &TileBounds,
        _tile_size: Vec2,
        _tile_id: TileId,
        _width: u32,
        _height: u32,
    ) -> Result<Vec<f32>, String> {
        Err("browser height requests must be completed through the ORBIS ABI".to_string())
    }
}

struct WasmLoaderFacade {
    loader: AsyncTileLoader,
    tile_resolution: u32,
    last_request: Option<RequestTicket>,
    last_cancellation: Option<RequestTicket>,
    last_error: Option<(RequestTicket, u32)>,
    last_completion: Option<TileData>,
    residency: HeightPageResidency,
    serialized_page_table: Option<SerializedPageTable>,
    submitted_payload_bytes: HashMap<RequestTicket, u64>,
    last_completion_bytes: u64,
}

impl WasmLoaderFacade {
    fn request(&mut self, requested: TileId) -> bool {
        let _ = self.residency.touch_resolved(requested);
        let mut candidate = requested;
        if self.residency.slot_of(requested).is_none() {
            let mut cursor = requested;
            while let Some(parent) = cursor.parent() {
                if self.residency.slot_of(parent).is_none() {
                    candidate = parent;
                }
                cursor = parent;
            }
        }
        !self
            .submitted_payload_bytes
            .keys()
            .any(|ticket| ticket.tile_id == candidate)
            && self.loader.request(candidate)
    }
}

struct PayloadAllocation {
    loader: u32,
    ticket: RequestTicket,
    width: u32,
    height: u32,
    heights: Box<[f32]>,
    coverage: Box<[u8]>,
    bytes: u64,
}

#[derive(Default)]
struct Registry {
    next_loader: u32,
    next_allocation: u32,
    loaders: HashMap<u32, WasmLoaderFacade>,
    allocations: HashMap<u32, PayloadAllocation>,
    allocation_by_ticket: HashMap<(u32, RequestTicket), u32>,
    live_payload_bytes: u64,
    peak_payload_bytes: u64,
    live_payload_count: usize,
}

impl Registry {
    fn next_free_loader(&mut self) -> Option<u32> {
        self.next_loader = self.next_loader.checked_add(1)?.max(1);
        Some(self.next_loader)
    }

    fn next_free_allocation(&mut self) -> Option<u32> {
        self.next_allocation = self.next_allocation.checked_add(1)?.max(1);
        Some(self.next_allocation)
    }

    fn take_registered(&mut self, allocation_id: u32, release: bool) -> Option<PayloadAllocation> {
        let allocation = self.allocations.remove(&allocation_id)?;
        self.allocation_by_ticket
            .remove(&(allocation.loader, allocation.ticket));
        if release {
            self.live_payload_bytes = self.live_payload_bytes.saturating_sub(allocation.bytes);
            self.live_payload_count = self.live_payload_count.saturating_sub(1);
        }
        Some(allocation)
    }

    fn release_loader_allocations(&mut self, handle: u32) {
        let ids: Vec<_> = self
            .allocations
            .iter()
            .filter_map(|(id, allocation)| (allocation.loader == handle).then_some(*id))
            .collect();
        for id in ids {
            let _ = self.take_registered(id, true);
        }
    }

    fn release_payload_accounting(&mut self, bytes: u64) {
        if bytes != 0 {
            self.live_payload_bytes = self.live_payload_bytes.saturating_sub(bytes);
            self.live_payload_count = self.live_payload_count.saturating_sub(1);
        }
    }
}

thread_local! {
    static REGISTRY: RefCell<Registry> = RefCell::new(Registry::default());
}

fn ticket_generation(lo: u32, hi: u32) -> u64 {
    u64::from(lo) | (u64::from(hi) << 32)
}

fn valid_tile(lod: u32, x: u32, y: u32) -> Option<TileId> {
    crate::terrain::planetary_tiles::validate_global_tile_id(lod, x, y).ok()
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_global_tile_bound(lod: u32, x: u32, y: u32, field: u32) -> f64 {
    crate::terrain::planetary_tiles::global_tile_lonlat_bounds(TileId::new(lod, x, y))
        .ok()
        .and_then(|bounds| match field {
            0 => Some(bounds.0),
            1 => Some(bounds.1),
            2 => Some(bounds.2),
            3 => Some(bounds.3),
            _ => None,
        })
        .unwrap_or(f64::NAN)
}

fn ticket_field(ticket: Option<RequestTicket>, field: u32) -> u32 {
    let Some(ticket) = ticket else { return 0 };
    match field {
        0 => ticket.tile_id.lod,
        1 => ticket.tile_id.x,
        2 => ticket.tile_id.y,
        3 => ticket.generation as u32,
        4 => (ticket.generation >> 32) as u32,
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_init(max_in_flight: u32, tile_resolution: u32) -> u32 {
    let Some(residency_capacity) = max_in_flight.checked_add(1) else {
        return 0;
    };
    if max_in_flight == 0
        || max_in_flight > MAX_IN_FLIGHT
        || tile_resolution == 0
        || tile_resolution > MAX_TILE_RESOLUTION
        || u64::from(tile_resolution)
            .checked_mul(u64::from(tile_resolution))
            .and_then(|count| count.checked_mul(PAYLOAD_BYTES_PER_SAMPLE))
            .is_none_or(|bytes| bytes > MAX_PAYLOAD_BYTES)
    {
        return 0;
    }
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        if registry.loaders.len() >= MAX_LOADERS {
            return 0;
        }
        let Some(handle) = registry.next_free_loader() else {
            return 0;
        };
        registry.loaders.insert(
            handle,
            WasmLoaderFacade {
                loader: AsyncTileLoader::new_with_reader(
                    TileBounds::new(Vec2::ZERO, Vec2::ONE),
                    Vec2::ONE,
                    tile_resolution,
                    max_in_flight as usize,
                    1,
                    std::sync::Arc::new(BrowserReader),
                    CoalescePolicy::PreferCoarse,
                ),
                tile_resolution,
                last_request: None,
                last_cancellation: None,
                last_error: None,
                last_completion: None,
                // Request concurrency counts leaves; the pinned root owns a
                // distinct slot and must never make a valid leaf unplaceable.
                residency: HeightPageResidency::new(residency_capacity, 1, None)
                    .expect("validated non-zero wasm residency"),
                serialized_page_table: None,
                submitted_payload_bytes: HashMap::new(),
                last_completion_bytes: 0,
            },
        );
        handle
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_destroy(handle: u32) -> u32 {
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let Some(facade) = registry.loaders.remove(&handle) else {
            return 0;
        };
        registry.release_loader_allocations(handle);
        for bytes in facade.submitted_payload_bytes.into_values() {
            registry.release_payload_accounting(bytes);
        }
        registry.release_payload_accounting(facade.last_completion_bytes);
        1
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_request(handle: u32, lod: u32, x: u32, y: u32) -> u32 {
    let Some(id) = valid_tile(lod, x, y) else {
        return 0;
    };
    REGISTRY.with(|registry| {
        registry
            .borrow_mut()
            .loaders
            .get_mut(&handle)
            .is_some_and(|facade| facade.request(id)) as u32
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_cancel(handle: u32, lod: u32, x: u32, y: u32) -> u32 {
    let Some(id) = valid_tile(lod, x, y) else {
        return 0;
    };
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let cancelled = registry
            .loaders
            .get_mut(&handle)
            .map(|facade| facade.loader.cancel(&[id]))
            .unwrap_or(0);
        if cancelled != 0 {
            if let Some(facade) = registry.loaders.get_mut(&handle) {
                if facade
                    .last_request
                    .is_some_and(|ticket| ticket.tile_id == id)
                {
                    facade.last_request = None;
                }
            }
            let allocation_ids: Vec<_> = registry
                .allocations
                .iter()
                .filter_map(|(allocation_id, allocation)| {
                    (allocation.loader == handle && allocation.ticket.tile_id == id)
                        .then_some(*allocation_id)
                })
                .collect();
            for allocation_id in allocation_ids {
                let _ = registry.take_registered(allocation_id, true);
            }
        }
        cancelled as u32
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_poll_request(handle: u32) -> u32 {
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let Some(facade) = registry.loaders.get_mut(&handle) else {
            return 0;
        };
        if facade
            .last_request
            .is_some_and(|ticket| facade.loader.is_wasm_running(ticket))
        {
            return 1;
        }
        facade.last_request = None;
        facade.last_request = facade.loader.next_wasm_request();
        facade.last_request.is_some() as u32
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_poll_cancellation(handle: u32) -> u32 {
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let Some(facade) = registry.loaders.get_mut(&handle) else {
            return 0;
        };
        facade.last_cancellation = facade.loader.next_wasm_cancellation();
        facade.last_cancellation.is_some() as u32
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_last_request(handle: u32, field: u32) -> u32 {
    REGISTRY.with(|registry| {
        registry
            .borrow()
            .loaders
            .get(&handle)
            .map(|facade| ticket_field(facade.last_request, field))
            .unwrap_or(0)
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_last_cancellation(handle: u32, field: u32) -> u32 {
    REGISTRY.with(|registry| {
        registry
            .borrow()
            .loaders
            .get(&handle)
            .map(|facade| ticket_field(facade.last_cancellation, field))
            .unwrap_or(0)
    })
}

/// Allocate exactly one configured tile payload for the currently polled
/// ticket. Returns an opaque allocation id, not a pointer.
#[no_mangle]
pub extern "C" fn forge3d_orbis_alloc_heights(
    handle: u32,
    lod: u32,
    x: u32,
    y: u32,
    generation_lo: u32,
    generation_hi: u32,
    width: u32,
    height: u32,
) -> u32 {
    let Some(tile_id) = valid_tile(lod, x, y) else {
        return 0;
    };
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let ticket = RequestTicket {
            tile_id,
            generation: ticket_generation(generation_lo, generation_hi),
        };
        let Some(facade) = registry.loaders.get(&handle) else {
            return 0;
        };
        let count = u64::from(width).checked_mul(u64::from(height));
        let bytes = count.and_then(|value| value.checked_mul(PAYLOAD_BYTES_PER_SAMPLE));
        if facade.last_request != Some(ticket)
            || !facade.loader.is_wasm_running(ticket)
            || width != facade.tile_resolution
            || height != facade.tile_resolution
            || bytes.is_none_or(|bytes| bytes > MAX_PAYLOAD_BYTES)
            || registry
                .allocation_by_ticket
                .contains_key(&(handle, ticket))
            || registry.live_payload_count >= MAX_LIVE_PAYLOADS
            || registry
                .live_payload_bytes
                .checked_add(bytes.unwrap_or(0))
                .is_none_or(|total| total > MAX_LIVE_PAYLOAD_BYTES)
        {
            return 0;
        }
        let Some(allocation_id) = registry.next_free_allocation() else {
            return 0;
        };
        let bytes = bytes.unwrap_or(0);
        registry.allocations.insert(
            allocation_id,
            PayloadAllocation {
                loader: handle,
                ticket,
                width,
                height,
                heights: vec![0.0; count.unwrap_or(0) as usize].into_boxed_slice(),
                // Invalid by default: browser code must explicitly describe
                // which samples replace the ordinary overview.
                coverage: vec![0; count.unwrap_or(0) as usize].into_boxed_slice(),
                bytes,
            },
        );
        registry
            .allocation_by_ticket
            .insert((handle, ticket), allocation_id);
        if let Some(facade) = registry.loaders.get_mut(&handle) {
            facade.last_request = None;
        }
        registry.live_payload_bytes += bytes;
        registry.live_payload_count += 1;
        registry.peak_payload_bytes = registry.peak_payload_bytes.max(registry.live_payload_bytes);
        allocation_id
    })
}

/// Return a temporary linear-memory view for a registered allocation.
#[no_mangle]
pub extern "C" fn forge3d_orbis_heights_pointer(handle: u32, allocation_id: u32) -> u32 {
    REGISTRY.with(|registry| {
        registry
            .borrow_mut()
            .allocations
            .get_mut(&allocation_id)
            .filter(|allocation| allocation.loader == handle)
            .map(|allocation| allocation.heights.as_mut_ptr() as usize as u32)
            .unwrap_or(0)
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_coverage_pointer(handle: u32, allocation_id: u32) -> u32 {
    REGISTRY.with(|registry| {
        registry
            .borrow_mut()
            .allocations
            .get_mut(&allocation_id)
            .filter(|allocation| allocation.loader == handle)
            .map(|allocation| allocation.coverage.as_mut_ptr() as usize as u32)
            .unwrap_or(0)
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_free_heights(handle: u32, allocation_id: u32) -> u32 {
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        if registry
            .allocations
            .get(&allocation_id)
            .is_none_or(|allocation| allocation.loader != handle)
        {
            return 0;
        }
        let _ = registry.take_registered(allocation_id, true);
        1
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_complete(handle: u32, allocation_id: u32) -> u32 {
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        if registry
            .allocations
            .get(&allocation_id)
            .is_none_or(|allocation| allocation.loader != handle)
        {
            return 0;
        }
        // Removal happens before ticket validation so stale/rejected payloads
        // are reclaimed exactly once.
        let allocation = registry.take_registered(allocation_id, false).unwrap();
        let bytes = allocation.bytes;
        let tile_id = allocation.ticket.tile_id;
        let Some(facade) = registry.loaders.get_mut(&handle) else {
            registry.release_payload_accounting(bytes);
            return 0;
        };
        let accepted = facade.loader.complete_wasm(
            allocation.ticket,
            TileData::new_covered(
                tile_id,
                allocation.heights.into_vec(),
                allocation.coverage.into_vec(),
                allocation.width,
                allocation.height,
            ),
        );
        if accepted {
            facade.last_request = None;
            facade
                .submitted_payload_bytes
                .insert(allocation.ticket, bytes);
            1
        } else {
            registry.release_payload_accounting(bytes);
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_ack_cancellation(
    handle: u32,
    lod: u32,
    x: u32,
    y: u32,
    generation_lo: u32,
    generation_hi: u32,
) -> u32 {
    let Some(tile_id) = valid_tile(lod, x, y) else {
        return 0;
    };
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let ticket = RequestTicket {
            tile_id,
            generation: ticket_generation(generation_lo, generation_hi),
        };
        let acknowledged = registry
            .loaders
            .get_mut(&handle)
            .is_some_and(|facade| facade.loader.acknowledge_wasm_cancellation(ticket));
        if acknowledged {
            if let Some(facade) = registry.loaders.get_mut(&handle) {
                if facade.last_cancellation == Some(ticket) {
                    facade.last_cancellation = None;
                }
                if facade.last_request == Some(ticket) {
                    facade.last_request = None;
                }
            }
            if let Some(allocation_id) = registry
                .allocation_by_ticket
                .get(&(handle, ticket))
                .copied()
            {
                let _ = registry.take_registered(allocation_id, true);
            }
        }
        acknowledged as u32
    })
}

/// Publish a browser fetch/decode failure for the exact Running ticket.
/// Capacity stays occupied until JavaScript polls and acknowledges the error.
#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_error(
    handle: u32,
    lod: u32,
    x: u32,
    y: u32,
    generation_lo: u32,
    generation_hi: u32,
    error_code: u32,
) -> u32 {
    let Some(tile_id) = valid_tile(lod, x, y) else {
        return 0;
    };
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let ticket = RequestTicket {
            tile_id,
            generation: ticket_generation(generation_lo, generation_hi),
        };
        let accepted = registry
            .loaders
            .get_mut(&handle)
            .is_some_and(|facade| facade.loader.fail_wasm(ticket, error_code));
        if !accepted {
            return 0;
        }
        if let Some(facade) = registry.loaders.get_mut(&handle) {
            if facade.last_request == Some(ticket) {
                facade.last_request = None;
            }
        }
        if let Some(allocation_id) = registry
            .allocation_by_ticket
            .get(&(handle, ticket))
            .copied()
        {
            let _ = registry.take_registered(allocation_id, true);
        }
        1
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_poll_error(handle: u32) -> u32 {
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let Some(facade) = registry.loaders.get_mut(&handle) else {
            return 0;
        };
        if facade.last_error.is_none() {
            facade.last_error = facade.loader.next_wasm_error();
        }
        facade.last_error.is_some() as u32
    })
}

/// Error fields: lod/x/y/generation-low/generation-high/error-code.
#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_last_error(handle: u32, field: u32) -> u32 {
    REGISTRY.with(|registry| {
        registry
            .borrow()
            .loaders
            .get(&handle)
            .and_then(|facade| facade.last_error)
            .map(|(ticket, code)| {
                if field == 5 {
                    code
                } else {
                    ticket_field(Some(ticket), field)
                }
            })
            .unwrap_or(0)
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_ack_error(
    handle: u32,
    lod: u32,
    x: u32,
    y: u32,
    generation_lo: u32,
    generation_hi: u32,
) -> u32 {
    let Some(tile_id) = valid_tile(lod, x, y) else {
        return 0;
    };
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let ticket = RequestTicket {
            tile_id,
            generation: ticket_generation(generation_lo, generation_hi),
        };
        let Some(facade) = registry.loaders.get_mut(&handle) else {
            return 0;
        };
        if facade
            .last_error
            .is_none_or(|(current, _)| current != ticket)
            || !facade.loader.acknowledge_wasm_error(ticket)
        {
            return 0;
        }
        facade.last_error = None;
        1
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_drain_completion(handle: u32) -> u32 {
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let Some(mut facade) = registry.loaders.remove(&handle) else {
            return 0;
        };
        let Some((ticket, tile)) = facade.loader.drain_wasm_completed(1).pop() else {
            // The existing completion is still readable and therefore still
            // owns its aggregate accounting.
            registry.loaders.insert(handle, facade);
            return 0;
        };
        let Some(new_bytes) = facade.submitted_payload_bytes.remove(&ticket) else {
            debug_assert!(false, "completed wasm ticket lost payload accounting");
            registry.loaders.insert(handle, facade);
            return 0;
        };
        if facade
            .residency
            .place(tile.tile_id, EvictionPolicy::PreserveRoot)
            .is_err()
        {
            // The loader terminal was genuine, but failed residency is never
            // exposed as a successful/readable completion.
            registry.release_payload_accounting(new_bytes);
            registry.loaders.insert(handle, facade);
            return 0;
        }
        // Replacement is the only implicit release: drop/account the old
        // readable completion atomically with installing the new one.
        registry.release_payload_accounting(facade.last_completion_bytes);
        facade.last_completion = None;
        facade.last_completion_bytes = new_bytes;
        let entries = facade.residency.entries();
        let target_lod = entries.iter().map(|(id, _)| id.lod).max().unwrap_or(0);
        facade.serialized_page_table = SerializedPageTable::from_entries(
            &entries,
            facade.residency.capacity(),
            target_lod,
            facade.tile_resolution,
            (
                facade.tile_resolution * facade.residency.capacity() as u32,
                facade.tile_resolution,
            ),
            facade.residency.capacity() as u32,
        )
        .ok();
        facade.last_completion = Some(tile);
        registry.loaders.insert(handle, facade);
        1
    })
}

/// Drop the most recently drained payload once JavaScript no longer needs to
/// inspect it. This is idempotent and never accepts a foreign loader handle.
#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_release_completion(handle: u32) -> u32 {
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let Some(mut facade) = registry.loaders.remove(&handle) else {
            return 0;
        };
        let had_completion = facade.last_completion.take().is_some();
        let bytes = std::mem::take(&mut facade.last_completion_bytes);
        registry.release_payload_accounting(bytes);
        registry.loaders.insert(handle, facade);
        had_completion as u32
    })
}

/// Aggregate payload accounting for automated ABI conformance tests.
/// Fields: 0 live allocations, 1 live bytes low, 2 live bytes high,
/// 3 peak bytes low, 4 peak bytes high, 5 loader count.
#[no_mangle]
pub extern "C" fn forge3d_orbis_registry_stat(field: u32) -> u32 {
    REGISTRY.with(|registry| {
        let registry = registry.borrow();
        match field {
            0 => registry.live_payload_count as u32,
            1 => registry.live_payload_bytes as u32,
            2 => (registry.live_payload_bytes >> 32) as u32,
            3 => registry.peak_payload_bytes as u32,
            4 => (registry.peak_payload_bytes >> 32) as u32,
            5 => registry.loaders.len() as u32,
            _ => 0,
        }
    })
}

/// Resolve a requested tile through the production sparse page-table ancestor
/// walk. Fields 0..5 are lod/x/y/sx/sy/slot.
#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_resolve_page(
    handle: u32,
    lod: u32,
    x: u32,
    y: u32,
    field: u32,
) -> u32 {
    let Some(tile_id) = valid_tile(lod, x, y) else {
        return u32::MAX;
    };
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let Some(facade) = registry.loaders.get_mut(&handle) else {
            return u32::MAX;
        };
        let _ = facade.residency.touch_resolved(tile_id);
        let Some(entry) = facade
            .serialized_page_table
            .as_ref()
            .and_then(|table| table.resolve_nearest_resident_ancestor(tile_id))
        else {
            return u32::MAX;
        };
        match field {
            0 => entry.lod,
            1 => entry.x,
            2 => entry.y,
            3 => entry.sx,
            4 => entry.sy,
            5 => entry.slot,
            _ => u32::MAX,
        }
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_completion_meta(handle: u32, field: u32) -> u32 {
    REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let Some(tile) = registry
            .loaders
            .get(&handle)
            .and_then(|facade| facade.last_completion.as_ref())
        else {
            return 0;
        };
        match field {
            0 => tile.tile_id.lod,
            1 => tile.tile_id.x,
            2 => tile.tile_id.y,
            3 => tile.width,
            4 => tile.height,
            5 => tile.height_data.len() as u32,
            6 => tile.coverage_data.len() as u32,
            _ => 0,
        }
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_completion_coverage_sample(handle: u32, index: u32) -> u32 {
    REGISTRY.with(|registry| {
        registry
            .borrow()
            .loaders
            .get(&handle)
            .and_then(|facade| facade.last_completion.as_ref())
            .and_then(|tile| tile.coverage_data.get(index as usize))
            .copied()
            .map(u32::from)
            .unwrap_or(u32::MAX)
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_completion_sample(handle: u32, index: u32) -> f32 {
    REGISTRY.with(|registry| {
        registry
            .borrow()
            .loaders
            .get(&handle)
            .and_then(|facade| facade.last_completion.as_ref())
            .and_then(|tile| tile.height_data.get(index as usize))
            .copied()
            .unwrap_or(f32::NAN)
    })
}

#[no_mangle]
pub extern "C" fn forge3d_orbis_loader_pending(handle: u32) -> u32 {
    REGISTRY.with(|registry| {
        registry
            .borrow()
            .loaders
            .get(&handle)
            .map(|facade| facade.loader.stats().0 as u32)
            .unwrap_or(0)
    })
}
