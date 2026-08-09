use glam::Vec2;
use std::collections::HashMap;
#[cfg(target_arch = "wasm32")]
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::thread;

use crate::terrain::tiling::{TileBounds, TileData, TileId};

use super::common::{is_descendant_of, CoalescePolicy};
#[cfg(not(target_arch = "wasm32"))]
use super::readers::HeightRead;
use super::readers::HeightReader;

/// Native height reads are I/O-bound; a small fixed ceiling prevents public
/// configuration from spawning an unbounded number of OS threads.
pub const MAX_LOADER_WORKERS: usize = 16;

pub(crate) fn bounded_loader_workers(pool_size: usize) -> usize {
    pool_size.max(1).min(MAX_LOADER_WORKERS)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RequestTicket {
    pub tile_id: TileId,
    pub generation: u64,
}

#[derive(Debug)]
pub enum TileLoadTerminal {
    Complete(TileData),
    Error(RequestTicket),
    Cancelled(RequestTicket),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RequestStatus {
    Queued,
    Running,
    Cancelled,
    Terminal,
    #[cfg(target_arch = "wasm32")]
    Error,
}

#[derive(Clone, Copy, Debug)]
struct RequestRecord {
    ticket: RequestTicket,
    status: RequestStatus,
}

#[derive(Default)]
struct RequestState {
    active: HashMap<TileId, RequestRecord>,
    #[cfg(target_arch = "wasm32")]
    wasm_requests: VecDeque<RequestTicket>,
    #[cfg(target_arch = "wasm32")]
    wasm_cancellations: VecDeque<RequestTicket>,
    #[cfg(target_arch = "wasm32")]
    wasm_terminals: VecDeque<TerminalEvent>,
    #[cfg(target_arch = "wasm32")]
    wasm_errors: VecDeque<(RequestTicket, u32)>,
}

#[derive(Debug)]
enum TerminalEvent {
    Complete(RequestTicket, TileData),
    #[cfg(not(target_arch = "wasm32"))]
    Error(RequestTicket),
    Cancelled(RequestTicket),
}

/// One lifecycle state shared by native workers and the browser facade.
/// Capacity is retained from admission through terminal acknowledgement.
struct BoundedRequestState {
    state: Mutex<RequestState>,
    max_in_flight: usize,
    next_generation: AtomicU64,
}

impl BoundedRequestState {
    fn new(max_in_flight: usize) -> Self {
        Self {
            state: Mutex::new(RequestState::default()),
            max_in_flight: max_in_flight.max(1),
            next_generation: AtomicU64::new(1),
        }
    }

    fn admit(&self, id: TileId, policy: CoalescePolicy) -> Option<RequestTicket> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.active.contains_key(&id) || state.active.len() >= self.max_in_flight {
            return None;
        }
        match policy {
            CoalescePolicy::PreferCoarse => {
                let mut ancestor = id;
                while let Some(parent) = ancestor.parent() {
                    if state.active.contains_key(&parent) {
                        return None;
                    }
                    ancestor = parent;
                }
            }
            CoalescePolicy::PreferFine => {
                if state
                    .active
                    .keys()
                    .copied()
                    .any(|candidate| is_descendant_of(candidate, id))
                {
                    return None;
                }
            }
        }
        let ticket = RequestTicket {
            tile_id: id,
            generation: self.next_generation.fetch_add(1, Ordering::Relaxed),
        };
        state.active.insert(
            id,
            RequestRecord {
                ticket,
                status: RequestStatus::Queued,
            },
        );
        Some(ticket)
    }

    fn rollback_admission(&self, ticket: RequestTicket) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state
            .active
            .get(&ticket.tile_id)
            .is_some_and(|record| record.ticket == ticket && record.status == RequestStatus::Queued)
        {
            state.active.remove(&ticket.tile_id);
        }
    }

    fn mark_running(&self, ticket: RequestTicket) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let Some(record) = state.active.get_mut(&ticket.tile_id) else {
            return false;
        };
        if record.ticket != ticket || record.status != RequestStatus::Queued {
            return false;
        }
        record.status = RequestStatus::Running;
        true
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn finish_native(
        &self,
        ticket: RequestTicket,
        result: Result<HeightRead, String>,
        width: u32,
        height: u32,
    ) -> Option<TerminalEvent> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let Some(record) = state.active.get_mut(&ticket.tile_id) else {
            return None;
        };
        if record.ticket != ticket || record.status == RequestStatus::Terminal {
            return None;
        }
        // Result selection and the terminal transition share this lock. A
        // cancellation that wins before this point can never emit Complete.
        let event = if record.status == RequestStatus::Cancelled {
            TerminalEvent::Cancelled(ticket)
        } else {
            match result {
                Ok(payload) => TerminalEvent::Complete(
                    ticket,
                    TileData::new_covered(
                        ticket.tile_id,
                        payload.heights,
                        payload.coverage,
                        width,
                        height,
                    ),
                ),
                Err(_) => TerminalEvent::Error(ticket),
            }
        };
        record.status = RequestStatus::Terminal;
        Some(event)
    }

    fn cancel(&self, ids: &[TileId]) -> Vec<RequestTicket> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let mut tickets = Vec::new();
        for id in ids {
            let Some(record) = state.active.get_mut(id) else {
                continue;
            };
            if matches!(
                record.status,
                RequestStatus::Queued | RequestStatus::Running
            ) {
                record.status = RequestStatus::Cancelled;
                tickets.push(record.ticket);
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            for ticket in &tickets {
                if state.wasm_cancellations.len() < self.max_in_flight {
                    state.wasm_cancellations.push_back(*ticket);
                }
            }
        }
        tickets
    }

    fn acknowledge(&self, ticket: RequestTicket) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.active.get(&ticket.tile_id).is_some_and(|record| {
            record.ticket == ticket && record.status == RequestStatus::Terminal
        }) {
            state.active.remove(&ticket.tile_id);
            true
        } else {
            false
        }
    }

    fn pending_len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .active
            .len()
    }

    #[cfg(target_arch = "wasm32")]
    fn enqueue_wasm(&self, ticket: RequestTicket) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.wasm_requests.len() >= self.max_in_flight {
            return false;
        }
        state.wasm_requests.push_back(ticket);
        true
    }

    #[cfg(target_arch = "wasm32")]
    fn next_wasm_request(&self) -> Option<RequestTicket> {
        let ticket = self
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .wasm_requests
            .pop_front()?;
        self.mark_running(ticket).then_some(ticket)
    }

    #[cfg(target_arch = "wasm32")]
    fn next_wasm_cancellation(&self) -> Option<RequestTicket> {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .wasm_cancellations
            .pop_front()
    }

    #[cfg(target_arch = "wasm32")]
    fn complete_wasm(&self, ticket: RequestTicket, tile: TileData) -> bool {
        let Some(expected_len) = tile
            .width
            .checked_mul(tile.height)
            .and_then(|count| usize::try_from(count).ok())
        else {
            return false;
        };
        if tile.height_data.len() != expected_len || tile.coverage_data.len() != expected_len {
            return false;
        }
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let terminal_full = state.wasm_terminals.len() >= self.max_in_flight;
        let Some(record) = state.active.get_mut(&ticket.tile_id) else {
            return false;
        };
        if record.ticket != ticket
            || record.status != RequestStatus::Running
            || tile.tile_id != ticket.tile_id
            || terminal_full
        {
            return false;
        }
        record.status = RequestStatus::Terminal;
        state
            .wasm_terminals
            .push_back(TerminalEvent::Complete(ticket, tile));
        true
    }

    #[cfg(target_arch = "wasm32")]
    fn is_wasm_running(&self, ticket: RequestTicket) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .active
            .get(&ticket.tile_id)
            .is_some_and(|record| {
                record.ticket == ticket && record.status == RequestStatus::Running
            })
    }

    #[cfg(target_arch = "wasm32")]
    fn fail_wasm(&self, ticket: RequestTicket, error_code: u32) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let terminal_full = state.wasm_errors.len() >= self.max_in_flight;
        let Some(record) = state.active.get_mut(&ticket.tile_id) else {
            return false;
        };
        if error_code == 0
            || record.ticket != ticket
            || record.status != RequestStatus::Running
            || terminal_full
        {
            return false;
        }
        record.status = RequestStatus::Error;
        state.wasm_errors.push_back((ticket, error_code));
        true
    }

    #[cfg(target_arch = "wasm32")]
    fn next_wasm_error(&self) -> Option<(RequestTicket, u32)> {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .wasm_errors
            .pop_front()
    }

    #[cfg(target_arch = "wasm32")]
    fn acknowledge_wasm_error(&self, ticket: RequestTicket) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state
            .active
            .get(&ticket.tile_id)
            .is_some_and(|record| record.ticket == ticket && record.status == RequestStatus::Error)
        {
            state.active.remove(&ticket.tile_id);
            true
        } else {
            false
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn acknowledge_wasm_cancellation(&self, ticket: RequestTicket) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let terminal_full = state.wasm_terminals.len() >= self.max_in_flight;
        let Some(record) = state.active.get_mut(&ticket.tile_id) else {
            return false;
        };
        if record.ticket != ticket || record.status != RequestStatus::Cancelled || terminal_full {
            return false;
        }
        record.status = RequestStatus::Terminal;
        state
            .wasm_terminals
            .push_back(TerminalEvent::Cancelled(ticket));
        true
    }

    #[cfg(target_arch = "wasm32")]
    fn next_wasm_terminal(&self) -> Option<TerminalEvent> {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .wasm_terminals
            .pop_front()
    }
}

pub struct AsyncTileLoader {
    lifecycle: Arc<BoundedRequestState>,
    #[cfg(not(target_arch = "wasm32"))]
    request_tx: SyncSender<RequestTicket>,
    #[cfg(not(target_arch = "wasm32"))]
    terminal_rx: Receiver<TerminalEvent>,
    #[cfg(not(target_arch = "wasm32"))]
    _workers: Vec<thread::JoinHandle<()>>,
    pool_size: usize,
    max_in_flight: usize,
    policy: CoalescePolicy,
    c_requests: AtomicUsize,
    c_enqueued: AtomicUsize,
    c_dropped_by_policy: AtomicUsize,
    c_canceled: AtomicUsize,
    c_send_fail: AtomicUsize,
    c_completed: AtomicUsize,
}

impl AsyncTileLoader {
    #[allow(unreachable_code)]
    pub fn new_with_reader(
        root_bounds: TileBounds,
        tile_size: Vec2,
        tile_resolution: u32,
        max_in_flight: usize,
        pool_size: usize,
        reader: Arc<dyn HeightReader>,
        policy: CoalescePolicy,
    ) -> Self {
        let max_in_flight = max_in_flight.max(1);
        let pool_size = bounded_loader_workers(pool_size);
        let lifecycle = Arc::new(BoundedRequestState::new(max_in_flight));
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (root_bounds, tile_size, tile_resolution, reader);
            return Self {
                lifecycle,
                pool_size,
                max_in_flight,
                policy,
                c_requests: AtomicUsize::new(0),
                c_enqueued: AtomicUsize::new(0),
                c_dropped_by_policy: AtomicUsize::new(0),
                c_canceled: AtomicUsize::new(0),
                c_send_fail: AtomicUsize::new(0),
                c_completed: AtomicUsize::new(0),
            };
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (request_tx, request_rx) = mpsc::sync_channel(max_in_flight);
            let (terminal_tx, terminal_rx) = mpsc::sync_channel(max_in_flight);
            let request_rx = Arc::new(Mutex::new(request_rx));
            let mut workers = Vec::new();
            for _ in 0..pool_size {
                let request_rx = request_rx.clone();
                let terminal_tx = terminal_tx.clone();
                let lifecycle = lifecycle.clone();
                let root_bounds = root_bounds.clone();
                let reader = reader.clone();
                workers.push(thread::spawn(move || loop {
                    let ticket = {
                        let receiver = request_rx.lock().unwrap_or_else(|p| p.into_inner());
                        match receiver.recv() {
                            Ok(ticket) => ticket,
                            Err(_) => break,
                        }
                    };
                    if !lifecycle.mark_running(ticket) {
                        if let Some(event) = lifecycle.finish_native(
                            ticket,
                            Err("request cancelled before worker start".to_string()),
                            tile_resolution,
                            tile_resolution,
                        ) {
                            if terminal_tx.send(event).is_err() {
                                break;
                            }
                        }
                        continue;
                    }
                    let result = reader.read_covered_result(
                        &root_bounds,
                        tile_size,
                        ticket.tile_id,
                        tile_resolution,
                        tile_resolution,
                    );
                    if let Some(event) =
                        lifecycle.finish_native(ticket, result, tile_resolution, tile_resolution)
                    {
                        if terminal_tx.send(event).is_err() {
                            break;
                        }
                    }
                }));
            }
            Self {
                lifecycle,
                request_tx,
                terminal_rx,
                _workers: workers,
                pool_size,
                max_in_flight,
                policy,
                c_requests: AtomicUsize::new(0),
                c_enqueued: AtomicUsize::new(0),
                c_dropped_by_policy: AtomicUsize::new(0),
                c_canceled: AtomicUsize::new(0),
                c_send_fail: AtomicUsize::new(0),
                c_completed: AtomicUsize::new(0),
            }
        }
    }

    pub fn request(&self, id: TileId) -> bool {
        self.c_requests.fetch_add(1, Ordering::Relaxed);
        let Some(ticket) = self.lifecycle.admit(id, self.policy) else {
            self.c_dropped_by_policy.fetch_add(1, Ordering::Relaxed);
            return false;
        };
        #[cfg(not(target_arch = "wasm32"))]
        let sent = self.request_tx.try_send(ticket).is_ok();
        #[cfg(target_arch = "wasm32")]
        let sent = self.lifecycle.enqueue_wasm(ticket);
        if !sent {
            self.lifecycle.rollback_admission(ticket);
            self.c_send_fail.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        self.c_enqueued.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn next_terminal(&self) -> Option<TerminalEvent> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.terminal_rx.try_recv().ok()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.lifecycle.next_wasm_terminal()
        }
    }

    pub fn drain_completed(&self, limit: usize) -> Vec<TileData> {
        self.drain_terminals(limit)
            .into_iter()
            .filter_map(|terminal| match terminal {
                TileLoadTerminal::Complete(tile) => Some(tile),
                TileLoadTerminal::Error(_) | TileLoadTerminal::Cancelled(_) => None,
            })
            .collect()
    }

    pub fn drain_terminals(&self, limit: usize) -> Vec<TileLoadTerminal> {
        let mut out = Vec::new();
        let mut acknowledged = 0;
        while acknowledged < limit {
            let Some(event) = self.next_terminal() else {
                break;
            };
            acknowledged += 1;
            match event {
                TerminalEvent::Complete(ticket, tile) => {
                    if self.lifecycle.acknowledge(ticket) {
                        self.c_completed.fetch_add(1, Ordering::Relaxed);
                        out.push(TileLoadTerminal::Complete(tile));
                    }
                }
                TerminalEvent::Cancelled(ticket) => {
                    if self.lifecycle.acknowledge(ticket) {
                        out.push(TileLoadTerminal::Cancelled(ticket));
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                TerminalEvent::Error(ticket) => {
                    if self.lifecycle.acknowledge(ticket) {
                        out.push(TileLoadTerminal::Error(ticket));
                    }
                }
            }
        }
        out
    }

    pub fn cancel(&self, ids: &[TileId]) -> usize {
        let tickets = self.lifecycle.cancel(ids);
        self.c_canceled.fetch_add(tickets.len(), Ordering::Relaxed);
        tickets.len()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn next_wasm_request(&self) -> Option<RequestTicket> {
        self.lifecycle.next_wasm_request()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn next_wasm_cancellation(&self) -> Option<RequestTicket> {
        self.lifecycle.next_wasm_cancellation()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn complete_wasm(&self, ticket: RequestTicket, tile: TileData) -> bool {
        self.lifecycle.complete_wasm(ticket, tile)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn drain_wasm_completed(&self, limit: usize) -> Vec<(RequestTicket, TileData)> {
        let mut out = Vec::new();
        while out.len() < limit {
            let Some(event) = self.lifecycle.next_wasm_terminal() else {
                break;
            };
            match event {
                TerminalEvent::Complete(ticket, tile) => {
                    if self.lifecycle.acknowledge(ticket) {
                        self.c_completed.fetch_add(1, Ordering::Relaxed);
                        out.push((ticket, tile));
                    }
                }
                TerminalEvent::Cancelled(ticket) => {
                    let _ = self.lifecycle.acknowledge(ticket);
                }
            }
        }
        out
    }

    #[cfg(target_arch = "wasm32")]
    pub fn is_wasm_running(&self, ticket: RequestTicket) -> bool {
        self.lifecycle.is_wasm_running(ticket)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn fail_wasm(&self, ticket: RequestTicket, error_code: u32) -> bool {
        self.lifecycle.fail_wasm(ticket, error_code)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn next_wasm_error(&self) -> Option<(RequestTicket, u32)> {
        self.lifecycle.next_wasm_error()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn acknowledge_wasm_error(&self, ticket: RequestTicket) -> bool {
        self.lifecycle.acknowledge_wasm_error(ticket)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn acknowledge_wasm_cancellation(&self, ticket: RequestTicket) -> bool {
        self.lifecycle.acknowledge_wasm_cancellation(ticket)
    }

    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.lifecycle.pending_len(),
            self.max_in_flight,
            self.pool_size,
        )
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn defensive_worker_bound_preserves_zero_semantics_and_caps_oversize() {
        assert_eq!(bounded_loader_workers(0), 1);
        assert_eq!(
            bounded_loader_workers(MAX_LOADER_WORKERS),
            MAX_LOADER_WORKERS
        );
        assert_eq!(bounded_loader_workers(usize::MAX), MAX_LOADER_WORKERS);
    }
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Barrier, Condvar};

    struct ConstantReader;
    impl HeightReader for ConstantReader {
        fn read_result(
            &self,
            _root_bounds: &TileBounds,
            _tile_size: Vec2,
            _tile_id: TileId,
            width: u32,
            height: u32,
        ) -> Result<Vec<f32>, String> {
            Ok(vec![1.0; (width * height) as usize])
        }
    }

    #[test]
    fn bounded_requests_deduplicate_and_release_after_nonblocking_completion() {
        let loader = AsyncTileLoader::new_with_reader(
            TileBounds::new(Vec2::ZERO, Vec2::ONE),
            Vec2::ONE,
            2,
            1,
            1,
            Arc::new(ConstantReader),
            CoalescePolicy::PreferFine,
        );
        let first = TileId::new(1, 0, 0);
        let second = TileId::new(1, 1, 0);
        assert!(loader.request(first));
        assert!(!loader.request(first));
        assert!(!loader.request(second));
        let completed = (0..10_000)
            .find_map(|_| {
                let result = loader.drain_completed(1);
                (!result.is_empty()).then_some(result)
            })
            .expect("worker completion");
        assert_eq!(completed[0].tile_id, first);
        assert!(loader.request(second));
    }

    #[test]
    fn cancellation_between_read_result_and_terminal_selection_wins_atomically() {
        let lifecycle = BoundedRequestState::new(1);
        let tile = TileId::new(2, 1, 1);
        let ticket = lifecycle.admit(tile, CoalescePolicy::PreferFine).unwrap();
        assert!(lifecycle.mark_running(ticket));

        // Deterministic form of the worker/canceller interleaving: the reader
        // has produced data, cancellation commits, then the worker publishes.
        assert_eq!(lifecycle.cancel(&[tile]), vec![ticket]);
        let event = lifecycle
            .finish_native(ticket, Ok(HeightRead::fully_covered(vec![7.0; 4])), 2, 2)
            .unwrap();
        assert!(matches!(event, TerminalEvent::Cancelled(t) if t == ticket));
        assert!(lifecycle.acknowledge(ticket));
    }

    struct TransientReader(AtomicUsize);

    impl HeightReader for TransientReader {
        fn read_result(
            &self,
            _root_bounds: &TileBounds,
            _tile_size: Vec2,
            _tile_id: TileId,
            width: u32,
            height: u32,
        ) -> Result<Vec<f32>, String> {
            if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                Err("transient COG range failure".to_string())
            } else {
                Ok(vec![9.0; (width * height) as usize])
            }
        }
    }

    #[test]
    fn source_error_is_terminal_without_zero_residency_and_root_can_retry() {
        let loader = AsyncTileLoader::new_with_reader(
            TileBounds::new(Vec2::ZERO, Vec2::ONE),
            Vec2::ONE,
            2,
            1,
            1,
            Arc::new(TransientReader(AtomicUsize::new(0))),
            CoalescePolicy::PreferFine,
        );
        let root = TileId::new(0, 0, 0);
        assert!(loader.request(root));
        let first = (0..20_000)
            .find_map(|_| loader.drain_terminals(1).pop())
            .expect("failure terminal");
        assert!(matches!(first, TileLoadTerminal::Error(ticket) if ticket.tile_id == root));
        assert!(
            loader.drain_completed(1).is_empty(),
            "errors must not fabricate zero tiles"
        );

        assert!(
            loader.request(root),
            "terminal failure releases bounded capacity for retry"
        );
        let recovered = (0..20_000)
            .find_map(|_| loader.drain_terminals(1).pop())
            .expect("recovery terminal");
        match recovered {
            TileLoadTerminal::Complete(tile) => {
                assert_eq!(tile.tile_id, root);
                assert_eq!(tile.height_data, vec![9.0; 4]);
            }
            _ => panic!("transient retry did not recover"),
        }
    }

    struct BlockedReader {
        entered: Arc<Barrier>,
        release: Arc<(Mutex<bool>, Condvar)>,
    }
    impl HeightReader for BlockedReader {
        fn read_result(
            &self,
            _root_bounds: &TileBounds,
            _tile_size: Vec2,
            _tile_id: TileId,
            width: u32,
            height: u32,
        ) -> Result<Vec<f32>, String> {
            self.entered.wait();
            let (lock, cv) = &*self.release;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = cv.wait(released).unwrap();
            }
            Ok(vec![1.0; (width * height) as usize])
        }
    }

    #[test]
    fn cancellation_retains_capacity_until_terminal_acknowledgement() {
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let loader = AsyncTileLoader::new_with_reader(
            TileBounds::new(Vec2::ZERO, Vec2::ONE),
            Vec2::ONE,
            2,
            1,
            1,
            Arc::new(BlockedReader {
                entered: entered.clone(),
                release: release.clone(),
            }),
            CoalescePolicy::PreferFine,
        );
        let first = TileId::new(1, 0, 0);
        let second = TileId::new(1, 1, 0);
        assert!(loader.request(first));
        entered.wait();
        assert_eq!(loader.cancel(&[first]), 1);
        assert!(!loader.request(second), "cancel is not acknowledgement");
        let (lock, cv) = &*release;
        *lock.lock().unwrap() = true;
        cv.notify_all();
        for _ in 0..10_000 {
            loader.drain_completed(1);
            if loader.stats().0 == 0 {
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(loader.stats().0, 0);
        assert!(loader.request(second));
    }

    #[test]
    fn bounded_cancellation_stress_waits_for_every_terminal_acknowledgement() {
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let loader = AsyncTileLoader::new_with_reader(
            TileBounds::new(Vec2::ZERO, Vec2::ONE),
            Vec2::ONE,
            2,
            4,
            1,
            Arc::new(BlockedReader {
                entered: entered.clone(),
                release: release.clone(),
            }),
            CoalescePolicy::PreferFine,
        );
        let admitted = [
            TileId::new(3, 0, 0),
            TileId::new(3, 1, 0),
            TileId::new(3, 2, 0),
            TileId::new(3, 3, 0),
        ];
        for tile in admitted {
            assert!(loader.request(tile));
        }
        entered.wait();
        assert_eq!(loader.stats().0, 4);
        assert_eq!(loader.cancel(&admitted), 4);
        let replacement = TileId::new(3, 4, 0);
        assert!(!loader.request(replacement));

        let (lock, cv) = &*release;
        *lock.lock().unwrap() = true;
        cv.notify_all();
        for _ in 0..20_000 {
            loader.drain_completed(4);
            if loader.stats().0 == 0 {
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(loader.stats().0, 0);
        assert!(loader.request(replacement));
    }
}
