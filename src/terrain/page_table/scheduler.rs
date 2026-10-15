//! Queue, worker and completion ownership for both concrete tile readers.
//! Height keeps acknowledgement-bound capacity; overlays keep replacement
//! coalescing, round-robin dispatch and their existing logical pending limit.
#[path = "scheduler/bounded.rs"]
mod bounded;
#[cfg(not(target_arch = "wasm32"))]
#[path = "scheduler/coalescing.rs"]
mod coalescing;

#[cfg(not(target_arch = "wasm32"))]
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{self, Receiver, SendError, Sender, SyncSender};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;

use super::common::CoalescePolicy;
use crate::terrain::tiling::TileId;
use bounded::BoundedRequestState;
#[cfg(not(target_arch = "wasm32"))]
use coalescing::CoalescingRequestState;

/// Existing height-loader worker ceiling; overlays retain their supplied pool.
pub const MAX_LOADER_WORKERS: usize = 16;

pub(super) fn bounded_loader_workers(pool_size: usize) -> usize {
    pool_size.max(1).min(MAX_LOADER_WORKERS)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RequestTicket {
    pub tile_id: TileId,
    pub generation: u64,
}

#[derive(Debug)]
pub(super) enum RequestTerminal<P> {
    Complete(RequestTicket, P),
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    Error(RequestTicket),
    Cancelled(RequestTicket),
}

#[derive(Clone, Copy)]
pub(super) enum SchedulingPolicy {
    Bounded,
    Coalescing,
}

enum RequestLifecycle<P> {
    Bounded(BoundedRequestState<P>),
    #[cfg(not(target_arch = "wasm32"))]
    Coalescing(CoalescingRequestState),
}

impl<P> RequestLifecycle<P> {
    fn pending_len(&self) -> usize {
        match self {
            Self::Bounded(state) => state.pending_len(),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Coalescing(state) => state.pending_len(),
        }
    }

    fn acknowledge(&self, event: &RequestTerminal<P>) -> bool {
        let ticket = match event {
            RequestTerminal::Complete(ticket, _)
            | RequestTerminal::Error(ticket)
            | RequestTerminal::Cancelled(ticket) => *ticket,
        };
        match self {
            Self::Bounded(state) => state.acknowledge(ticket),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Coalescing(state) => state.acknowledge(ticket.tile_id),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn begin(&self, ticket: RequestTicket) -> bool {
        match self {
            Self::Bounded(state) => state.mark_running(ticket),
            Self::Coalescing(_) => true,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn finish(
        &self,
        ticket: RequestTicket,
        result: Result<P, String>,
    ) -> Option<RequestTerminal<P>> {
        match self {
            Self::Bounded(state) => state.finish_native(ticket, result),
            Self::Coalescing(_) => Some(match result {
                Ok(payload) => RequestTerminal::Complete(ticket, payload),
                Err(_) => RequestTerminal::Error(ticket),
            }),
        }
    }
}

#[derive(Default)]
struct Counters {
    c_requests: AtomicUsize,
    c_enqueued: AtomicUsize,
    c_dropped_by_policy: AtomicUsize,
    c_canceled: AtomicUsize,
    c_send_fail: AtomicUsize,
    c_completed: AtomicUsize,
}

#[cfg(not(target_arch = "wasm32"))]
enum TerminalSender<P> {
    Bounded(SyncSender<RequestTerminal<P>>),
    Coalescing(Sender<RequestTerminal<P>>),
}

#[cfg(not(target_arch = "wasm32"))]
impl<P> Clone for TerminalSender<P> {
    fn clone(&self) -> Self {
        match self {
            Self::Bounded(tx) => Self::Bounded(tx.clone()),
            Self::Coalescing(tx) => Self::Coalescing(tx.clone()),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<P> TerminalSender<P> {
    fn send(&self, event: RequestTerminal<P>) -> Result<(), SendError<RequestTerminal<P>>> {
        match self {
            Self::Bounded(tx) => tx.send(event),
            Self::Coalescing(tx) => tx.send(event),
        }
    }
}

pub(super) struct TileRequestScheduler<P> {
    lifecycle: Arc<RequestLifecycle<P>>,
    #[cfg(not(target_arch = "wasm32"))]
    request_tx: SyncSender<RequestTicket>,
    #[cfg(not(target_arch = "wasm32"))]
    terminal_rx: Receiver<RequestTerminal<P>>,
    #[cfg(not(target_arch = "wasm32"))]
    test_ready_terminal: RefCell<Option<RequestTerminal<P>>>,
    #[cfg(not(target_arch = "wasm32"))]
    _workers: Vec<thread::JoinHandle<()>>,
    #[cfg(not(target_arch = "wasm32"))]
    _dispatcher: Option<thread::JoinHandle<()>>,
    pool_size: usize,
    max_in_flight: usize,
    policy: CoalescePolicy,
    scheduling: SchedulingPolicy,
    counters: Counters,
}

impl<P: Send + 'static> TileRequestScheduler<P> {
    pub(super) fn new(
        max_in_flight: usize,
        pool_size: usize,
        policy: CoalescePolicy,
        scheduling: SchedulingPolicy,
        read: impl Fn(TileId) -> Result<P, String> + Send + Sync + 'static,
    ) -> Self {
        let max_in_flight = max_in_flight.max(1);
        let pool_size = match scheduling {
            SchedulingPolicy::Bounded => bounded_loader_workers(pool_size),
            SchedulingPolicy::Coalescing => pool_size.max(1),
        };
        #[cfg(target_arch = "wasm32")]
        {
            // Overlay's OS-thread reader has never been a browser capability.
            assert!(
                matches!(scheduling, SchedulingPolicy::Bounded),
                "asynchronous overlay readers require native threads"
            );
            let _ = read;
            Self {
                lifecycle: Arc::new(RequestLifecycle::Bounded(BoundedRequestState::new(
                    max_in_flight,
                ))),
                pool_size,
                max_in_flight,
                policy,
                scheduling,
                counters: Counters::default(),
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (request_tx, request_rx) = mpsc::sync_channel(max_in_flight);
            let (lifecycle, terminal_tx, terminal_rx) = match scheduling {
                SchedulingPolicy::Bounded => {
                    let (tx, rx) = mpsc::sync_channel(max_in_flight);
                    (
                        RequestLifecycle::Bounded(BoundedRequestState::new(max_in_flight)),
                        TerminalSender::Bounded(tx),
                        rx,
                    )
                }
                SchedulingPolicy::Coalescing => {
                    let (tx, rx) = mpsc::channel();
                    (
                        RequestLifecycle::Coalescing(CoalescingRequestState::new(max_in_flight)),
                        TerminalSender::Coalescing(tx),
                        rx,
                    )
                }
            };
            let lifecycle = Arc::new(lifecycle);
            let read = Arc::new(read);
            let request_rx = Arc::new(Mutex::new(request_rx));
            let mut workers = Vec::new();
            let mut worker_txs = Vec::new();
            for _ in 0..pool_size {
                let worker_rx = match scheduling {
                    SchedulingPolicy::Bounded => request_rx.clone(),
                    SchedulingPolicy::Coalescing => {
                        let (tx, rx) = mpsc::channel();
                        worker_txs.push(tx);
                        Arc::new(Mutex::new(rx))
                    }
                };
                let terminal_tx = terminal_tx.clone();
                let lifecycle = lifecycle.clone();
                let read = read.clone();
                workers.push(thread::spawn(move || loop {
                    let ticket = {
                        let receiver = worker_rx.lock().unwrap_or_else(|p| p.into_inner());
                        match receiver.recv() {
                            Ok(ticket) => ticket,
                            Err(_) => break,
                        }
                    };
                    let result = if lifecycle.begin(ticket) {
                        read(ticket.tile_id)
                    } else {
                        Err("request cancelled before worker start".to_string())
                    };
                    if let Some(event) = lifecycle.finish(ticket, result) {
                        if terminal_tx.send(event).is_err()
                            && matches!(scheduling, SchedulingPolicy::Bounded)
                        {
                            break;
                        }
                    }
                }));
            }
            let dispatcher = if matches!(scheduling, SchedulingPolicy::Coalescing) {
                Some(thread::spawn(move || {
                    let mut idx: usize = 0;
                    while let Ok(ticket) =
                        request_rx.lock().unwrap_or_else(|p| p.into_inner()).recv()
                    {
                        let _ = worker_txs[idx % worker_txs.len()].send(ticket);
                        idx = idx.wrapping_add(1);
                    }
                }))
            } else {
                None
            };
            Self {
                lifecycle,
                request_tx,
                terminal_rx,
                test_ready_terminal: RefCell::new(None),
                _workers: workers,
                _dispatcher: dispatcher,
                pool_size,
                max_in_flight,
                policy,
                scheduling,
                counters: Counters::default(),
            }
        }
    }

    pub(super) fn request(&self, id: TileId) -> bool {
        let state = match self.lifecycle.as_ref() {
            RequestLifecycle::Bounded(state) => state,
            #[cfg(not(target_arch = "wasm32"))]
            RequestLifecycle::Coalescing(state) => {
                return state.request(id, &self.request_tx, self.policy, &self.counters)
            }
        };
        self.counters.c_requests.fetch_add(1, Ordering::Relaxed);
        let Some(ticket) = state.admit(id, self.policy) else {
            self.counters
                .c_dropped_by_policy
                .fetch_add(1, Ordering::Relaxed);
            return false;
        };
        #[cfg(not(target_arch = "wasm32"))]
        let sent = self.request_tx.try_send(ticket).is_ok();
        #[cfg(target_arch = "wasm32")]
        let sent = state.enqueue_wasm(ticket);
        if !sent {
            state.rollback_admission(ticket);
            self.counters.c_send_fail.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        self.counters.c_enqueued.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn next_terminal(&self) -> Option<RequestTerminal<P>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.test_ready_terminal
                .borrow_mut()
                .take()
                .or_else(|| self.terminal_rx.try_recv().ok())
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.bounded().next_wasm_terminal()
        }
    }

    /// Hold one terminal inside its existing capacity slot until normal drain.
    #[cfg(all(not(target_arch = "wasm32"), any(feature = "extension-module", test)))]
    pub(super) fn wait_for_terminal_for_test(&mut self) -> Result<(), String> {
        if self.lifecycle.pending_len() > 0 && self.test_ready_terminal.get_mut().is_none() {
            *self.test_ready_terminal.get_mut() =
                Some(self.terminal_rx.recv().map_err(|error| {
                    format!("height worker terminal channel disconnected: {error}")
                })?);
        }
        Ok(())
    }

    fn drain_events(&self, limit: usize, completed_limit: bool) -> Vec<RequestTerminal<P>> {
        let mut out = Vec::new();
        let mut consumed = 0;
        while if completed_limit {
            out.len() < limit
        } else {
            consumed < limit
        } {
            let Some(event) = self.next_terminal() else {
                break;
            };
            consumed += 1;
            if self.lifecycle.acknowledge(&event) {
                let complete = matches!(event, RequestTerminal::Complete(_, _));
                if complete {
                    self.counters.c_completed.fetch_add(1, Ordering::Relaxed);
                }
                if complete || !completed_limit {
                    out.push(event);
                }
            }
        }
        out
    }

    pub(super) fn drain_terminals(&self, limit: usize) -> Vec<RequestTerminal<P>> {
        self.drain_events(limit, false)
    }

    pub(super) fn drain_completed(&self, limit: usize) -> Vec<P> {
        self.drain_events(
            limit,
            matches!(self.scheduling, SchedulingPolicy::Coalescing),
        )
        .into_iter()
        .filter_map(|event| match event {
            RequestTerminal::Complete(_, payload) => Some(payload),
            RequestTerminal::Error(_) | RequestTerminal::Cancelled(_) => None,
        })
        .collect()
    }

    pub(super) fn cancel(&self, ids: &[TileId]) -> usize {
        match self.lifecycle.as_ref() {
            RequestLifecycle::Bounded(state) => {
                let n = state.cancel(ids).len();
                self.counters.c_canceled.fetch_add(n, Ordering::Relaxed);
                n
            }
            #[cfg(not(target_arch = "wasm32"))]
            RequestLifecycle::Coalescing(state) => state.cancel(ids),
        }
    }

    pub(super) fn stats(&self) -> (usize, usize, usize) {
        (
            self.lifecycle.pending_len(),
            self.max_in_flight,
            self.pool_size,
        )
    }

    pub(super) fn counters(&self) -> (usize, usize, usize, usize, usize, usize) {
        let c = &self.counters;
        (
            c.c_requests.load(Ordering::Relaxed),
            c.c_enqueued.load(Ordering::Relaxed),
            c.c_dropped_by_policy.load(Ordering::Relaxed),
            c.c_canceled.load(Ordering::Relaxed),
            c.c_send_fail.load(Ordering::Relaxed),
            c.c_completed.load(Ordering::Relaxed),
        )
    }

    #[cfg(target_arch = "wasm32")]
    fn bounded(&self) -> &BoundedRequestState<P> {
        let RequestLifecycle::Bounded(state) = self.lifecycle.as_ref();
        state
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn next_wasm_request(&self) -> Option<RequestTicket> {
        self.bounded().next_wasm_request()
    }
    #[cfg(target_arch = "wasm32")]
    pub(super) fn next_wasm_cancellation(&self) -> Option<RequestTicket> {
        self.bounded().next_wasm_cancellation()
    }
    #[cfg(target_arch = "wasm32")]
    pub(super) fn complete_wasm(&self, ticket: RequestTicket, payload: P) -> bool {
        self.bounded().complete_wasm(ticket, payload)
    }
    #[cfg(target_arch = "wasm32")]
    pub(super) fn drain_wasm_completed(&self, limit: usize) -> Vec<(RequestTicket, P)> {
        self.drain_events(limit, true)
            .into_iter()
            .filter_map(|event| match event {
                RequestTerminal::Complete(ticket, payload) => Some((ticket, payload)),
                RequestTerminal::Error(_) | RequestTerminal::Cancelled(_) => None,
            })
            .collect()
    }
    #[cfg(target_arch = "wasm32")]
    pub(super) fn is_wasm_running(&self, ticket: RequestTicket) -> bool {
        self.bounded().is_wasm_running(ticket)
    }
    #[cfg(target_arch = "wasm32")]
    pub(super) fn fail_wasm(&self, ticket: RequestTicket, error_code: u32) -> bool {
        self.bounded().fail_wasm(ticket, error_code)
    }
    #[cfg(target_arch = "wasm32")]
    pub(super) fn next_wasm_error(&self) -> Option<(RequestTicket, u32)> {
        self.bounded().next_wasm_error()
    }
    #[cfg(target_arch = "wasm32")]
    pub(super) fn acknowledge_wasm_error(&self, ticket: RequestTicket) -> bool {
        self.bounded().acknowledge_wasm_error(ticket)
    }
    #[cfg(target_arch = "wasm32")]
    pub(super) fn acknowledge_wasm_cancellation(&self, ticket: RequestTicket) -> bool {
        self.bounded().acknowledge_wasm_cancellation(ticket)
    }
}
