use std::collections::HashMap;
#[cfg(target_arch = "wasm32")]
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use super::super::common::{is_descendant_of, CoalescePolicy};
use super::{RequestTerminal, RequestTicket};
use crate::terrain::tiling::TileId;

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

struct RequestState<P> {
    #[cfg(not(target_arch = "wasm32"))]
    _payload: std::marker::PhantomData<fn() -> P>,
    active: HashMap<TileId, RequestRecord>,
    #[cfg(target_arch = "wasm32")]
    wasm_requests: VecDeque<RequestTicket>,
    #[cfg(target_arch = "wasm32")]
    wasm_cancellations: VecDeque<RequestTicket>,
    #[cfg(target_arch = "wasm32")]
    wasm_terminals: VecDeque<RequestTerminal<P>>,
    #[cfg(target_arch = "wasm32")]
    wasm_errors: VecDeque<(RequestTicket, u32)>,
}

/// One lifecycle state shared by native workers and the browser facade.
/// Capacity is retained from admission through terminal acknowledgement.
pub(super) struct BoundedRequestState<P> {
    state: Mutex<RequestState<P>>,
    max_in_flight: usize,
    next_generation: AtomicU64,
}

impl<P> BoundedRequestState<P> {
    pub(super) fn new(max_in_flight: usize) -> Self {
        Self {
            state: Mutex::new(RequestState {
                active: HashMap::new(),
                #[cfg(not(target_arch = "wasm32"))]
                _payload: std::marker::PhantomData,
                #[cfg(target_arch = "wasm32")]
                wasm_requests: VecDeque::new(),
                #[cfg(target_arch = "wasm32")]
                wasm_cancellations: VecDeque::new(),
                #[cfg(target_arch = "wasm32")]
                wasm_terminals: VecDeque::new(),
                #[cfg(target_arch = "wasm32")]
                wasm_errors: VecDeque::new(),
            }),
            max_in_flight: max_in_flight.max(1),
            next_generation: AtomicU64::new(1),
        }
    }

    pub(super) fn admit(&self, id: TileId, policy: CoalescePolicy) -> Option<RequestTicket> {
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

    pub(super) fn rollback_admission(&self, ticket: RequestTicket) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state
            .active
            .get(&ticket.tile_id)
            .is_some_and(|record| record.ticket == ticket && record.status == RequestStatus::Queued)
        {
            state.active.remove(&ticket.tile_id);
        }
    }

    pub(super) fn mark_running(&self, ticket: RequestTicket) -> bool {
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
    pub(super) fn finish_native(
        &self,
        ticket: RequestTicket,
        result: Result<P, String>,
    ) -> Option<RequestTerminal<P>> {
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
            RequestTerminal::Cancelled(ticket)
        } else {
            match result {
                Ok(payload) => RequestTerminal::Complete(ticket, payload),
                Err(_) => RequestTerminal::Error(ticket),
            }
        };
        record.status = RequestStatus::Terminal;
        Some(event)
    }

    pub(super) fn cancel(&self, ids: &[TileId]) -> Vec<RequestTicket> {
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

    pub(super) fn acknowledge(&self, ticket: RequestTicket) -> bool {
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

    pub(super) fn pending_len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .active
            .len()
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn enqueue_wasm(&self, ticket: RequestTicket) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.wasm_requests.len() >= self.max_in_flight {
            return false;
        }
        state.wasm_requests.push_back(ticket);
        true
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn next_wasm_request(&self) -> Option<RequestTicket> {
        let ticket = self
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .wasm_requests
            .pop_front()?;
        self.mark_running(ticket).then_some(ticket)
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn next_wasm_cancellation(&self) -> Option<RequestTicket> {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .wasm_cancellations
            .pop_front()
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn complete_wasm(&self, ticket: RequestTicket, payload: P) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let terminal_full = state.wasm_terminals.len() >= self.max_in_flight;
        let Some(record) = state.active.get_mut(&ticket.tile_id) else {
            return false;
        };
        if record.ticket != ticket || record.status != RequestStatus::Running || terminal_full {
            return false;
        }
        record.status = RequestStatus::Terminal;
        state
            .wasm_terminals
            .push_back(RequestTerminal::Complete(ticket, payload));
        true
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn is_wasm_running(&self, ticket: RequestTicket) -> bool {
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
    pub(super) fn fail_wasm(&self, ticket: RequestTicket, error_code: u32) -> bool {
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
    pub(super) fn next_wasm_error(&self) -> Option<(RequestTicket, u32)> {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .wasm_errors
            .pop_front()
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn acknowledge_wasm_error(&self, ticket: RequestTicket) -> bool {
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
    pub(super) fn acknowledge_wasm_cancellation(&self, ticket: RequestTicket) -> bool {
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
            .push_back(RequestTerminal::Cancelled(ticket));
        true
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn next_wasm_terminal(&self) -> Option<RequestTerminal<P>> {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .wasm_terminals
            .pop_front()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::terrain::tiling::TileData;

    #[test]
    fn cancellation_between_read_result_and_terminal_selection_wins_atomically() {
        let lifecycle = BoundedRequestState::<TileData>::new(1);
        let tile = TileId::new(2, 1, 1);
        let ticket = lifecycle.admit(tile, CoalescePolicy::PreferFine).unwrap();
        assert!(lifecycle.mark_running(ticket));

        // Deterministic form of the worker/canceller interleaving: the reader
        // has produced data, cancellation commits, then the worker publishes.
        assert_eq!(lifecycle.cancel(&[tile]), vec![ticket]);
        let event = lifecycle
            .finish_native(
                ticket,
                Ok(TileData::new_covered(
                    tile,
                    vec![7.0; 4],
                    vec![255; 4],
                    2,
                    2,
                )),
            )
            .unwrap();
        assert!(matches!(event, RequestTerminal::Cancelled(t) if t == ticket));
        assert!(lifecycle.acknowledge(ticket));
    }
}
