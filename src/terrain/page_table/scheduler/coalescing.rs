//! Overlay's established replacement policy releases logical slots on cancel.
//! Keep it distinct from height's acknowledgement-bound admission policy.
use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::sync::mpsc::SyncSender;
use std::sync::Mutex;

use super::super::common::{is_descendant_of, CoalescePolicy};
use super::{Counters, RequestTicket};
use crate::terrain::tiling::TileId;

pub(super) struct CoalescingRequestState {
    pending: Mutex<HashSet<TileId>>,
    cancelled: Mutex<HashSet<TileId>>,
    max_in_flight: usize,
}

impl CoalescingRequestState {
    pub(super) fn new(max_in_flight: usize) -> Self {
        Self {
            pending: Mutex::new(HashSet::new()),
            cancelled: Mutex::new(HashSet::new()),
            max_in_flight,
        }
    }

    pub(super) fn request(
        &self,
        id: TileId,
        request_tx: &SyncSender<RequestTicket>,
        policy: CoalescePolicy,
        counters: &Counters,
    ) -> bool {
        if let Ok(mut pend) = self.pending.lock() {
            counters.c_requests.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut can) = self.cancelled.lock() {
                can.remove(&id);
            }
            match policy {
                CoalescePolicy::PreferCoarse => {
                    let mut p = id;
                    while let Some(parent) = p.parent() {
                        if pend.contains(&parent) {
                            counters.c_dropped_by_policy.fetch_add(1, Ordering::Relaxed);
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
                    if request_tx
                        .try_send(RequestTicket {
                            tile_id: id,
                            generation: 0,
                        })
                        .is_err()
                    {
                        pend.remove(&id);
                        counters.c_send_fail.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                    counters.c_enqueued.fetch_add(1, Ordering::Relaxed);
                    if !to_cancel.is_empty() {
                        if let Ok(mut can) = self.cancelled.lock() {
                            for d in &to_cancel {
                                pend.remove(d);
                                can.insert(*d);
                            }
                            counters
                                .c_canceled
                                .fetch_add(to_cancel.len(), Ordering::Relaxed);
                        }
                    }
                    true
                }
                CoalescePolicy::PreferFine => {
                    for &d in pend.iter() {
                        if is_descendant_of(d, id) {
                            counters.c_dropped_by_policy.fetch_add(1, Ordering::Relaxed);
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
                    if request_tx
                        .try_send(RequestTicket {
                            tile_id: id,
                            generation: 0,
                        })
                        .is_err()
                    {
                        pend.remove(&id);
                        counters.c_send_fail.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                    counters.c_enqueued.fetch_add(1, Ordering::Relaxed);
                    if let Ok(mut can) = self.cancelled.lock() {
                        let mut p = id;
                        while let Some(parent) = p.parent() {
                            if pend.remove(&parent) {
                                can.insert(parent);
                            }
                            p = parent;
                        }
                        counters.c_canceled.fetch_add(1, Ordering::Relaxed);
                    }
                    true
                }
            }
        } else {
            false
        }
    }

    pub(super) fn cancel(&self, ids: &[TileId]) -> usize {
        let mut n = 0usize;
        if let Ok(mut pend) = self.pending.lock() {
            if let Ok(mut can) = self.cancelled.lock() {
                for id in ids {
                    if pend.remove(id) {
                        can.insert(*id);
                        n += 1;
                    }
                }
            }
        }
        n
    }

    pub(super) fn acknowledge(&self, id: TileId) -> bool {
        if let Ok(mut pending) = self.pending.lock() {
            pending.remove(&id);
        }
        if let Ok(mut cancelled) = self.cancelled.lock() {
            if cancelled.remove(&id) {
                return false;
            }
        }
        true
    }

    pub(super) fn pending_len(&self) -> usize {
        self.pending
            .lock()
            .map(|pending| pending.len())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturated_dispatch_queue_rolls_back_replacement_without_cancelling_existing_requests() {
        let root = TileId::new(0, 0, 0);
        let child = TileId::new(1, 0, 0);
        for (policy, first, replacement) in [
            (CoalescePolicy::PreferCoarse, child, root),
            (CoalescePolicy::PreferFine, root, child),
        ] {
            let state = CoalescingRequestState::new(2);
            let counters = Counters::default();
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            assert!(state.request(first, &tx, policy, &counters));
            assert!(!state.request(replacement, &tx, policy, &counters));
            assert_eq!(state.pending_len(), 1);
            assert!(state.cancelled.lock().unwrap().is_empty());
            assert_eq!(counters.c_send_fail.load(Ordering::Relaxed), 1);
            assert_eq!(counters.c_enqueued.load(Ordering::Relaxed), 1);
            assert_eq!(rx.recv().unwrap().tile_id, first);
            assert!(state.acknowledge(first));
            assert_eq!(state.pending_len(), 0);
            assert!(state.request(replacement, &tx, policy, &counters));
        }
    }
}
