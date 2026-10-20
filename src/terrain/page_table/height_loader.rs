use glam::Vec2;
use std::sync::Arc;

use super::common::CoalescePolicy;
use super::readers::HeightReader;
use super::scheduler::{RequestTerminal, SchedulingPolicy, TileRequestScheduler};
pub use super::scheduler::{RequestTicket, MAX_LOADER_WORKERS};
use crate::terrain::tiling::{TileBounds, TileData, TileId};

#[derive(Debug)]
pub enum TileLoadTerminal {
    Complete(TileData),
    Error(RequestTicket),
    Cancelled(RequestTicket),
}

pub struct AsyncTileLoader {
    scheduler: TileRequestScheduler<TileData>,
}

impl AsyncTileLoader {
    pub fn new_with_reader(
        root_bounds: TileBounds,
        tile_size: Vec2,
        tile_resolution: u32,
        max_in_flight: usize,
        pool_size: usize,
        reader: Arc<dyn HeightReader>,
        policy: CoalescePolicy,
    ) -> Self {
        Self {
            scheduler: TileRequestScheduler::new(
                max_in_flight,
                pool_size,
                policy,
                SchedulingPolicy::Bounded,
                move |id| {
                    reader
                        .read_covered_result(
                            &root_bounds,
                            tile_size,
                            id,
                            tile_resolution,
                            tile_resolution,
                        )
                        .map(|payload| {
                            TileData::new_covered(
                                id,
                                payload.heights,
                                payload.coverage,
                                tile_resolution,
                                tile_resolution,
                            )
                        })
                },
            ),
        }
    }

    pub fn request(&self, id: TileId) -> bool {
        self.scheduler.request(id)
    }

    #[cfg(all(not(target_arch = "wasm32"), any(feature = "extension-module", test)))]
    pub(crate) fn wait_for_terminal_for_test(&mut self) -> Result<(), String> {
        self.scheduler.wait_for_terminal_for_test()
    }

    pub fn drain_completed(&self, limit: usize) -> Vec<TileData> {
        self.scheduler.drain_completed(limit)
    }

    pub fn drain_terminals(&self, limit: usize) -> Vec<TileLoadTerminal> {
        self.scheduler
            .drain_terminals(limit)
            .into_iter()
            .map(|event| match event {
                RequestTerminal::Complete(_, tile) => TileLoadTerminal::Complete(tile),
                RequestTerminal::Error(ticket) => TileLoadTerminal::Error(ticket),
                RequestTerminal::Cancelled(ticket) => TileLoadTerminal::Cancelled(ticket),
            })
            .collect()
    }

    pub fn cancel(&self, ids: &[TileId]) -> usize {
        self.scheduler.cancel(ids)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn next_wasm_request(&self) -> Option<RequestTicket> {
        self.scheduler.next_wasm_request()
    }
    #[cfg(target_arch = "wasm32")]
    pub fn next_wasm_cancellation(&self) -> Option<RequestTicket> {
        self.scheduler.next_wasm_cancellation()
    }
    #[cfg(target_arch = "wasm32")]
    pub fn complete_wasm(&self, ticket: RequestTicket, tile: TileData) -> bool {
        let Some(expected_len) = tile
            .width
            .checked_mul(tile.height)
            .and_then(|n| usize::try_from(n).ok())
        else {
            return false;
        };
        if tile.tile_id != ticket.tile_id
            || tile.height_data.len() != expected_len
            || tile.coverage_data.len() != expected_len
        {
            return false;
        }
        self.scheduler.complete_wasm(ticket, tile)
    }
    #[cfg(target_arch = "wasm32")]
    pub fn drain_wasm_completed(&self, limit: usize) -> Vec<(RequestTicket, TileData)> {
        self.scheduler.drain_wasm_completed(limit)
    }
    #[cfg(target_arch = "wasm32")]
    pub fn is_wasm_running(&self, ticket: RequestTicket) -> bool {
        self.scheduler.is_wasm_running(ticket)
    }
    #[cfg(target_arch = "wasm32")]
    pub fn fail_wasm(&self, ticket: RequestTicket, error_code: u32) -> bool {
        self.scheduler.fail_wasm(ticket, error_code)
    }
    #[cfg(target_arch = "wasm32")]
    pub fn next_wasm_error(&self) -> Option<(RequestTicket, u32)> {
        self.scheduler.next_wasm_error()
    }
    #[cfg(target_arch = "wasm32")]
    pub fn acknowledge_wasm_error(&self, ticket: RequestTicket) -> bool {
        self.scheduler.acknowledge_wasm_error(ticket)
    }
    #[cfg(target_arch = "wasm32")]
    pub fn acknowledge_wasm_cancellation(&self, ticket: RequestTicket) -> bool {
        self.scheduler.acknowledge_wasm_cancellation(ticket)
    }

    pub fn stats(&self) -> (usize, usize, usize) {
        self.scheduler.stats()
    }
    pub fn counters(&self) -> (usize, usize, usize, usize, usize, usize) {
        self.scheduler.counters()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::super::scheduler::bounded_loader_workers;
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn defensive_worker_bound_preserves_zero_semantics_and_caps_oversize() {
        assert_eq!(bounded_loader_workers(0), 1);
        assert_eq!(
            bounded_loader_workers(MAX_LOADER_WORKERS),
            MAX_LOADER_WORKERS
        );
        assert_eq!(bounded_loader_workers(usize::MAX), MAX_LOADER_WORKERS);
    }
    use super::super::loader_tests::wait_for;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Barrier, Condvar, Mutex};

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
        let completed = wait_for("worker completion", || {
            let result = loader.drain_completed(1);
            (!result.is_empty()).then_some(result)
        });
        assert_eq!(completed[0].tile_id, first);
        assert!(loader.request(second));
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
    fn test_barrier_preserves_terminal_errors_and_bounded_capacity_until_drain() {
        let mut loader = AsyncTileLoader::new_with_reader(
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
        loader.wait_for_terminal_for_test().unwrap();
        assert_eq!(loader.stats().0, 1);
        assert!(!loader.request(TileId::new(1, 0, 0)));
        assert!(matches!(loader.drain_terminals(1).pop(),
            Some(TileLoadTerminal::Error(ticket)) if ticket.tile_id == root));
        assert_eq!(loader.stats().0, 0);

        assert!(loader.request(root));
        loader.wait_for_terminal_for_test().unwrap();
        loader.wait_for_terminal_for_test().unwrap();
        assert_eq!(loader.stats().0, 1);
        let tile = loader.drain_completed(1).pop().unwrap();
        assert_eq!(tile.tile_id, root);
        assert_eq!(tile.height_data, vec![9.0; 4]);
        assert_eq!(loader.stats().0, 0);
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
        let first = wait_for("failure terminal", || loader.drain_terminals(1).pop());
        assert!(matches!(first, TileLoadTerminal::Error(ticket) if ticket.tile_id == root));
        assert!(
            loader.drain_completed(1).is_empty(),
            "errors must not fabricate zero tiles"
        );

        assert!(
            loader.request(root),
            "terminal failure releases bounded capacity for retry"
        );
        let recovered = wait_for("recovery terminal", || loader.drain_terminals(1).pop());
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
        wait_for("terminal acknowledgement", || {
            loader.drain_completed(1);
            (loader.stats().0 == 0).then_some(())
        });
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
        wait_for("terminal acknowledgement", || {
            loader.drain_completed(4);
            (loader.stats().0 == 0).then_some(())
        });
        assert!(loader.request(replacement));
    }
}
