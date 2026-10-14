//! Characterize the shared scheduler through the unchanged reader adapters.
use super::*;
use crate::terrain::tiling::{TileBounds, TileId};
use glam::Vec2;
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

/// Polls the non-blocking loader API until `poll` yields a value. Workers
/// are OS threads, so a fixed spin count races the scheduler on loaded
/// runners; a wall-clock deadline does not.
pub(super) fn wait_for<T>(what: &str, mut poll: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(value) = poll() {
            return value;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Height,
    Overlay,
}
const KINDS: [Kind; 2] = [Kind::Height, Kind::Overlay];

enum Loader {
    Height(AsyncTileLoader),
    Overlay(AsyncOverlayLoader),
}
impl Loader {
    fn request(&self, id: TileId) -> bool {
        match self {
            Self::Height(l) => l.request(id),
            Self::Overlay(l) => l.request(id),
        }
    }
    fn cancel(&self, ids: &[TileId]) -> usize {
        match self {
            Self::Height(l) => l.cancel(ids),
            Self::Overlay(l) => l.cancel(ids),
        }
    }
    fn stats(&self) -> (usize, usize, usize) {
        match self {
            Self::Height(l) => l.stats(),
            Self::Overlay(l) => l.stats(),
        }
    }
    fn counters(&self) -> (usize, usize, usize, usize, usize, usize) {
        match self {
            Self::Height(l) => l.counters(),
            Self::Overlay(l) => l.counters(),
        }
    }
    fn drain(&self, limit: usize) -> Vec<TileId> {
        match self {
            Self::Height(l) => l
                .drain_completed(limit)
                .into_iter()
                .map(|tile| {
                    assert_eq!((tile.width, tile.height), (2, 2));
                    assert_eq!(tile.height_data, vec![7.0; 4]);
                    assert_eq!(tile.coverage_data, vec![128; 4]);
                    assert_eq!(tile.host_memory_size, 20);
                    tile.tile_id
                })
                .collect(),
            Self::Overlay(l) => l
                .drain_completed(limit)
                .into_iter()
                .map(|tile| {
                    assert_eq!((tile.width, tile.height), (2, 2));
                    assert_eq!(tile.rgba_data, [1, 2, 3, 255].repeat(4));
                    assert_eq!(tile.host_memory_size, 16);
                    tile.tile_id
                })
                .collect(),
        }
    }
}

#[derive(Default)]
struct Gate {
    open: Mutex<bool>,
    changed: Condvar,
}
impl Gate {
    fn release(&self) {
        *self.open.lock().unwrap() = true;
        self.changed.notify_all();
    }
    fn wait(&self) {
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.changed.wait(open).unwrap();
        }
    }
}
struct Reader {
    gate: Arc<Gate>,
    started: mpsc::Sender<(TileId, ThreadId)>,
    exited: mpsc::Sender<()>,
}
impl Reader {
    fn begin(&self, id: TileId) {
        let _ = self.started.send((id, std::thread::current().id()));
        self.gate.wait();
    }
}
impl Drop for Reader {
    fn drop(&mut self) {
        let _ = self.exited.send(());
    }
}
impl HeightReader for Reader {
    fn read_result(
        &self,
        _root: &TileBounds,
        _size: Vec2,
        id: TileId,
        width: u32,
        height: u32,
    ) -> Result<Vec<f32>, String> {
        self.begin(id);
        Ok(vec![7.0; (width * height) as usize])
    }
    fn read_covered_result(
        &self,
        root: &TileBounds,
        size: Vec2,
        id: TileId,
        width: u32,
        height: u32,
    ) -> Result<HeightRead, String> {
        self.read_result(root, size, id, width, height)
            .map(|heights| HeightRead {
                heights,
                coverage: vec![128; (width * height) as usize],
            })
    }
}
impl OverlayReader for Reader {
    fn read(
        &self,
        _root: &TileBounds,
        _size: Vec2,
        id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        self.begin(id);
        [1, 2, 3, 255].repeat((width * height) as usize)
    }
}
struct Fixture {
    loader: Option<Loader>,
    gate: Arc<Gate>,
    started: mpsc::Receiver<(TileId, ThreadId)>,
    exited: mpsc::Receiver<()>,
    shutdown_observed: bool,
}
impl Fixture {
    fn new(kind: Kind, capacity: usize, pool: usize, policy: CoalescePolicy) -> Self {
        let gate = Arc::new(Gate::default());
        let (started, started_rx) = mpsc::channel();
        let (exited, exited_rx) = mpsc::channel();
        let reader = Arc::new(Reader {
            gate: gate.clone(),
            started,
            exited,
        });
        let root = TileBounds::new(Vec2::ZERO, Vec2::ONE);
        let loader = match kind {
            Kind::Height => Loader::Height(AsyncTileLoader::new_with_reader(
                root,
                Vec2::ONE,
                2,
                capacity,
                pool,
                reader,
                policy,
            )),
            Kind::Overlay => Loader::Overlay(AsyncOverlayLoader::new_with_reader(
                root,
                Vec2::ONE,
                2,
                capacity,
                pool,
                reader,
                policy,
            )),
        };
        Self {
            loader: Some(loader),
            gate,
            started: started_rx,
            exited: exited_rx,
            shutdown_observed: false,
        }
    }
    fn loader(&self) -> &Loader {
        self.loader.as_ref().unwrap()
    }
    fn started(&self) -> (TileId, ThreadId) {
        wait_for("reader start", || self.started.try_recv().ok())
    }
    fn finish(&self) -> Vec<TileId> {
        let mut completed = Vec::new();
        self.gate.release();
        wait_for("adapter completion", || {
            completed.extend(self.loader().drain(1));
            (self.loader().stats().0 == 0).then_some(())
        });
        completed
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.gate.release();
        self.loader.take();
        if !std::thread::panicking() && !self.shutdown_observed {
            wait_for("worker shutdown", || self.exited.try_recv().ok());
        }
    }
}

#[test]
fn both_adapters_preserve_duplicate_saturation_payload_and_completion_limits() {
    for kind in KINDS {
        let f = Fixture::new(kind, 2, 1, CoalescePolicy::PreferFine);
        let first = TileId::new(3, 0, 0);
        let second = TileId::new(3, 1, 0);
        let third = TileId::new(3, 2, 0);
        assert!(f.loader().request(first));
        assert_eq!(f.started().0, first);
        assert!(!f.loader().request(first));
        assert!(f.loader().request(second));
        assert!(!f.loader().request(third));
        assert_eq!(f.loader().stats(), (2, 2, 1));
        assert!(f.loader().drain(0).is_empty());
        assert!(f.loader().drain(1).is_empty());
        assert_eq!(f.loader().counters().0, 4);
        assert_eq!(f.loader().counters().1, 2);
        f.gate.release();
        assert_eq!(
            wait_for("first completion", || f.loader().drain(1).pop()),
            first
        );
        assert_eq!(f.loader().stats().0, 1);
        assert_eq!(
            wait_for("second completion", || f.loader().drain(1).pop()),
            second
        );
        assert_eq!(f.loader().stats().0, 0);
        assert_eq!(f.loader().counters().5, 2);
        assert!(f.loader().request(third));
        assert_eq!(f.finish(), vec![third]);
    }
}

#[test]
fn both_adapters_preserve_each_coalescing_policy_and_request_order() {
    let root = TileId::new(0, 0, 0);
    let child = TileId::new(1, 0, 0);
    for kind in KINDS {
        for policy in [CoalescePolicy::PreferCoarse, CoalescePolicy::PreferFine] {
            for (first, second) in [(root, child), (child, root)] {
                let f = Fixture::new(kind, 2, 1, policy);
                assert!(f.loader().request(first));
                assert_eq!(f.started().0, first);
                let accepted = matches!(policy, CoalescePolicy::PreferCoarse) == (second == root);
                assert_eq!(f.loader().request(second), accepted, "{kind:?} {policy:?}");
                let expected = if accepted {
                    match kind {
                        Kind::Height => vec![first, second],
                        Kind::Overlay => vec![second],
                    }
                } else {
                    vec![first]
                };
                assert_eq!(f.loader().stats().0, expected.len());
                assert_eq!(f.finish(), expected, "{kind:?} {policy:?}");
                assert_eq!(f.loader().counters().5, expected.len());
            }
        }
    }
}

#[test]
fn both_adapters_preserve_cancellation_capacity_and_counter_policies() {
    for kind in KINDS {
        let f = Fixture::new(kind, 2, 1, CoalescePolicy::PreferFine);
        let first = TileId::new(3, 0, 0);
        let second = TileId::new(3, 1, 0);
        let replacement = TileId::new(3, 2, 0);
        assert!(f.loader().request(first));
        assert_eq!(f.started().0, first);
        assert!(f.loader().request(second));
        let cancelled_before = f.loader().counters().3;
        assert_eq!(f.loader().cancel(&[first, second, first]), 2);
        assert_eq!(f.loader().cancel(&[first, second]), 0);
        match kind {
            Kind::Height => {
                assert_eq!(f.loader().stats().0, 2);
                assert!(!f.loader().request(replacement));
                assert_eq!(cancelled_before, 0);
                assert_eq!(f.loader().counters().3, cancelled_before + 2);
                assert!(f.finish().is_empty());
                assert!(f.loader().request(replacement));
                assert_eq!(f.finish(), vec![replacement]);
            }
            Kind::Overlay => {
                assert_eq!(f.loader().stats().0, 0);
                // Preserve the existing PreferFine admission counter, even
                // when no ancestor was present, and explicit cancel's no-op.
                assert_eq!(cancelled_before, 2);
                assert_eq!(f.loader().counters().3, cancelled_before);
                assert!(f.loader().request(replacement));
                assert_eq!(f.finish(), vec![replacement]);
            }
        }
        assert_eq!(f.loader().counters().5, 1);
    }
}

#[test]
fn both_adapters_keep_pool_parallelism_and_overlay_round_robin_priority() {
    for kind in KINDS {
        let f = Fixture::new(kind, 3, 2, CoalescePolicy::PreferFine);
        let tiles = [
            TileId::new(3, 0, 0),
            TileId::new(3, 1, 0),
            TileId::new(3, 2, 0),
        ];
        assert!(f.loader().request(tiles[0]));
        let first = f.started();
        assert_eq!(first.0, tiles[0]);
        assert!(f.loader().request(tiles[1]));
        let second = f.started();
        assert_eq!(second.0, tiles[1]);
        assert_ne!(first.1, second.1);
        assert!(f.loader().request(tiles[2]));
        assert!(matches!(
            f.started.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        let mut completed = f.finish();
        let third = f.started();
        assert_eq!(third.0, tiles[2]);
        if matches!(kind, Kind::Overlay) {
            assert_eq!(third.1, first.1);
        }
        completed.sort_by_key(|id| id.x);
        assert_eq!(completed, tiles);
    }
}

#[test]
fn both_adapters_drop_without_waiting_and_release_workers_with_queued_reads() {
    for kind in KINDS {
        let mut f = Fixture::new(kind, 2, 1, CoalescePolicy::PreferFine);
        let first = TileId::new(3, 0, 0);
        let second = TileId::new(3, 1, 0);
        assert!(f.loader().request(first));
        assert_eq!(f.started().0, first);
        assert!(f.loader().request(second));
        // This must return while the reader is still blocked, then releasing
        // the reader must let every worker/dispatcher relinquish its reader.
        f.loader.take();
        f.gate.release();
        wait_for("detached reader release", || f.exited.try_recv().ok());
        let reads: Vec<_> = f.started.try_iter().map(|(tile, _)| tile).collect();
        assert_eq!(
            reads,
            match kind {
                Kind::Height => vec![],
                Kind::Overlay => vec![second],
            }
        );
        f.shutdown_observed = true;
    }
}

#[test]
fn both_adapters_preserve_zero_configuration_and_existing_worker_policies() {
    for kind in KINDS {
        let f = Fixture::new(kind, 0, 0, CoalescePolicy::PreferCoarse);
        assert_eq!(f.loader().stats(), (0, 1, 1));
    }
    let f = Fixture::new(
        Kind::Overlay,
        1,
        MAX_LOADER_WORKERS + 1,
        CoalescePolicy::PreferCoarse,
    );
    assert_eq!(
        f.loader().stats().2,
        MAX_LOADER_WORKERS + 1,
        "height's worker ceiling must not become a new overlay budget"
    );
}
