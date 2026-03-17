use std::sync::mpsc::{self, Receiver, Sender};

use crate::terrain::tiling::TileId;

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
