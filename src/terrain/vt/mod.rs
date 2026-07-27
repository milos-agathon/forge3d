//! TESSELLA disk-backed virtual-texture stores.

pub(crate) mod requests;
mod store;

pub use store::{
    write_packed_store, MmapPageStore, PackedPage, PageBytes, PageFormat, PageKey, StoreManifest,
    StoreMetadata, VirtualTextureStore, HEIGHT_FAMILY,
};

#[cfg(feature = "cog_streaming")]
pub use store::CogPageStore;
