//! TESSELLA disk-backed virtual-texture stores.

mod procedural;
pub(crate) mod requests;
mod store;

pub use procedural::{
    page_tag, procedural_page, procedural_page_format, procedural_page_quadrants,
    MaterializationBand, MaterializationPlan,
};
pub use store::{
    write_packed_store, MmapPageStore, PackedPage, PageBytes, PageFormat, PageKey, StoreManifest,
    StoreMetadata, VirtualTextureStore, HEIGHT_FAMILY,
};

#[cfg(feature = "cog_streaming")]
pub use store::CogPageStore;
