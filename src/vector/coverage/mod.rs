//! LIMES analytic-coverage vector rasterization.
//!
//! This opt-in path consumes polygon rings before lyon tessellation and
//! polyline centerlines before the legacy quad expansion.  Geometry is kept as
//! directed linear edges and y-monotone circular arcs so the raster kernel can
//! integrate coverage rather than approximate it with samples.

mod binning;
mod ingest;
mod types;

pub use binning::{BinLayout, CoverageBinner, CoverageBins};
pub use ingest::CoverageGeometryBuilder;
pub use types::{
    CoverageGeometry, CoverageLayer, FillRule, PrimitiveKind, PrimitiveRecord, VectorQuality,
    COVERAGE_TILE_SIZE,
};
