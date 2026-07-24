use bytemuck::{Pod, Zeroable};

/// LIMES uses fixed 16x16 screen-space bins.
pub const COVERAGE_TILE_SIZE: u32 = 16;

/// Per-layer vector quality selection.  `Default` deliberately preserves the
/// existing polygon and line pipelines.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VectorQuality {
    #[default]
    Default,
    Analytic,
}

impl VectorQuality {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "default" => Some(Self::Default),
            "analytic" => Some(Self::Analytic),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Analytic => "analytic",
        }
    }
}

/// Fill interpretation applied after all primitives in one layer have been
/// gathered.  This is what makes same-layer polygon conflation a single
/// coverage decision instead of per-triangle blending.
#[repr(u32)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FillRule {
    #[default]
    NonZero = 0,
    EvenOdd = 1,
}

impl FillRule {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "nonzero" => Some(Self::NonZero),
            "evenodd" => Some(Self::EvenOdd),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::NonZero => "nonzero",
            Self::EvenOdd => "evenodd",
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveKind {
    Line = 0,
    Arc = 1,
}

/// GPU ABI shared by the bin and raster kernels (48-byte storage stride).
///
/// Line:
/// - `geometry = [x0, y0, x1, y1]`
/// - `bounds = [min_x, min_y, max_x, max_y]`
///
/// Y-monotone circular arc:
/// - `geometry = [center_x, center_y, radius, x_branch]`
/// - `bounds = [min_x, min_y, max_x, max_y]`
/// - `x_branch` is -1 for the left and +1 for the right circle branch.
///
/// `meta = [kind, layer, winding_as_bits, stable_primitive_id]`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PrimitiveRecord {
    pub geometry: [f32; 4],
    pub bounds: [f32; 4],
    pub meta: [u32; 4],
}

impl PrimitiveRecord {
    pub fn line(p0: [f32; 2], p1: [f32; 2], layer: u32, stable_id: u32) -> Option<Self> {
        if !p0.into_iter().chain(p1).all(f32::is_finite) || p0 == p1 {
            return None;
        }
        let winding = if p1[1] > p0[1] {
            1_i32
        } else if p1[1] < p0[1] {
            -1_i32
        } else {
            0_i32
        };
        Some(Self {
            geometry: [p0[0], p0[1], p1[0], p1[1]],
            bounds: [
                p0[0].min(p1[0]),
                p0[1].min(p1[1]),
                p0[0].max(p1[0]),
                p0[1].max(p1[1]),
            ],
            meta: [PrimitiveKind::Line as u32, layer, winding as u32, stable_id],
        })
    }

    pub fn arc(
        center: [f32; 2],
        radius: f32,
        x_branch: f32,
        bounds: [f32; 4],
        winding: i32,
        layer: u32,
        stable_id: u32,
    ) -> Option<Self> {
        if !center
            .into_iter()
            .chain([radius, x_branch])
            .chain(bounds)
            .all(f32::is_finite)
            || radius <= 0.0
            || !matches!(winding, -1 | 1)
        {
            return None;
        }
        Some(Self {
            geometry: [center[0], center[1], radius, x_branch.signum()],
            bounds,
            meta: [PrimitiveKind::Arc as u32, layer, winding as u32, stable_id],
        })
    }

    pub fn kind(self) -> PrimitiveKind {
        if self.meta[0] == PrimitiveKind::Arc as u32 {
            PrimitiveKind::Arc
        } else {
            PrimitiveKind::Line
        }
    }

    pub fn layer(self) -> u32 {
        self.meta[1]
    }

    pub fn winding(self) -> i32 {
        self.meta[2] as i32
    }

    pub fn stable_id(self) -> u32 {
        self.meta[3]
    }
}

#[derive(Debug, Clone)]
pub struct CoverageLayer {
    pub fill_rule: FillRule,
    pub color: [f32; 4],
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct CoverageGeometry {
    pub width: u32,
    pub height: u32,
    pub primitives: Vec<PrimitiveRecord>,
    pub layers: Vec<CoverageLayer>,
}

impl CoverageGeometry {
    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_storage_stride_is_locked() {
        assert_eq!(std::mem::size_of::<PrimitiveRecord>(), 48);
        assert_eq!(std::mem::align_of::<PrimitiveRecord>(), 4);
    }

    #[test]
    fn vector_quality_default_is_legacy() {
        assert_eq!(VectorQuality::default(), VectorQuality::Default);
        assert_eq!(
            VectorQuality::parse("analytic"),
            Some(VectorQuality::Analytic)
        );
        assert_eq!(VectorQuality::parse("other"), None);
    }
}
