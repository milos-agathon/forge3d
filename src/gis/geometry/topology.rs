use crate::gis::error::{GisError, GisResult};

#[cfg(feature = "geos-topology")]
use super::model::{Coord, UNSUPPORTED_GEOMETRY_TYPE};
use super::model::{Geometry, BACKEND_UNAVAILABLE};

pub(crate) fn topology_backend_available() -> bool {
    cfg!(feature = "geos-topology")
}

pub(crate) fn require_topology_backend(operation: &str) -> GisResult<()> {
    if operation == "union_geometries" && topology_backend_available() {
        return Ok(());
    }
    Err(GisError::BackendUnavailable(format!(
        "{BACKEND_UNAVAILABLE}: geos-topology feature required for {operation}"
    )))
}

#[cfg(feature = "geos-topology")]
pub(super) fn union_polygonal(geometries: &[Geometry]) -> GisResult<Geometry> {
    use geo::algorithm::bool_ops::unary_union;
    use geo::MultiPolygon;

    let mut polygons = Vec::new();
    for geometry in geometries {
        match geometry {
            Geometry::Polygon(rings) => polygons.push(polygon_from_rings(rings)?),
            Geometry::MultiPolygon(items) => {
                for rings in items {
                    polygons.push(polygon_from_rings(rings)?);
                }
            }
            other => {
                return Err(GisError::InvalidGeometry(format!(
                    "{UNSUPPORTED_GEOMETRY_TYPE}: union_geometries supports Polygon and MultiPolygon, got {}",
                    other.geometry_type()
                )));
            }
        }
    }

    let union: MultiPolygon<f64> = unary_union(polygons.iter());
    multi_polygon_to_geometry(union)
}

#[cfg(not(feature = "geos-topology"))]
pub(super) fn union_polygonal(_geometries: &[Geometry]) -> GisResult<Geometry> {
    require_topology_backend("union_geometries")?;
    unreachable!("topology backend is feature-gated")
}

#[cfg(feature = "geos-topology")]
fn polygon_from_rings(rings: &[Vec<Coord>]) -> GisResult<geo::Polygon<f64>> {
    let exterior = rings
        .first()
        .ok_or_else(|| GisError::InvalidGeometry("empty_geometry: Polygon is empty".to_string()))?;
    Ok(geo::Polygon::new(
        line_string_from_ring(exterior),
        rings[1..]
            .iter()
            .map(|ring| line_string_from_ring(ring))
            .collect(),
    ))
}

#[cfg(feature = "geos-topology")]
fn line_string_from_ring(ring: &[Coord]) -> geo::LineString<f64> {
    geo::LineString::from(
        ring.iter()
            .map(|coord| (coord.x, coord.y))
            .collect::<Vec<_>>(),
    )
}

#[cfg(feature = "geos-topology")]
fn multi_polygon_to_geometry(value: geo::MultiPolygon<f64>) -> GisResult<Geometry> {
    let polygons = value
        .0
        .iter()
        .map(polygon_to_rings)
        .collect::<Vec<Vec<Vec<Coord>>>>();
    if polygons.len() == 1 {
        Ok(Geometry::Polygon(polygons.into_iter().next().unwrap()))
    } else {
        Ok(Geometry::MultiPolygon(polygons))
    }
}

#[cfg(feature = "geos-topology")]
fn polygon_to_rings(polygon: &geo::Polygon<f64>) -> Vec<Vec<Coord>> {
    std::iter::once(polygon.exterior())
        .chain(polygon.interiors())
        .map(line_string_to_ring)
        .collect()
}

#[cfg(feature = "geos-topology")]
fn line_string_to_ring(line: &geo::LineString<f64>) -> Vec<Coord> {
    line.coords()
        .map(|coord| Coord {
            x: coord.x,
            y: coord.y,
        })
        .collect()
}
