//! Optional geo-backed buffer implementation; never used for boolean overlay.

#[cfg(feature = "geos-topology")]
use crate::gis::error::GisError;
use crate::gis::error::GisResult;

use super::model::Geometry;
#[cfg(feature = "geos-topology")]
use super::model::{Coord, EMPTY_GEOMETRY};
#[cfg(not(feature = "geos-topology"))]
use super::topology::require_topology_backend;

#[cfg(feature = "geos-topology")]
pub(super) fn buffer_topology(
    geometry: &Geometry,
    distance: f64,
    quad_segs: usize,
) -> GisResult<Geometry> {
    use geo::algorithm::buffer::{Buffer, BufferStyle, LineCap, LineJoin};

    let angle = std::f64::consts::FRAC_PI_2 / quad_segs as f64;
    let style = BufferStyle::new(distance)
        .line_cap(LineCap::Round(angle))
        .line_join(LineJoin::Round(angle));
    let output = geometry_to_geo(geometry)?.buffer_with_style(style);
    multi_polygon_from_geo(output)
}

#[cfg(not(feature = "geos-topology"))]
pub(super) fn buffer_topology(
    _geometry: &Geometry,
    _distance: f64,
    _quad_segs: usize,
) -> GisResult<Geometry> {
    require_topology_backend("buffer_geometry")?;
    unreachable!("buffer backend is feature-gated")
}

#[cfg(feature = "geos-topology")]
fn geometry_to_geo(geometry: &Geometry) -> GisResult<geo::Geometry<f64>> {
    match geometry {
        Geometry::Empty => Err(GisError::InvalidGeometry(format!(
            "{EMPTY_GEOMETRY}: geometry is empty"
        ))),
        Geometry::Point(point) => Ok(geo::Point::new(point.x, point.y).into()),
        Geometry::LineString(line) => Ok(geo::LineString::from(
            line.iter()
                .map(|point| (point.x, point.y))
                .collect::<Vec<_>>(),
        )
        .into()),
        Geometry::Polygon(rings) => Ok(polygon_to_geo(rings)?.into()),
        Geometry::MultiPoint(points) => Ok(geo::MultiPoint(
            points
                .iter()
                .map(|point| geo::Point::new(point.x, point.y))
                .collect(),
        )
        .into()),
        Geometry::MultiLineString(lines) => Ok(geo::MultiLineString(
            lines
                .iter()
                .map(|line| {
                    geo::LineString::from(
                        line.iter()
                            .map(|point| (point.x, point.y))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect(),
        )
        .into()),
        Geometry::MultiPolygon(polygons) => Ok(geo::MultiPolygon(
            polygons
                .iter()
                .map(|rings| polygon_to_geo(rings))
                .collect::<GisResult<Vec<_>>>()?,
        )
        .into()),
        Geometry::Collection(items) => Ok(geo::Geometry::GeometryCollection(
            geo::GeometryCollection::new_from(
                items
                    .iter()
                    .map(geometry_to_geo)
                    .collect::<GisResult<Vec<_>>>()?,
            ),
        )),
    }
}

#[cfg(feature = "geos-topology")]
fn polygon_to_geo(rings: &[Vec<Coord>]) -> GisResult<geo::Polygon<f64>> {
    let exterior = rings
        .first()
        .ok_or_else(|| GisError::InvalidGeometry(format!("{EMPTY_GEOMETRY}: Polygon is empty")))?;
    let line = |ring: &[Coord]| {
        geo::LineString::from(
            ring.iter()
                .map(|point| (point.x, point.y))
                .collect::<Vec<_>>(),
        )
    };
    Ok(geo::Polygon::new(
        line(exterior),
        rings[1..].iter().map(|ring| line(ring)).collect(),
    ))
}

#[cfg(feature = "geos-topology")]
fn multi_polygon_from_geo(value: geo::MultiPolygon<f64>) -> GisResult<Geometry> {
    let polygons = value
        .iter()
        .map(|polygon| {
            std::iter::once(polygon.exterior())
                .chain(polygon.interiors())
                .map(|ring| {
                    ring.coords()
                        .map(|coord| Coord {
                            x: coord.x,
                            y: coord.y,
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok(if polygons.len() == 1 {
        Geometry::Polygon(polygons.into_iter().next().unwrap())
    } else {
        Geometry::MultiPolygon(polygons)
    })
}
