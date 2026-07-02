use crate::gis::error::{GisError, GisResult};

#[cfg(feature = "geos-topology")]
use super::model::{Coord, EMPTY_GEOMETRY, UNSUPPORTED_GEOMETRY_TYPE};
use super::model::{Geometry, BACKEND_UNAVAILABLE};

pub(crate) fn topology_backend_available() -> bool {
    cfg!(feature = "geos-topology")
}

pub(crate) fn require_topology_backend(operation: &str) -> GisResult<()> {
    if matches!(
        operation,
        "union_geometries"
            | "buffer_geometry"
            | "clip_vector"
            | "intersect_vectors"
            | "dissolve_vector"
            | "simplify_geometry"
    ) && topology_backend_available()
    {
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
    multi_polygon_to_geometry(output)
}

#[cfg(not(feature = "geos-topology"))]
pub(super) fn buffer_topology(
    _geometry: &Geometry,
    _distance: f64,
    _quad_segs: usize,
) -> GisResult<Geometry> {
    require_topology_backend("buffer_geometry")?;
    unreachable!("topology backend is feature-gated")
}

#[cfg(feature = "geos-topology")]
pub(super) fn intersection_polygonal(
    left: &Geometry,
    right: &Geometry,
    operation: &str,
) -> GisResult<Geometry> {
    use geo::BooleanOps;

    let left = polygonal_to_multi_polygon(left, operation)?;
    let right = polygonal_to_multi_polygon(right, operation)?;
    multi_polygon_to_geometry(left.intersection(&right))
}

#[cfg(not(feature = "geos-topology"))]
pub(super) fn intersection_polygonal(
    _left: &Geometry,
    _right: &Geometry,
    operation: &str,
) -> GisResult<Geometry> {
    require_topology_backend(operation)?;
    unreachable!("topology backend is feature-gated")
}

#[cfg(feature = "geos-topology")]
pub(super) fn simplify_topology(
    geometry: &Geometry,
    tolerance: f64,
    preserve_topology: bool,
) -> GisResult<Geometry> {
    use geo::{SimplifyVw, SimplifyVwPreserve};

    if tolerance == 0.0 {
        return Ok(geometry.clone());
    }
    match geometry {
        Geometry::LineString(points) => {
            let line = line_string_from_ring(points);
            let simplified = if preserve_topology {
                line.simplify_vw_preserve(tolerance)
            } else {
                line.simplify_vw(tolerance)
            };
            Ok(Geometry::LineString(line_string_to_ring(&simplified)))
        }
        Geometry::MultiLineString(lines) => {
            let multilines = geo::MultiLineString(
                lines
                    .iter()
                    .map(|line| line_string_from_ring(line))
                    .collect(),
            );
            let simplified = if preserve_topology {
                multilines.simplify_vw_preserve(tolerance)
            } else {
                multilines.simplify_vw(tolerance)
            };
            Ok(Geometry::MultiLineString(
                simplified.iter().map(line_string_to_ring).collect(),
            ))
        }
        Geometry::Polygon(rings) => {
            let polygon = polygon_from_rings(rings)?;
            let simplified = if preserve_topology {
                polygon.simplify_vw_preserve(tolerance)
            } else {
                polygon.simplify_vw(tolerance)
            };
            Ok(Geometry::Polygon(polygon_to_rings(&simplified)))
        }
        Geometry::MultiPolygon(polygons) => {
            let multipolygon = multi_polygon_from_model(polygons)?;
            let simplified = if preserve_topology {
                multipolygon.simplify_vw_preserve(tolerance)
            } else {
                multipolygon.simplify_vw(tolerance)
            };
            Ok(Geometry::MultiPolygon(
                simplified.iter().map(polygon_to_rings).collect(),
            ))
        }
        other => Err(GisError::InvalidGeometry(format!(
            "{UNSUPPORTED_GEOMETRY_TYPE}: simplify_geometry supports LineString, MultiLineString, Polygon, and MultiPolygon, got {}",
            other.geometry_type()
        ))),
    }
}

#[cfg(not(feature = "geos-topology"))]
pub(super) fn simplify_topology(
    _geometry: &Geometry,
    _tolerance: f64,
    _preserve_topology: bool,
) -> GisResult<Geometry> {
    require_topology_backend("simplify_geometry")?;
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
fn geometry_to_geo(geometry: &Geometry) -> GisResult<geo::Geometry<f64>> {
    match geometry {
        Geometry::Empty => Err(GisError::InvalidGeometry(format!(
            "{EMPTY_GEOMETRY}: geometry is empty"
        ))),
        Geometry::Point(coord) => Ok(geo::Geometry::Point(geo::Point::new(coord.x, coord.y))),
        Geometry::LineString(points) => {
            Ok(geo::Geometry::LineString(line_string_from_ring(points)))
        }
        Geometry::Polygon(rings) => Ok(geo::Geometry::Polygon(polygon_from_rings(rings)?)),
        Geometry::MultiPoint(points) => Ok(geo::Geometry::MultiPoint(geo::MultiPoint(
            points
                .iter()
                .map(|coord| geo::Point::new(coord.x, coord.y))
                .collect(),
        ))),
        Geometry::MultiLineString(lines) => {
            Ok(geo::Geometry::MultiLineString(geo::MultiLineString(
                lines
                    .iter()
                    .map(|line| line_string_from_ring(line))
                    .collect(),
            )))
        }
        Geometry::MultiPolygon(polygons) => Ok(geo::Geometry::MultiPolygon(
            multi_polygon_from_model(polygons)?,
        )),
        Geometry::Collection(geometries) => {
            let items = geometries
                .iter()
                .map(geometry_to_geo)
                .collect::<GisResult<Vec<_>>>()?;
            Ok(geo::Geometry::GeometryCollection(
                geo::GeometryCollection::new_from(items),
            ))
        }
    }
}

#[cfg(feature = "geos-topology")]
fn multi_polygon_from_model(polygons: &[Vec<Vec<Coord>>]) -> GisResult<geo::MultiPolygon<f64>> {
    polygons
        .iter()
        .map(|rings| polygon_from_rings(rings))
        .collect::<GisResult<Vec<_>>>()
        .map(geo::MultiPolygon)
}

#[cfg(feature = "geos-topology")]
fn polygonal_to_multi_polygon(
    geometry: &Geometry,
    operation: &str,
) -> GisResult<geo::MultiPolygon<f64>> {
    match geometry {
        Geometry::Polygon(rings) => Ok(geo::MultiPolygon(vec![polygon_from_rings(rings)?])),
        Geometry::MultiPolygon(polygons) => multi_polygon_from_model(polygons),
        other => Err(GisError::InvalidGeometry(format!(
            "{UNSUPPORTED_GEOMETRY_TYPE}: {operation} supports Polygon and MultiPolygon, got {}",
            other.geometry_type()
        ))),
    }
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
