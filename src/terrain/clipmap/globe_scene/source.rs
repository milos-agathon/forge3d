use crate::terrain::cog::py_bindings::PyCogDataset;
use crate::terrain::cog::HeightTileRequest;
use crate::terrain::tiling::TileId;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::DEFAULT_START_ALTITUDE_M;

pub(super) const OVERVIEW_SIZE: u32 = 96;
const MIN_SOURCE_LOD: u32 = 10;
const MAX_SOURCE_LOD: u32 = 14;

pub(super) fn normalize_source(source: &str) -> Result<String, String> {
    if source.starts_with("http://")
        || source.starts_with("https://")
        || source.starts_with("file://")
    {
        return Ok(source.to_string());
    }
    let path = Path::new(source);
    let absolute = if path.is_absolute() {
        PathBuf::from(path)
    } else {
        std::env::current_dir()
            .map_err(|error| format!("cannot resolve current directory: {error}"))?
            .join(path)
    };
    let absolute = absolute
        .canonicalize()
        .map_err(|error| format!("cannot resolve local COG: {error}"))?;
    let raw = absolute.to_string_lossy();
    // `canonicalize` returns an extended-length path on Windows.  Such a path
    // is valid for Win32 APIs, but `file:///\\?\C:/...` is not a valid file URI
    // for our range reader, so remove the namespace marker before URI encoding.
    let value = raw.replace(r"\\?\", "").replace('\\', "/");
    #[cfg(windows)]
    return Ok(format!("file:///{value}"));
    #[cfg(not(windows))]
    Ok(format!("file://{value}"))
}

#[cfg(all(test, windows))]
#[test]
fn absolute_windows_source_is_a_regular_file_uri() {
    let source = normalize_source(file!()).expect("source file should resolve");
    assert!(source.starts_with("file:///"));
    assert!(!source.contains("/?/"), "unexpected device path URI: {source}");
}

pub(super) fn target_tile(lon: f64, lat: f64, lod: u32) -> TileId {
    let axis = 1u32 << lod;
    let scale = f64::from(axis);
    let x = (((lon + 180.0) / 360.0 * scale).floor() as i64)
        .clamp(0, i64::from(axis - 1)) as u32;
    let y = (((90.0 - lat) / 180.0 * scale).floor() as i64)
        .clamp(0, i64::from(axis - 1)) as u32;
    TileId::new(lod, x, y)
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct OverviewSeed {
    pub tile: TileId,
    pub heights: Vec<f32>,
    pub bounds: (f64, f64, f64, f64),
}

impl OverviewSeed {
    pub fn sample_height(&self, lon: f64, lat: f64) -> Result<f32, String> {
        let (west, south, east, north) = self.bounds;
        if !lon.is_finite()
            || !lat.is_finite()
            || lon < west
            || lon > east
            || lat < south
            || lat > north
        {
            return Err(format!(
                "coordinate ({lon}, {lat}) is outside overview bounds {:?}",
                self.bounds
            ));
        }
        let size = OVERVIEW_SIZE as usize;
        if self.heights.len() != size * size || size < 2 {
            return Err("overview height dimensions are inconsistent".to_string());
        }
        let x = ((lon - west) / (east - west) * (size - 1) as f64)
            .clamp(0.0, (size - 1) as f64);
        let y = ((north - lat) / (north - south) * (size - 1) as f64)
            .clamp(0.0, (size - 1) as f64);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(size - 1);
        let y1 = (y0 + 1).min(size - 1);
        let tx = (x - x0 as f64) as f32;
        let ty = (y - y0 as f64) as f32;
        let h00 = self.heights[y0 * size + x0];
        let h10 = self.heights[y0 * size + x1];
        let h01 = self.heights[y1 * size + x0];
        let h11 = self.heights[y1 * size + x1];
        let top = h00 + (h10 - h00) * tx;
        let bottom = h01 + (h11 - h01) * tx;
        let height = top + (bottom - top) * ty;
        if !height.is_finite() {
            return Err("overview target height is not finite".to_string());
        }
        Ok(height)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct OverviewSeedCache {
    seeds: HashMap<TileId, OverviewSeed>,
}

impl OverviewSeedCache {
    pub fn new(initial: OverviewSeed) -> Self {
        Self {
            seeds: HashMap::from([(initial.tile, initial)]),
        }
    }

    pub fn get(&self, tile: TileId) -> Option<&OverviewSeed> {
        self.seeds.get(&tile)
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.seeds.len()
    }

    pub fn find_covering(&self, lon: f64, lat: f64) -> Option<TileId> {
        (MIN_SOURCE_LOD..=MAX_SOURCE_LOD)
            .map(|lod| target_tile(lon, lat, lod))
            .find(|tile| self.seeds.contains_key(tile))
    }

    pub fn prepare_with(
        &mut self,
        tile: TileId,
        load: impl FnOnce() -> Result<OverviewSeed, String>,
    ) -> Result<(), String> {
        if self.seeds.contains_key(&tile) {
            return Ok(());
        }
        let seed = load()?;
        if seed.tile != tile {
            return Err(format!(
                "overview loader returned tile {:?} for requested tile {:?}",
                seed.tile, tile
            ));
        }
        self.seeds.insert(tile, seed);
        Ok(())
    }

}

fn read_covered_seed(
    dataset: &PyCogDataset,
    tile: TileId,
) -> Result<Option<OverviewSeed>, String> {
    let read = dataset
        .reader()
        .read_height_tile_covered(HeightTileRequest {
            tile_id: tile,
            output_width: OVERVIEW_SIZE,
            output_height: OVERVIEW_SIZE,
        })
        .map_err(|error| format!("COG height read failed: {error:?}"))?;
    if !read.coverage.iter().all(|value| *value == u8::MAX)
        || !read.heights.iter().all(|value| value.is_finite())
    {
        return Ok(None);
    }
    let bounds = crate::terrain::planetary_tiles::global_tile_lonlat_bounds(tile)?;
    Ok(Some(OverviewSeed {
        tile,
        heights: read.heights,
        bounds,
    }))
}

pub(super) fn load_covered_overview(
    dataset: &PyCogDataset,
    lon: f64,
    lat: f64,
) -> Result<(Vec<f32>, u32, TileId), String> {
    for lod in MIN_SOURCE_LOD..=MAX_SOURCE_LOD {
        let tile = target_tile(lon, lat, lod);
        if let Some(seed) = read_covered_seed(dataset, tile)? {
            return Ok((seed.heights, lod, tile));
        }
    }
    Err(format!(
        "target ({lon:.6}, {lat:.6}) has no fully covered finite COG tile at LOD {MIN_SOURCE_LOD}..={MAX_SOURCE_LOD}; refusing to fabricate overview heights"
    ))
}

pub(super) fn tile_extent_m(bounds: (f64, f64, f64, f64), latitude: f64) -> f32 {
    let radius = super::super::globe::GlobeFrame::WGS84_MEAN_RADIUS_M;
    let width = (bounds.2 - bounds.0).to_radians()
        * radius
        * latitude.to_radians().cos().abs();
    let height = (bounds.3 - bounds.1).to_radians() * radius;
    width.max(height).clamp(1_000.0, 100_000.0) as f32
}

pub(super) fn source_error(
    source: &str,
    name: &str,
    lon: f64,
    lat: f64,
    error: impl std::fmt::Display,
) -> PyErr {
    PyRuntimeError::new_err(format!(
        "failed to open ORBIS COG source {source} for target {name} at ({lon}, {lat}): {error}"
    ))
}

pub(super) fn coverage_error(
    source: &str,
    name: &str,
    lon: f64,
    lat: f64,
    bounds: (f64, f64, f64, f64),
) -> PyErr {
    PyValueError::new_err(format!(
        "target {name} at ({lon}, {lat}) is outside georeferenced coverage {bounds:?} of COG source {source}"
    ))
}

pub(super) fn default_render_params(
    py: Python<'_>,
    terrain_extent_m: f32,
) -> PyResult<crate::terrain::render_params::TerrainRenderParams> {
    let module = py.import_bound("forge3d.terrain_params")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("size_px", (256, 192))?;
    kwargs.set_item("render_scale", 1.0)?;
    kwargs.set_item("terrain_span", terrain_extent_m)?;
    kwargs.set_item("msaa_samples", 1)?;
    kwargs.set_item("z_scale", 1.0)?;
    kwargs.set_item("exposure", 1.0)?;
    kwargs.set_item("domain", (0.0, 5000.0))?;
    kwargs.set_item("camera_mode", "clipmap:4:32:32:10:0.3:zup")?;
    kwargs.set_item("culling", "frustum")?;
    kwargs.set_item("shading", "forward")?;
    kwargs.set_item("cam_radius", DEFAULT_START_ALTITUDE_M as f32)?;
    kwargs.set_item("cam_phi_deg", 0.0)?;
    kwargs.set_item("cam_theta_deg", 0.0)?;
    kwargs.set_item(
        "cam_target",
        (0.0, 0.0, -(DEFAULT_START_ALTITUDE_M as f32)),
    )?;
    kwargs.set_item(
        "clip",
        (
            0.1,
            DEFAULT_START_ALTITUDE_M as f32 + terrain_extent_m * 4.0,
        ),
    )?;
    let config = module
        .getattr("make_terrain_params_config")?
        .call((), Some(&kwargs))?;
    crate::terrain::render_params::TerrainRenderParams::from_python_params(py, config)
}

#[cfg(test)]
pub(super) fn assert_test_points_in_tiles() {
    for (lon, lat) in [
        (0.0, 0.0),
        (-180.0, 90.0),
        (180.0, -90.0),
        (-121.7603, 46.8523),
    ] {
        let tile = target_tile(lon, lat, 14);
        let bounds = crate::terrain::planetary_tiles::global_tile_lonlat_bounds(tile).unwrap();
        assert!(lon >= bounds.0 && lon <= bounds.2);
        assert!(lat >= bounds.1 && lat <= bounds.3);
    }
}
pub(super) fn extract_source_path(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<String> {
    let value = py.import_bound("os")?.getattr("fspath")?.call1((source,))?;
    value.extract::<String>().map_err(|_| {
        PyTypeError::new_err("cog_source must be str or os.PathLike[str], not bytes")
    })
}

#[cfg(test)]
mod overview_seed_tests {
    use super::*;

    fn seed(tile: TileId, value: f32) -> OverviewSeed {
        OverviewSeed {
            tile,
            heights: vec![value; (OVERVIEW_SIZE * OVERVIEW_SIZE) as usize],
            bounds: crate::terrain::planetary_tiles::global_tile_lonlat_bounds(tile).unwrap(),
        }
    }

    #[test]
    fn cache_deduplicates_reads_and_failed_seed_is_transactional() {
        let initial = target_tile(-121.7603, 46.8523, 10);
        let adjacent = TileId::new(initial.lod, initial.x.saturating_sub(1), initial.y);
        let missing = TileId::new(initial.lod, initial.x.saturating_add(1), initial.y);
        let mut cache = OverviewSeedCache::new(seed(initial, 1.0));
        let mut reads = 0_u32;
        cache
            .prepare_with(adjacent, || {
                reads += 1;
                Ok(seed(adjacent, 2.0))
            })
            .unwrap();
        cache
            .prepare_with(adjacent, || {
                reads += 1;
                Ok(seed(adjacent, 99.0))
            })
            .unwrap();
        assert_eq!(reads, 1, "returning to a cached tile must not reread the COG");
        assert_eq!(cache.get(adjacent).unwrap().heights[0], 2.0);

        let before = cache.clone();
        assert!(cache
            .prepare_with(missing, || Err("deliberate later seed failure".to_string()))
            .is_err());
        assert_eq!(cache, before, "failed seed must not mutate the cache");
    }

    #[test]
    fn preload_deduplicates_every_tile_before_frame_iteration() {
        let first = target_tile(-121.7603, 46.8523, 10);
        let second = TileId::new(first.lod, first.x.saturating_sub(1), first.y);
        let mut cache = OverviewSeedCache::new(seed(first, 1.0));
        let mut reads = Vec::new();
        for tile in [first, second, second, first] {
            cache
                .prepare_with(tile, || {
                reads.push(tile);
                Ok(seed(tile, tile.x as f32))
            })
            .unwrap();
        }
        assert_eq!(reads, vec![second]);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn overview_seed_bilinear_sample_uses_top_down_lonlat_registration() {
        let mut heights = vec![0.0; (OVERVIEW_SIZE * OVERVIEW_SIZE) as usize];
        for y in 0..OVERVIEW_SIZE as usize {
            for x in 0..OVERVIEW_SIZE as usize {
                heights[y * OVERVIEW_SIZE as usize + x] = x as f32 + 100.0 * y as f32;
            }
        }
        let seed = OverviewSeed {
            tile: TileId::new(0, 0, 0),
            heights,
            bounds: (-122.0, 46.0, -121.0, 47.0),
        };
        assert_eq!(seed.sample_height(-121.5, 46.5).unwrap(), 4_797.5);
        assert!(seed.sample_height(-120.0, 46.5).is_err());
    }
}
