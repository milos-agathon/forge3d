use std::fs::{self, File};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
#[cfg(feature = "gis-remote")]
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::gis::error::{GisError, GisResult};
use crate::gis::types::RasterWarning;

#[derive(Debug, Clone)]
pub struct RemoteDatasetInfo {
    pub url: String,
    pub cache_path: Option<PathBuf>,
    pub status: String,
    pub content_type: Option<String>,
    pub byte_size: u64,
    pub checksum: String,
    pub etag: Option<String>,
    pub last_modified: Option<String>,
    pub from_cache: bool,
    pub warnings: Vec<RasterWarning>,
}

#[derive(Debug, Clone)]
struct HttpResponse {
    body: Vec<u8>,
    content_type: Option<String>,
    etag: Option<String>,
    last_modified: Option<String>,
}

pub(crate) fn is_remote_url(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://")
}

pub(crate) fn ensure_remote_url(value: &str) -> GisResult<()> {
    if is_remote_url(value) {
        Ok(())
    } else {
        Err(GisError::InvalidArgument(
            "unsupported_scheme: only http and https URLs are supported".to_string(),
        ))
    }
}

pub fn cache_key(url: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    hex(&hasher.finalize())
}

pub fn fetch_remote_geodata(
    url: &str,
    cache_dir: Option<&Path>,
    timeout: Option<f64>,
    checksum: Option<&str>,
) -> GisResult<RemoteDatasetInfo> {
    fetch_remote_geodata_inner(url, cache_dir, timeout, checksum, true)
}

/// Fetch an explicit remote payload while retaining the bytes for an in-process
/// consumer such as the OSM or Terrarium adapters. This follows the same
/// validation, checksum, timeout, and cache policy as `fetch_remote_geodata`;
/// it is not a hidden default download surface.
pub(crate) fn fetch_remote_geodata_payload(
    url: &str,
    cache_dir: Option<&Path>,
    timeout: Option<f64>,
) -> GisResult<(Vec<u8>, RemoteDatasetInfo)> {
    ensure_remote_url(url)?;
    if let Some(path) = cache_dir.and_then(|dir| existing_cache_path_for_url(dir, url)) {
        let bytes = fs::read(&path)?;
        let info = info_from_cache(url, &path, "hit", Vec::new(), None)?;
        return Ok((bytes, info));
    }
    let response = http_get(url, timeout)?;
    validate_driver(url, response.content_type.as_deref())?;
    let digest = sha256_hex(&response.body);
    let cache_path =
        cache_dir.map(|dir| cache_path_for_url(dir, url, response.content_type.as_deref()));
    if let Some(path) = cache_path.as_ref() {
        atomic_write(path, &response.body)?;
    }
    let info = RemoteDatasetInfo {
        url: url.to_string(),
        cache_path,
        status: "fetched".to_string(),
        content_type: response.content_type,
        byte_size: response.body.len() as u64,
        checksum: format!("sha256:{digest}"),
        etag: response.etag,
        last_modified: response.last_modified,
        from_cache: false,
        warnings: Vec::new(),
    };
    Ok((response.body, info))
}

fn fetch_remote_geodata_inner(
    url: &str,
    cache_dir: Option<&Path>,
    timeout: Option<f64>,
    checksum: Option<&str>,
    allow_cache_hit: bool,
) -> GisResult<RemoteDatasetInfo> {
    ensure_remote_url(url)?;
    let existing_cache_path = cache_dir.and_then(|dir| existing_cache_path_for_url(dir, url));
    if allow_cache_hit {
        if let Some(path) = existing_cache_path.as_ref() {
            return info_from_cache(url, path, "hit", Vec::new(), checksum);
        }
    }

    let response = match http_get(url, timeout) {
        Ok(response) => response,
        Err(err) => {
            if let Some(path) = existing_cache_path.as_ref().filter(|path| path.exists()) {
                return info_from_cache(
                    url,
                    path,
                    "stale",
                    vec![RasterWarning::new(
                        "cache_stale",
                        format!("cache_stale: using stale cache after fetch failed: {err}"),
                        Some("cache"),
                    )],
                    checksum,
                );
            }
            return Err(err);
        }
    };
    validate_driver(url, response.content_type.as_deref())?;
    let digest = sha256_hex(&response.body);
    validate_checksum(&digest, checksum)?;

    let cache_path =
        cache_dir.map(|dir| cache_path_for_url(dir, url, response.content_type.as_deref()));
    if let Some(path) = cache_path.as_ref() {
        atomic_write(path, &response.body)?;
    }

    Ok(RemoteDatasetInfo {
        url: url.to_string(),
        cache_path,
        status: "fetched".to_string(),
        content_type: response.content_type,
        byte_size: response.body.len() as u64,
        checksum: format!("sha256:{digest}"),
        etag: response.etag,
        last_modified: response.last_modified,
        from_cache: false,
        warnings: Vec::new(),
    })
}

pub fn fetch_remote_geodata_bytes(
    url: &str,
    body: Vec<u8>,
    content_type: Option<String>,
    cache_dir: Option<&Path>,
    checksum: Option<&str>,
) -> GisResult<RemoteDatasetInfo> {
    ensure_remote_url(url)?;
    validate_driver(url, content_type.as_deref())?;
    let digest = sha256_hex(&body);
    validate_checksum(&digest, checksum)?;
    let cache_path = cache_dir.map(|dir| cache_path_for_url(dir, url, content_type.as_deref()));
    if let Some(path) = cache_path.as_ref() {
        atomic_write(path, &body)?;
    }
    Ok(RemoteDatasetInfo {
        url: url.to_string(),
        cache_path,
        status: "fetched".to_string(),
        content_type,
        byte_size: body.len() as u64,
        checksum: format!("sha256:{digest}"),
        etag: None,
        last_modified: None,
        from_cache: false,
        warnings: Vec::new(),
    })
}

pub fn cache_geodata(
    key_or_url: &str,
    cache_dir: &Path,
    refresh: bool,
) -> GisResult<RemoteDatasetInfo> {
    if is_remote_url(key_or_url) {
        if let Some(path) = existing_cache_path_for_url(cache_dir, key_or_url) {
            if !refresh {
                return info_from_cache(key_or_url, &path, "hit", Vec::new(), None);
            }
        }
        return fetch_remote_geodata_inner(key_or_url, Some(cache_dir), None, None, !refresh);
    }
    let path = safe_key_path(cache_dir, key_or_url)?;
    if !path.exists() {
        return Err(GisError::InvalidRaster(format!(
            "cache_miss: cached geodata key {key_or_url:?} was not found"
        )));
    }
    info_from_cache(key_or_url, &path, "hit", Vec::new(), None)
}

fn info_from_cache(
    url: &str,
    path: &Path,
    status: &str,
    warnings: Vec<RasterWarning>,
    checksum: Option<&str>,
) -> GisResult<RemoteDatasetInfo> {
    let bytes = fs::read(path)?;
    let digest = sha256_hex(&bytes);
    validate_checksum(&digest, checksum)?;
    validate_driver(path.to_string_lossy().as_ref(), None)?;
    Ok(RemoteDatasetInfo {
        url: url.to_string(),
        cache_path: Some(path.to_path_buf()),
        status: status.to_string(),
        content_type: content_type_from_path(path),
        byte_size: bytes.len() as u64,
        checksum: format!("sha256:{digest}"),
        etag: None,
        last_modified: None,
        from_cache: true,
        warnings,
    })
}

fn safe_key_path(cache_dir: &Path, key: &str) -> GisResult<PathBuf> {
    let key_path = Path::new(key);
    if key_path.components().any(|part| {
        matches!(
            part,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(GisError::InvalidArgument(
            "invalid_argument: cache key must be relative and stay inside cache_dir".to_string(),
        ));
    }
    Ok(cache_dir.join(key_path))
}

fn cache_path_for_url(cache_dir: &Path, url: &str, content_type: Option<&str>) -> PathBuf {
    let ext = extension_for(url, content_type).unwrap_or("bin");
    cache_dir.join(format!("{}.{}", cache_key(url), ext))
}

fn existing_cache_path_for_url(cache_dir: &Path, url: &str) -> Option<PathBuf> {
    let preferred = cache_path_for_url(cache_dir, url, None);
    if preferred.exists() {
        return Some(preferred);
    }
    let key = cache_key(url);
    fs::read_dir(cache_dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.is_file()
                && path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .is_some_and(|stem| stem == key)
        })
}

fn extension_for(url_or_path: &str, content_type: Option<&str>) -> Option<&'static str> {
    let lower = url_or_path
        .split(['?', '#'])
        .next()
        .unwrap_or(url_or_path)
        .to_ascii_lowercase();
    if lower.ends_with(".geojson") {
        return Some("geojson");
    }
    if lower.ends_with(".json") {
        return Some("json");
    }
    if lower.ends_with(".tif") || lower.ends_with(".tiff") {
        return Some("tif");
    }
    if lower.ends_with(".png") {
        return Some("png");
    }
    let ct = content_type?
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    match ct.as_str() {
        "application/geo+json" | "application/geojson" => Some("geojson"),
        "application/json" => Some("json"),
        "image/tiff" | "image/geotiff" | "application/geotiff" => Some("tif"),
        "image/png" => Some("png"),
        _ => None,
    }
}

fn content_type_from_path(path: &Path) -> Option<String> {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
    {
        Some(ext) if ext == "geojson" => Some("application/geo+json".to_string()),
        Some(ext) if ext == "json" => Some("application/json".to_string()),
        Some(ext) if ext == "tif" || ext == "tiff" => Some("image/tiff".to_string()),
        Some(ext) if ext == "png" => Some("image/png".to_string()),
        _ => None,
    }
}

fn validate_driver(url: &str, content_type: Option<&str>) -> GisResult<()> {
    if let Some(content_type) = content_type {
        let media_type = content_type
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        if matches!(
            media_type.as_str(),
            "application/geo+json"
                | "application/geojson"
                | "application/json"
                | "image/tiff"
                | "image/geotiff"
                | "application/geotiff"
                | "image/png"
        ) {
            return Ok(());
        }
        if media_type == "application/octet-stream" && extension_for(url, None).is_some() {
            return Ok(());
        }
    } else if extension_for(url, None).is_some() {
        return Ok(());
    }
    Err(GisError::UnsupportedDriver(
        "unsupported_driver: remote geodata content type or extension is not supported".to_string(),
    ))
}

fn validate_checksum(actual_hex: &str, expected: Option<&str>) -> GisResult<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let expected = expected.strip_prefix("sha256:").unwrap_or(expected);
    if expected.len() != 64 || !expected.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err(GisError::InvalidArgument(
            "invalid_argument: checksum must be sha256:<hex> or a 64-hex SHA-256".to_string(),
        ));
    }
    if !actual_hex.eq_ignore_ascii_case(expected) {
        return Err(GisError::InvalidRaster(
            "checksum_mismatch: fetched bytes do not match expected SHA-256".to_string(),
        ));
    }
    Ok(())
}

fn atomic_write(path: &Path, data: &[u8]) -> GisResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_file_name(format!(
        ".{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("cache")
    ));
    {
        let mut file = File::create(&tmp)?;
        file.write_all(data)?;
        file.sync_all()?;
    }
    if let Err(err) = fs::rename(&tmp, path) {
        if err.kind() == std::io::ErrorKind::AlreadyExists && path.exists() {
            fs::remove_file(path)?;
            fs::rename(&tmp, path).inspect_err(|_| {
                let _ = fs::remove_file(&tmp);
            })?;
        } else {
            let _ = fs::remove_file(&tmp);
            return Err(err.into());
        }
    }
    if let Some(parent) = path.parent() {
        let _ = File::open(parent).and_then(|file| file.sync_all());
    }
    Ok(())
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(feature = "gis-remote")]
fn http_get(url: &str, timeout: Option<f64>) -> GisResult<HttpResponse> {
    let timeout = Duration::from_secs_f64(timeout.unwrap_or(30.0).max(0.001));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| GisError::InvalidRaster(format!("network_error: {err}")))?;
    runtime.block_on(async move {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|err| GisError::InvalidRaster(format!("network_error: {err}")))?;
        let response = client.get(url).send().await.map_err(|err| {
            if err.is_timeout() {
                GisError::InvalidRaster(format!("network_timeout: {err}"))
            } else {
                GisError::InvalidRaster(format!("network_error: {err}"))
            }
        })?;
        let status = response.status();
        if status.as_u16() == 429 {
            return Err(GisError::InvalidRaster(
                "rate_limited: remote service returned HTTP 429".to_string(),
            ));
        }
        if !status.is_success() {
            return Err(GisError::InvalidRaster(format!(
                "malformed_payload: remote service returned HTTP {status}"
            )));
        }
        let headers = response.headers().clone();
        let content_type = headers
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let etag = headers
            .get(reqwest::header::ETAG)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let last_modified = headers
            .get(reqwest::header::LAST_MODIFIED)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let body = response
            .bytes()
            .await
            .map_err(|err| GisError::InvalidRaster(format!("malformed_payload: {err}")))?
            .to_vec();
        Ok(HttpResponse {
            body,
            content_type,
            etag,
            last_modified,
        })
    })
}

#[cfg(not(feature = "gis-remote"))]
fn http_get(_url: &str, _timeout: Option<f64>) -> GisResult<HttpResponse> {
    Err(GisError::BackendUnavailable(
        "backend_unavailable: gis-remote feature required for remote fetch".to_string(),
    ))
}

#[cfg(feature = "extension-module")]
pub use py::{cache_geodata_py, fetch_remote_geodata_py, fetch_vector_py};

#[cfg(feature = "extension-module")]
mod py {
    use std::path::PathBuf;

    use pyo3::prelude::*;
    use pyo3::types::{PyAny, PyDict, PyDictMethods};
    use pyo3::IntoPy;
    use serde_json::Value;

    use crate::gis::error::GisError;
    use crate::gis::py_json::{json_to_py, warnings_to_py};
    use crate::gis::remote::{
        cache_geodata, fetch_remote_geodata, fetch_remote_geodata_bytes, is_remote_url,
        RemoteDatasetInfo,
    };
    use crate::gis::vector::{read_vector, VectorInfo, VectorReadOptions};

    #[pyfunction(name = "fetch_remote_geodata", signature = (url, cache = None, timeout = None, checksum = None))]
    pub fn fetch_remote_geodata_py(
        py: Python<'_>,
        url: String,
        cache: Option<&Bound<'_, PyAny>>,
        timeout: Option<f64>,
        checksum: Option<String>,
    ) -> PyResult<PyObject> {
        let cache_dir = cache_dir_from_py(cache)?;
        let result = if let Some((body, content_type)) = mock_response_from_py(cache)? {
            fetch_remote_geodata_bytes(
                &url,
                body,
                content_type,
                cache_dir.as_deref(),
                checksum.as_deref(),
            )?
        } else {
            let url = url.clone();
            py.allow_threads(|| {
                fetch_remote_geodata(&url, cache_dir.as_deref(), timeout, checksum.as_deref())
            })?
        };
        remote_info_to_py(py, &result)
    }

    #[pyfunction(name = "cache_geodata", signature = (key_or_url, cache_dir, refresh = false))]
    pub fn cache_geodata_py(
        py: Python<'_>,
        key_or_url: String,
        cache_dir: String,
        refresh: bool,
    ) -> PyResult<PyObject> {
        let cache_dir = PathBuf::from(cache_dir);
        let result = py.allow_threads(|| cache_geodata(&key_or_url, &cache_dir, refresh))?;
        remote_info_to_py(py, &result)
    }

    #[pyfunction(name = "fetch_vector", signature = (url, cache = None))]
    pub fn fetch_vector_py(
        py: Python<'_>,
        url: String,
        cache: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        if !is_remote_url(&url) {
            return Err(GisError::InvalidArgument(
                "unsupported_scheme: only http and https URLs are supported".to_string(),
            )
            .into());
        }
        let lower = url
            .split(['?', '#'])
            .next()
            .unwrap_or(&url)
            .to_ascii_lowercase();
        if lower.ends_with(".gpkg") {
            return Err(GisError::BackendUnavailable(
                "backend_unavailable: gdal-vector feature required for GPKG".to_string(),
            )
            .into());
        }
        if !(lower.ends_with(".geojson") || lower.ends_with(".json")) {
            return Err(GisError::UnsupportedDriver(
                "unsupported_driver: fetch_vector first pass supports GeoJSON URLs".to_string(),
            )
            .into());
        }
        let cache_dir = cache_dir_from_py(cache)?.ok_or_else(|| {
            GisError::InvalidArgument(
                "invalid_argument: fetch_vector requires an explicit cache directory".to_string(),
            )
        })?;
        let remote = if let Some((body, content_type)) = mock_response_from_py(cache)? {
            fetch_remote_geodata_bytes(&url, body, content_type, Some(&cache_dir), None)?
        } else {
            let url = url.clone();
            py.allow_threads(|| fetch_remote_geodata(&url, Some(&cache_dir), None, None))?
        };
        let path = remote.cache_path.as_ref().ok_or_else(|| {
            GisError::InvalidRaster(
                "cache_miss: remote fetch did not produce a local cache path".to_string(),
            )
        })?;
        let vector = read_vector(
            path,
            VectorReadOptions {
                layer: None,
                columns: None,
                bbox: None,
                limit: None,
            },
        )?;
        let dict = PyDict::new_bound(py);
        dict.set_item("type", "FeatureCollection")?;
        dict.set_item("features", json_to_py(py, &Value::Array(vector.features))?)?;
        dict.set_item("info", vector_info_to_py(py, &vector.info)?)?;
        dict.set_item("remote", remote_info_to_py(py, &remote)?)?;
        dict.set_item("warnings", warnings_to_py(py, &vector.warnings)?)?;
        Ok(dict.into_py(py))
    }

    fn cache_dir_from_py(cache: Option<&Bound<'_, PyAny>>) -> PyResult<Option<PathBuf>> {
        let Some(cache) = cache else {
            return Ok(None);
        };
        if cache.is_none() {
            return Ok(None);
        }
        if let Ok(path) = cache.extract::<String>() {
            return Ok(Some(PathBuf::from(path)));
        }
        let dict = cache.downcast::<PyDict>().map_err(|_| {
            GisError::InvalidArgument(
                "invalid_argument: cache must be None, a path, or a dict with cache_dir"
                    .to_string(),
            )
        })?;
        dict.get_item("cache_dir")?
            .map(|value| value.extract::<String>().map(PathBuf::from))
            .transpose()
    }

    fn mock_response_from_py(
        cache: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<(Vec<u8>, Option<String>)>> {
        let Some(cache) = cache else {
            return Ok(None);
        };
        if cache.is_none() {
            return Ok(None);
        }
        let Ok(dict) = cache.downcast::<PyDict>() else {
            return Ok(None);
        };
        let Some(body) = dict.get_item("mock_body")? else {
            return Ok(None);
        };
        let body = if let Ok(bytes) = body.extract::<Vec<u8>>() {
            bytes
        } else {
            body.extract::<String>()?.into_bytes()
        };
        let content_type = dict
            .get_item("content_type")?
            .map(|value| value.extract::<String>())
            .transpose()?;
        Ok(Some((body, content_type)))
    }

    pub(crate) fn remote_info_to_py(
        py: Python<'_>,
        info: &RemoteDatasetInfo,
    ) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("url", info.url.clone())?;
        dict.set_item(
            "cache_path",
            info.cache_path
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
        )?;
        dict.set_item("status", info.status.clone())?;
        dict.set_item("content_type", info.content_type.clone())?;
        dict.set_item("byte_size", info.byte_size)?;
        dict.set_item("checksum", info.checksum.clone())?;
        dict.set_item("etag", info.etag.clone())?;
        dict.set_item("last_modified", info.last_modified.clone())?;
        dict.set_item("from_cache", info.from_cache)?;
        dict.set_item("warnings", warnings_to_py(py, &info.warnings)?)?;
        Ok(dict.into_py(py))
    }

    fn vector_info_to_py(py: Python<'_>, info: &VectorInfo) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("path", info.path.clone())?;
        dict.set_item("driver", info.driver.clone())?;
        dict.set_item("layer_name", info.layer_name.clone())?;
        dict.set_item("layer_count", info.layer_count)?;
        dict.set_item("geometry_type", info.geometry_type.clone())?;
        dict.set_item("feature_count", info.feature_count)?;
        dict.set_item("crs_wkt", info.crs_wkt.clone())?;
        dict.set_item("crs_authority", info.crs_authority.clone())?;
        dict.set_item("bounds", info.bounds)?;
        dict.set_item("is_georeferenced", info.is_georeferenced)?;
        dict.set_item("warnings", warnings_to_py(py, &info.warnings)?)?;
        Ok(dict.into_py(py))
    }
}
