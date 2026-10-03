// src/splat/load.rs
// Reader/writer for the standard 3D Gaussian Splatting `.ply` layout
// (`x y z`, `scale_0..2`, `rot_0..3`, `opacity`, `f_dc_0..2`, `f_rest_*`).
// Parsing uses only std byte/IO primitives in the style of the mesh readers
// under src/io/ — no PLY crate. The reader is chunked so the out-of-core
// page-store builder in `stream` can ingest files that never fit in memory.
// RELEVANT FILES: src/splat/mod.rs, src/splat/stream.rs, src/io/obj_read.rs

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use super::{sh_rest_count, GaussianSplatCloud, ShRest};
use crate::core::error::RenderError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
    BinaryBigEndian,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlyScalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl PlyScalar {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Decode one scalar into f64 (exact for every PLY scalar type).
    fn decode(self, bytes: &[u8], big_endian: bool) -> f64 {
        macro_rules! read {
            ($ty:ty) => {{
                let raw: [u8; std::mem::size_of::<$ty>()] =
                    bytes[..std::mem::size_of::<$ty>()].try_into().unwrap();
                if big_endian {
                    <$ty>::from_be_bytes(raw)
                } else {
                    <$ty>::from_le_bytes(raw)
                }
            }};
        }
        match self {
            Self::I8 => f64::from(read!(i8)),
            Self::U8 => f64::from(read!(u8)),
            Self::I16 => f64::from(read!(i16)),
            Self::U16 => f64::from(read!(u16)),
            Self::I32 => f64::from(read!(i32)),
            Self::U32 => f64::from(read!(u32)),
            Self::F32 => f64::from(read!(f32)),
            Self::F64 => read!(f64),
        }
    }
}

/// Column roles of the 3DGS vertex element.
#[derive(Clone, Debug)]
struct SplatColumns {
    position: [usize; 3],
    scale: [usize; 3],
    rotation: [usize; 4],
    opacity: usize,
    dc: [usize; 3],
    /// `f_rest_k` columns ordered by `k` (channel-major: all R, then G, then B).
    rest: Vec<usize>,
}

/// Parsed header of a 3DGS `.ply`.
#[derive(Clone, Debug)]
pub struct SplatPlyHeader {
    format: PlyFormat,
    /// Number of splats in the file.
    pub count: usize,
    /// Stored spherical-harmonic degree (0..=3).
    pub sh_degree: u32,
    properties: Vec<(String, PlyScalar)>,
    offsets: Vec<usize>,
    record_size: usize,
    columns: SplatColumns,
}

/// Raw splat attributes for a contiguous run of records, already activated
/// (sigma = exp(scale), opacity = logistic(raw), unit quaternion).
#[derive(Clone, Debug, Default)]
pub struct SplatChunk {
    pub positions: Vec<[f32; 3]>,
    pub scales: Vec<[f32; 3]>,
    pub rotations: Vec<[f32; 4]>,
    pub opacities: Vec<f32>,
    pub sh0: Vec<[f32; 3]>,
    /// `[splat][coefficient][rgb]`, `sh_rest_count(degree)` per splat.
    pub sh_rest: Vec<[f32; 3]>,
}

impl SplatChunk {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// Streaming reader over the vertex element of a 3DGS `.ply`.
pub struct SplatPlyReader<R> {
    header: SplatPlyHeader,
    reader: R,
    remaining: usize,
    record: Vec<u8>,
    line: String,
}

fn format_error(path: &Path, message: impl std::fmt::Display) -> RenderError {
    RenderError::Upload(format!("splat PLY {}: {message}", path.display()))
}

fn parse_header<R: BufRead>(reader: &mut R, path: &Path) -> Result<SplatPlyHeader, RenderError> {
    let mut line = Vec::new();
    let mut next_line = |reader: &mut R| -> Result<String, RenderError> {
        line.clear();
        let read = reader.read_until(b'\n', &mut line)?;
        if read == 0 {
            return Err(format_error(path, "header ended before `end_header`"));
        }
        let text = std::str::from_utf8(&line)
            .map_err(|_| format_error(path, "header is not valid ASCII"))?;
        Ok(text.trim_end_matches(['\n', '\r']).to_string())
    };

    if next_line(reader)?.trim() != "ply" {
        return Err(format_error(path, "missing `ply` magic"));
    }
    let mut format = None;
    let mut count = None;
    let mut properties: Vec<(String, PlyScalar)> = Vec::new();
    let mut in_vertex = false;
    let mut seen_vertex = false;
    loop {
        let text = next_line(reader)?;
        let mut tokens = text.split_whitespace();
        match tokens.next() {
            Some("comment") | Some("obj_info") | None => {}
            Some("format") => {
                format = Some(match tokens.next() {
                    Some("ascii") => PlyFormat::Ascii,
                    Some("binary_little_endian") => PlyFormat::BinaryLittleEndian,
                    Some("binary_big_endian") => PlyFormat::BinaryBigEndian,
                    other => {
                        return Err(format_error(
                            path,
                            format!("unsupported format {other:?}"),
                        ))
                    }
                });
            }
            Some("element") => {
                let name = tokens.next().unwrap_or("");
                let n = tokens
                    .next()
                    .and_then(|value| value.parse::<usize>().ok())
                    .ok_or_else(|| format_error(path, "element line has no count"))?;
                if name == "vertex" {
                    if seen_vertex {
                        return Err(format_error(path, "duplicate vertex element"));
                    }
                    in_vertex = true;
                    seen_vertex = true;
                    count = Some(n);
                } else {
                    if !seen_vertex && n > 0 {
                        return Err(format_error(
                            path,
                            format!(
                                "element `{name}` precedes the vertex element; only \
                                 vertex-first 3DGS files are supported"
                            ),
                        ));
                    }
                    in_vertex = false;
                }
            }
            Some("property") => {
                if !in_vertex {
                    continue;
                }
                let ty = tokens.next().unwrap_or("");
                if ty == "list" {
                    return Err(format_error(
                        path,
                        "list properties are not valid on a splat vertex element",
                    ));
                }
                let scalar = PlyScalar::parse(ty).ok_or_else(|| {
                    format_error(path, format!("unknown property type `{ty}`"))
                })?;
                let name = tokens
                    .next()
                    .ok_or_else(|| format_error(path, "property line has no name"))?;
                properties.push((name.to_string(), scalar));
            }
            Some("end_header") => break,
            Some(other) => {
                return Err(format_error(
                    path,
                    format!("unexpected header keyword `{other}`"),
                ))
            }
        }
    }
    let format = format.ok_or_else(|| format_error(path, "missing format line"))?;
    let count = count.ok_or_else(|| format_error(path, "missing vertex element"))?;

    let column = |name: &str| -> Result<usize, RenderError> {
        properties
            .iter()
            .position(|(property, _)| property == name)
            .ok_or_else(|| {
                format_error(
                    path,
                    format!("required 3DGS property `{name}` is missing"),
                )
            })
    };
    let mut rest: Vec<(usize, usize)> = properties
        .iter()
        .enumerate()
        .filter_map(|(index, (name, _))| {
            name.strip_prefix("f_rest_")
                .and_then(|suffix| suffix.parse::<usize>().ok())
                .map(|k| (k, index))
        })
        .collect();
    rest.sort_unstable();
    if rest.iter().enumerate().any(|(i, (k, _))| *k != i) {
        return Err(format_error(
            path,
            "f_rest_* properties are not a contiguous 0-based sequence",
        ));
    }
    let sh_degree = match rest.len() {
        0 => 0,
        9 => 1,
        24 => 2,
        45 => 3,
        other => {
            return Err(format_error(
                path,
                format!("{other} f_rest_* properties do not form SH degree 0..=3 (0/9/24/45)"),
            ))
        }
    };
    let columns = SplatColumns {
        position: [column("x")?, column("y")?, column("z")?],
        scale: [column("scale_0")?, column("scale_1")?, column("scale_2")?],
        rotation: [
            column("rot_0")?,
            column("rot_1")?,
            column("rot_2")?,
            column("rot_3")?,
        ],
        opacity: column("opacity")?,
        dc: [column("f_dc_0")?, column("f_dc_1")?, column("f_dc_2")?],
        rest: rest.into_iter().map(|(_, index)| index).collect(),
    };
    let mut offsets = Vec::with_capacity(properties.len());
    let mut record_size = 0usize;
    for (_, scalar) in &properties {
        offsets.push(record_size);
        record_size += scalar.size();
    }
    Ok(SplatPlyHeader {
        format,
        count,
        sh_degree,
        properties,
        offsets,
        record_size,
        columns,
    })
}

impl SplatPlyReader<BufReader<File>> {
    /// Open a 3DGS `.ply` and parse its header.
    pub fn open(path: &Path) -> Result<Self, RenderError> {
        let file = File::open(path).map_err(|e| {
            RenderError::Upload(format!("splat PLY {}: cannot open: {e}", path.display()))
        })?;
        let mut reader = BufReader::with_capacity(1 << 20, file);
        let header = parse_header(&mut reader, path)?;
        Ok(Self {
            remaining: header.count,
            record: vec![0u8; header.record_size],
            header,
            reader,
            line: String::new(),
        })
    }
}

impl<R: BufRead> SplatPlyReader<R> {
    pub fn header(&self) -> &SplatPlyHeader {
        &self.header
    }

    /// Splats not yet returned by `read_chunk`.
    pub fn remaining(&self) -> usize {
        self.remaining
    }

    fn read_record(&mut self, values: &mut [f64]) -> Result<(), RenderError> {
        match self.header.format {
            PlyFormat::Ascii => {
                self.line.clear();
                loop {
                    if self.reader.read_line(&mut self.line)? == 0 {
                        return Err(RenderError::Upload(
                            "splat PLY ended before every vertex record was read".into(),
                        ));
                    }
                    if !self.line.trim().is_empty() {
                        break;
                    }
                    self.line.clear();
                }
                let mut tokens = self.line.split_whitespace();
                for value in values.iter_mut() {
                    *value = tokens
                        .next()
                        .and_then(|token| token.parse::<f64>().ok())
                        .ok_or_else(|| {
                            RenderError::Upload(
                                "splat PLY ASCII record has a missing or non-numeric field".into(),
                            )
                        })?;
                }
            }
            PlyFormat::BinaryLittleEndian | PlyFormat::BinaryBigEndian => {
                self.reader.read_exact(&mut self.record).map_err(|e| {
                    RenderError::Upload(format!(
                        "splat PLY ended before every vertex record was read: {e}"
                    ))
                })?;
                let big = self.header.format == PlyFormat::BinaryBigEndian;
                for (index, value) in values.iter_mut().enumerate() {
                    let (_, scalar) = self.header.properties[index];
                    *value = scalar.decode(&self.record[self.header.offsets[index]..], big);
                }
            }
        }
        Ok(())
    }

    /// Read up to `max` splats. Returns an empty chunk at end of data.
    pub fn read_chunk(&mut self, max: usize) -> Result<SplatChunk, RenderError> {
        let take = max.min(self.remaining);
        let per_rest = sh_rest_count(self.header.sh_degree);
        let mut chunk = SplatChunk {
            positions: Vec::with_capacity(take),
            scales: Vec::with_capacity(take),
            rotations: Vec::with_capacity(take),
            opacities: Vec::with_capacity(take),
            sh0: Vec::with_capacity(take),
            sh_rest: Vec::with_capacity(take * per_rest),
        };
        let mut values = vec![0.0f64; self.header.properties.len()];
        let columns = self.header.columns.clone();
        // File values are f32 attributes widened for decoding; narrowing them
        // back is exact for the float columns 3DGS writes. Positions leave
        // f64 through the typed Anchor exit (scene-local frame, zero origin).
        let narrow = |value: f64| -> f32 { f32_from_f64_attribute(value) };
        let anchor = crate::camera::Anchor::new();
        for _ in 0..take {
            self.read_record(&mut values)?;
            let [x, y, z] = columns.position.map(|c| values[c]);
            chunk.positions.push(
                anchor
                    .to_render_f32(crate::geo::units::SceneCoord::scene(glam::DVec3::new(x, y, z)))
                    .to_array(),
            );
            chunk
                .scales
                .push(columns.scale.map(|c| narrow(values[c].exp())));
            let q = columns.rotation.map(|c| values[c]);
            let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
            if !(norm.is_finite() && norm > 1e-12) {
                return Err(RenderError::Upload(format!(
                    "splat PLY record {} has a degenerate rotation quaternion",
                    self.header.count - self.remaining
                )));
            }
            chunk.rotations.push(q.map(|v| narrow(v / norm)));
            chunk
                .opacities
                .push(narrow(1.0 / (1.0 + (-values[columns.opacity]).exp())));
            chunk.sh0.push(columns.dc.map(|c| narrow(values[c])));
            for k in 0..per_rest {
                chunk.sh_rest.push([
                    narrow(values[columns.rest[k]]),
                    narrow(values[columns.rest[per_rest + k]]),
                    narrow(values[columns.rest[2 * per_rest + k]]),
                ]);
            }
            self.remaining -= 1;
        }
        Ok(chunk)
    }
}

/// Narrow a decoded, dimensionless PLY attribute (sigma, quaternion component,
/// opacity, SH coefficient) to the f32 the file stores. Positions do not pass
/// through here; they use the Anchor exit in `read_chunk`.
fn f32_from_f64_attribute(value: f64) -> f32 {
    value as f32
}

/// Load a 3DGS `.ply` fully into memory as a `GaussianSplatCloud`.
///
/// The cloud is registered with the memory tracker; a file whose SoA would
/// exceed the host budget fails with the tracker's budget error instead of
/// being silently truncated — page it with
/// `stream::build_splat_page_store_from_ply` instead.
pub fn load_gaussian_splats(path: impl AsRef<Path>) -> Result<GaussianSplatCloud, RenderError> {
    let path = path.as_ref();
    let mut reader = SplatPlyReader::open(path)?;
    let degree = reader.header().sh_degree;
    let chunk = reader.read_chunk(usize::MAX)?;
    let sh_rest = (degree > 0).then_some(ShRest {
        degree,
        coeffs: chunk.sh_rest,
    });
    GaussianSplatCloud::from_parts(
        chunk.positions,
        chunk.scales,
        chunk.rotations,
        chunk.opacities,
        chunk.sh0,
        sh_rest,
    )
}

/// Write a cloud as a binary little-endian 3DGS `.ply` (inverse activations
/// applied: log sigma, logit opacity). Opacities are clamped to
/// `[1e-6, 1 - 1e-6]` so the logit stays finite.
pub fn save_gaussian_splats(
    path: impl AsRef<Path>,
    cloud: &GaussianSplatCloud,
) -> Result<(), RenderError> {
    let path = path.as_ref();
    let per_rest = sh_rest_count(cloud.sh_degree());
    let mut out = BufWriter::new(File::create(path)?);
    let mut header = String::from("ply\nformat binary_little_endian 1.0\n");
    header.push_str("comment forge3d gaussian splat cloud\n");
    header.push_str(&format!("element vertex {}\n", cloud.len()));
    for name in ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] {
        header.push_str(&format!("property float {name}\n"));
    }
    for k in 0..per_rest * 3 {
        header.push_str(&format!("property float f_rest_{k}\n"));
    }
    for name in [
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        header.push_str(&format!("property float {name}\n"));
    }
    header.push_str("end_header\n");
    out.write_all(header.as_bytes())?;
    let mut put = |value: f32| out.write_all(&value.to_le_bytes());
    for i in 0..cloud.len() {
        for value in cloud.positions[i] {
            put(value)?;
        }
        for value in cloud.sh0[i] {
            put(value)?;
        }
        if let Some(rest) = &cloud.sh_rest {
            let coeffs = &rest.coeffs[i * per_rest..(i + 1) * per_rest];
            for channel in 0..3 {
                for coeff in coeffs {
                    put(coeff[channel])?;
                }
            }
        }
        let opacity = cloud.opacities[i].clamp(1e-6, 1.0 - 1e-6);
        put((opacity / (1.0 - opacity)).ln())?;
        for value in cloud.scales[i] {
            put(value.ln())?;
        }
        for value in cloud.rotations[i] {
            put(value)?;
        }
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cloud(degree: u32) -> GaussianSplatCloud {
        let n = 5usize;
        let per = sh_rest_count(degree);
        let f = |i: usize, k: usize| 0.125 * ((i * 7 + k * 3) % 11) as f32 - 0.5;
        GaussianSplatCloud::from_parts(
            (0..n).map(|i| [f(i, 0), f(i, 1), f(i, 2)]).collect(),
            (0..n)
                .map(|i| [0.25 + f(i, 3).abs(), 0.5, 0.125 + f(i, 4).abs()])
                .collect(),
            (0..n).map(|i| [1.0, f(i, 5), f(i, 6), f(i, 7)]).collect(),
            (0..n).map(|i| 0.1 + 0.15 * i as f32).collect(),
            (0..n).map(|i| [f(i, 8), f(i, 9), f(i, 10)]).collect(),
            (degree > 0).then(|| ShRest {
                degree,
                coeffs: (0..n * per).map(|j| [f(j, 11), f(j, 12), f(j, 13)]).collect(),
            }),
        )
        .unwrap()
    }

    fn assert_close(a: &GaussianSplatCloud, b: &GaussianSplatCloud) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.sh_degree(), b.sh_degree());
        for i in 0..a.len() {
            assert_eq!(a.positions[i], b.positions[i]);
            assert_eq!(a.sh0[i], b.sh0[i]);
            for axis in 0..3 {
                assert!((a.scales[i][axis] - b.scales[i][axis]).abs() < 1e-5);
            }
            for axis in 0..4 {
                assert!((a.rotations[i][axis] - b.rotations[i][axis]).abs() < 1e-6);
            }
            assert!((a.opacities[i] - b.opacities[i]).abs() < 1e-5);
            for k in 0..6 {
                let scale = a.inv_cov()[i][k].abs().max(1.0);
                assert!((a.inv_cov()[i][k] - b.inv_cov()[i][k]).abs() < 1e-3 * scale);
            }
        }
        assert_eq!(
            a.sh_rest.as_ref().map(|r| &r.coeffs),
            b.sh_rest.as_ref().map(|r| &r.coeffs)
        );
    }

    #[test]
    fn binary_ply_round_trips_every_sh_degree() {
        let dir = std::env::temp_dir().join(format!("forge3d-splat-ply-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for degree in 0..=3 {
            let cloud = sample_cloud(degree);
            let path = dir.join(format!("cloud-{degree}.ply"));
            save_gaussian_splats(&path, &cloud).unwrap();
            let loaded = load_gaussian_splats(&path).unwrap();
            assert_close(&cloud, &loaded);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ascii_ply_with_reordered_and_extra_properties_parses() {
        let dir = std::env::temp_dir().join(format!("forge3d-splat-ascii-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ascii.ply");
        let text = "ply\nformat ascii 1.0\ncomment reordered\nelement vertex 2\n\
            property float opacity\nproperty double x\nproperty float y\nproperty float z\n\
            property float nx\nproperty float scale_0\nproperty float scale_1\n\
            property float scale_2\nproperty float rot_0\nproperty float rot_1\n\
            property float rot_2\nproperty float rot_3\nproperty float f_dc_0\n\
            property float f_dc_1\nproperty float f_dc_2\nend_header\n\
            0.0 1.5 2.5 3.5 0 0.0 0.0 0.0 2 0 0 0 0.1 0.2 0.3\n\
            2.0 -1 -2 -3 0 -1.0 0.0 1.0 0 0 3 0 0.4 0.5 0.6\n";
        std::fs::write(&path, text).unwrap();
        let cloud = load_gaussian_splats(&path).unwrap();
        assert_eq!(cloud.len(), 2);
        assert_eq!(cloud.positions[0], [1.5, 2.5, 3.5]);
        assert_eq!(cloud.positions[1], [-1.0, -2.0, -3.0]);
        assert!((cloud.opacities[0] - 0.5).abs() < 1e-7);
        assert!((cloud.opacities[1] - 1.0 / (1.0 + (-2.0f32).exp())).abs() < 1e-6);
        assert_eq!(cloud.scales[0], [1.0, 1.0, 1.0]);
        assert!((cloud.scales[1][0] - (-1.0f32).exp()).abs() < 1e-6);
        assert_eq!(cloud.rotations[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(cloud.rotations[1], [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(cloud.sh0[1], [0.4, 0.5, 0.6]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn chunked_reads_cover_the_file_exactly_once() {
        let dir = std::env::temp_dir().join(format!("forge3d-splat-chunk-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cloud = sample_cloud(1);
        let path = dir.join("chunked.ply");
        save_gaussian_splats(&path, &cloud).unwrap();
        let mut reader = SplatPlyReader::open(&path).unwrap();
        assert_eq!(reader.header().count, 5);
        let mut seen = Vec::new();
        loop {
            let chunk = reader.read_chunk(2).unwrap();
            if chunk.is_empty() {
                break;
            }
            assert_eq!(chunk.sh_rest.len(), chunk.len() * 3);
            seen.extend(chunk.positions);
        }
        assert_eq!(seen, cloud.positions);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn malformed_files_raise_diagnostics_instead_of_rendering() {
        let dir = std::env::temp_dir().join(format!("forge3d-splat-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cases = [
            ("not-ply.ply", "plx\n", "magic"),
            (
                "no-opacity.ply",
                "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\n\
                 property float z\nproperty float scale_0\nproperty float scale_1\n\
                 property float scale_2\nproperty float rot_0\nproperty float rot_1\n\
                 property float rot_2\nproperty float rot_3\nproperty float f_dc_0\n\
                 property float f_dc_1\nproperty float f_dc_2\nend_header\n",
                "opacity",
            ),
            (
                "bad-rest.ply",
                "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\n\
                 property float z\nproperty float scale_0\nproperty float scale_1\n\
                 property float scale_2\nproperty float rot_0\nproperty float rot_1\n\
                 property float rot_2\nproperty float rot_3\nproperty float opacity\n\
                 property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n\
                 property float f_rest_0\nend_header\n",
                "SH degree",
            ),
        ];
        for (name, text, needle) in cases {
            let path = dir.join(name);
            std::fs::write(&path, text).unwrap();
            let error = format!("{}", load_gaussian_splats(&path).unwrap_err());
            assert!(error.contains(needle), "{name}: {error}");
        }
        // Truncated binary payload.
        let cloud = sample_cloud(0);
        let path = dir.join("truncated.ply");
        save_gaussian_splats(&path, &cloud).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 7]).unwrap();
        let error = format!("{}", load_gaussian_splats(&path).unwrap_err());
        assert!(error.contains("ended before"), "{error}");
        assert!(load_gaussian_splats(dir.join("missing.ply")).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
