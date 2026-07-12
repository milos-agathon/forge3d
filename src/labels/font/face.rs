use super::{outline::PathSink, parse_named_instances, to_q26_6};
use crate::core::provenance::sha256;
use lyon_path::Path;
use std::fmt;
use std::sync::Arc;
use ttf_parser::{gdef::GlyphClass, Face, GlyphId, Tag};

#[derive(Clone)]
pub struct FontRequest {
    pub source: String,
    pub bytes: Arc<[u8]>,
    pub face_index: u32,
    pub named_instance: Option<String>,
}

impl FontRequest {
    pub fn from_bytes(source: impl Into<String>, bytes: impl Into<Arc<[u8]>>) -> Self {
        Self {
            source: source.into(),
            bytes: bytes.into(),
            face_index: 0,
            named_instance: None,
        }
    }

    pub fn with_named_instance(mut self, name: impl Into<String>) -> Self {
        self.named_instance = Some(name.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaceDescriptor {
    pub sha256: [u8; 32],
    pub face_index: u32,
    pub variations: Vec<([u8; 4], i32)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FontGlyph {
    pub font_index: usize,
    pub glyph_id: GlyphId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceMetrics {
    pub units_per_em: u16,
    pub ascender: i16,
    pub descender: i16,
    pub line_gap: i16,
    pub typographic_ascender: Option<i16>,
    pub typographic_descender: Option<i16>,
    pub typographic_line_gap: Option<i16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextError {
    InvalidFont {
        source: String,
        face_index: u32,
    },
    MissingGlyph {
        codepoint: u32,
        sources: Vec<String>,
    },
    InvalidFontIndex(usize),
    MissingOutline {
        font_index: usize,
        glyph_id: u16,
    },
    MissingHorizontalAdvance {
        font_index: usize,
        glyph_id: u16,
    },
    MalformedFvar,
    NamedInstanceNotFound {
        source: String,
        name: String,
    },
}

impl fmt::Display for TextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFont { source, face_index } => {
                write!(f, "invalid font {source} face index {face_index}")
            }
            Self::MissingGlyph { codepoint, sources } => write!(
                f,
                "missing glyph U+{codepoint:04X}; font chain tried: {}",
                sources.join(", ")
            ),
            Self::InvalidFontIndex(index) => write!(f, "invalid font index {index}"),
            Self::MissingOutline {
                font_index,
                glyph_id,
            } => write!(
                f,
                "glyph {glyph_id} in font index {font_index} has no outline"
            ),
            Self::MissingHorizontalAdvance {
                font_index,
                glyph_id,
            } => write!(
                f,
                "glyph {glyph_id} in font index {font_index} has no horizontal advance"
            ),
            Self::MalformedFvar => write!(f, "malformed fvar table"),
            Self::NamedInstanceNotFound { source, name } => {
                write!(f, "named font instance {name:?} not found in {source}")
            }
        }
    }
}

impl std::error::Error for TextError {}

#[derive(Clone)]
struct FontData {
    source: String,
    bytes: Arc<[u8]>,
    descriptor: FaceDescriptor,
}

#[derive(Clone)]
pub struct FontCollection {
    faces: Vec<FontData>,
}

impl FontCollection {
    pub fn load(requests: &[FontRequest]) -> Result<Self, TextError> {
        let mut faces = Vec::with_capacity(requests.len());
        for request in requests {
            let face = Face::parse(&request.bytes, request.face_index).map_err(|_| {
                TextError::InvalidFont {
                    source: request.source.clone(),
                    face_index: request.face_index,
                }
            })?;
            let mut variations = if let Some(name) = request.named_instance.as_deref() {
                let table = face
                    .raw_face()
                    .table(Tag::from_bytes(b"fvar"))
                    .ok_or_else(|| TextError::NamedInstanceNotFound {
                        source: request.source.clone(),
                        name: name.to_owned(),
                    })?;
                let instances = parse_named_instances(table)?;
                instances
                    .into_iter()
                    .find(|instance| {
                        face.names().into_iter().any(|record| {
                            record.name_id == instance.name_id
                                && record.to_string().as_deref() == Some(name)
                        })
                    })
                    .ok_or_else(|| TextError::NamedInstanceNotFound {
                        source: request.source.clone(),
                        name: name.to_owned(),
                    })?
                    .coordinates
            } else {
                Vec::new()
            };
            variations.sort_by_key(|(tag, _)| *tag);
            faces.push(FontData {
                source: request.source.clone(),
                bytes: Arc::clone(&request.bytes),
                descriptor: FaceDescriptor {
                    sha256: sha256(&request.bytes),
                    face_index: request.face_index,
                    variations,
                },
            });
        }
        Ok(Self { faces })
    }

    pub fn descriptors(&self) -> Vec<FaceDescriptor> {
        self.faces
            .iter()
            .map(|face| face.descriptor.clone())
            .collect()
    }

    pub fn glyph_for(&self, codepoint: char) -> Result<FontGlyph, TextError> {
        for (font_index, data) in self.faces.iter().enumerate() {
            let face = Face::parse(&data.bytes, data.descriptor.face_index).map_err(|_| {
                TextError::InvalidFont {
                    source: data.source.clone(),
                    face_index: data.descriptor.face_index,
                }
            })?;
            if let Some(glyph_id) = face.glyph_index(codepoint) {
                if glyph_id.0 != 0 {
                    return Ok(FontGlyph {
                        font_index,
                        glyph_id,
                    });
                }
            }
        }
        Err(TextError::MissingGlyph {
            codepoint: codepoint as u32,
            sources: self.faces.iter().map(|face| face.source.clone()).collect(),
        })
    }

    pub(crate) fn face(&self, index: usize) -> Result<Face<'_>, TextError> {
        let data = self
            .faces
            .get(index)
            .ok_or(TextError::InvalidFontIndex(index))?;
        let mut face = Face::parse(&data.bytes, data.descriptor.face_index).map_err(|_| {
            TextError::InvalidFont {
                source: data.source.clone(),
                face_index: data.descriptor.face_index,
            }
        })?;
        for (tag, value) in &data.descriptor.variations {
            face.set_variation(Tag::from_bytes(tag), *value as f32 / 65_536.0)
                .ok_or(TextError::MalformedFvar)?;
        }
        Ok(face)
    }

    pub fn horizontal_advance(&self, glyph: FontGlyph) -> Result<u16, TextError> {
        self.face(glyph.font_index)?
            .glyph_hor_advance(glyph.glyph_id)
            .ok_or(TextError::MissingHorizontalAdvance {
                font_index: glyph.font_index,
                glyph_id: glyph.glyph_id.0,
            })
    }

    pub fn horizontal_advance_q26_6(&self, glyph: FontGlyph) -> Result<i32, TextError> {
        Ok(to_q26_6(
            i32::from(self.horizontal_advance(glyph)?),
            self.metrics(glyph.font_index)?.units_per_em,
        ))
    }

    pub fn vertical_advance(&self, glyph: FontGlyph) -> Result<Option<u16>, TextError> {
        Ok(self
            .face(glyph.font_index)?
            .glyph_ver_advance(glyph.glyph_id))
    }

    pub fn glyph_class(&self, glyph: FontGlyph) -> Result<Option<GlyphClass>, TextError> {
        Ok(self
            .face(glyph.font_index)?
            .tables()
            .gdef
            .and_then(|table| table.glyph_class(glyph.glyph_id)))
    }

    pub fn mark_attachment_class(&self, glyph: FontGlyph) -> Result<u16, TextError> {
        Ok(self
            .face(glyph.font_index)?
            .tables()
            .gdef
            .map(|table| table.glyph_mark_attachment_class(glyph.glyph_id))
            .unwrap_or(0))
    }

    pub fn metrics(&self, font_index: usize) -> Result<FaceMetrics, TextError> {
        let face = self.face(font_index)?;
        Ok(FaceMetrics {
            units_per_em: face.units_per_em(),
            ascender: face.ascender(),
            descender: face.descender(),
            line_gap: face.line_gap(),
            typographic_ascender: face.typographic_ascender(),
            typographic_descender: face.typographic_descender(),
            typographic_line_gap: face.typographic_line_gap(),
        })
    }

    pub fn outline(&self, font_index: usize, glyph_id: GlyphId) -> Result<Path, TextError> {
        let face = self.face(font_index)?;
        let mut builder = Path::builder();
        let mut sink = PathSink::new(&mut builder, 1.0, glam::Vec2::ZERO);
        face.outline_glyph(glyph_id, &mut sink)
            .ok_or(TextError::MissingOutline {
                font_index,
                glyph_id: glyph_id.0,
            })?;
        Ok(builder.build())
    }
}
