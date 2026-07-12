use super::ot::Reader;
use super::TextError;
use ttf_parser::{gsub::SubstitutionSubtable, Face, GlyphId, Tag};

#[path = "gsub_apply.rs"]
mod apply;

#[path = "gsub_context.rs"]
mod context;
#[path = "gsub_filter.rs"]
mod filter;
pub(super) use filter::GlyphFilter;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Glyph {
    pub id: GlyphId,
    pub cluster: u32,
    pub component_clusters: Vec<u32>,
    features: Vec<Tag>,
    pub(super) origin: u64,
}

impl Glyph {
    pub fn new(id: GlyphId, cluster: u32) -> Self {
        Self {
            id,
            cluster,
            component_clusters: vec![cluster],
            features: Vec::new(),
            origin: 0,
        }
    }

    pub fn enable_feature(&mut self, feature: Tag) {
        if !self.features.contains(&feature) {
            self.features.push(feature);
        }
    }

    pub fn allows_feature(&self, feature: Tag) -> bool {
        !is_masked_feature(feature) || self.features.contains(&feature)
    }

    pub fn has_feature(&self, feature: Tag) -> bool {
        self.features.contains(&feature)
    }
}

fn is_masked_feature(feature: Tag) -> bool {
    [
        b"isol", b"init", b"medi", b"fina", b"rphf", b"half", b"pref",
    ]
    .iter()
    .any(|tag| feature == Tag::from_bytes(tag))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GsubLookup {
    pub index: u16,
    pub features: Vec<Tag>,
}

impl GsubLookup {
    pub fn unmasked(index: u16) -> Self {
        Self {
            index,
            features: Vec::new(),
        }
    }

    pub fn for_feature(index: u16, feature: Tag) -> Self {
        Self {
            index,
            features: vec![feature],
        }
    }

    pub fn selected(
        layout: &super::LayoutTable<'_>,
        script: Tag,
        language: Option<Tag>,
        settings: &[super::FeatureSetting],
    ) -> Result<Vec<Self>, TextError> {
        Ok(layout
            .selected_feature_lookups(script, language, settings)?
            .into_iter()
            .map(|(index, features)| Self { index, features })
            .collect())
    }
}

pub fn apply_gsub(
    face: &Face<'_>,
    buffer: &mut Vec<Glyph>,
    selection: &[GsubLookup],
    script: Tag,
) -> Result<(), TextError> {
    let data = face.raw_face().table(Tag::from_bytes(b"GSUB"));
    apply_gsub_with_data(face, data, buffer, selection, script)
}

fn apply_gsub_with_data(
    face: &Face<'_>,
    data: Option<&[u8]>,
    buffer: &mut Vec<Glyph>,
    selection: &[GsubLookup],
    script: Tag,
) -> Result<(), TextError> {
    let mut next_origin = buffer.iter().map(|glyph| glyph.origin).max().unwrap_or(0) + 1;
    for glyph in buffer.iter_mut() {
        if glyph.origin == 0 {
            glyph.origin = next_origin;
            next_origin += 1;
        }
    }
    let table = face
        .tables()
        .gsub
        .ok_or(TextError::MalformedOpenType("GSUB"))?;
    for selected in selection {
        if let Some(data) = data {
            validate_lookup_types(data, &[selected.index], script)?;
        }
        let lookup = table
            .lookups
            .get(selected.index)
            .ok_or(TextError::MalformedOpenType("GSUB LookupList"))?;
        let mut position = 0;
        while position < buffer.len() {
            if !selected.features.is_empty()
                && !selected
                    .features
                    .iter()
                    .any(|&feature| buffer[position].allows_feature(feature))
            {
                position += 1;
                continue;
            }
            let changed = apply_lookup_at(
                face,
                data,
                table,
                selected.index,
                lookup,
                buffer,
                position,
                script,
                0,
            )?;
            position += changed.max(1);
        }
    }
    Ok(())
}

fn validate_lookup_types(data: &[u8], selection: &[u16], script: Tag) -> Result<(), TextError> {
    let reader = Reader::new(data);
    let list = usize::from(reader.u16(8)?);
    let count = usize::from(reader.u16(list)?);
    reader.slice_at(list + 2, count * 2)?;
    for &index in selection {
        if usize::from(index) >= count {
            return Err(TextError::MalformedOpenType("GSUB LookupList"));
        }
        let lookup = list + usize::from(reader.u16(list + 2 + usize::from(index) * 2)?);
        let lookup_type = reader.u16(lookup)?;
        if !matches!(lookup_type, 1 | 2 | 3 | 4 | 6 | 7) {
            return Err(TextError::UnsupportedLookup {
                table: "GSUB",
                lookup_type,
                script,
            });
        }
        if lookup_type == 7 {
            let subtable_count = usize::from(reader.u16(lookup + 4)?);
            reader.slice_at(lookup + 6, subtable_count * 2)?;
            for subtable in 0..subtable_count {
                let offset = usize::from(reader.u16(lookup + 6 + subtable * 2)?);
                let extension = lookup + offset;
                if reader.u16(extension)? != 1 {
                    return Err(TextError::MalformedOpenType("GSUB extension"));
                }
                let nested = reader.u16(extension + 2)?;
                if !matches!(nested, 1 | 2 | 3 | 4 | 6) {
                    return Err(TextError::UnsupportedLookup {
                        table: "GSUB",
                        lookup_type: nested,
                        script,
                    });
                }
                let target = extension
                    .checked_add(reader.u32(extension + 4)? as usize)
                    .ok_or(TextError::MalformedOpenType("GSUB extension"))?;
                reader.slice_at(target, 2)?;
            }
        }
    }
    Ok(())
}

pub(super) fn apply_lookup_at(
    face: &Face<'_>,
    data: Option<&[u8]>,
    table: ttf_parser::opentype_layout::LayoutTable<'_>,
    lookup_index: u16,
    lookup: ttf_parser::opentype_layout::Lookup<'_>,
    buffer: &mut Vec<Glyph>,
    position: usize,
    script: Tag,
    depth: u8,
) -> Result<usize, TextError> {
    if depth == 16 {
        return Err(TextError::MalformedOpenType("GSUB recursion"));
    }
    if depth > 0 {
        if let Some(data) = data {
            validate_lookup_types(data, &[lookup_index], script)?;
        }
    }
    let filter = GlyphFilter::new(face, lookup);
    if buffer
        .get(position)
        .is_none_or(|glyph| filter.ignored(glyph))
    {
        return Ok(0);
    }
    for subtable in lookup.subtables.into_iter::<SubstitutionSubtable<'_>>() {
        if let Some(count) = apply::apply_subtable(
            face, data, table, filter, subtable, buffer, position, script, depth,
        )? {
            return Ok(count);
        }
    }
    Ok(0)
}

#[cfg(test)]
fn apply_gsub_data(
    data: &[u8],
    buffer: &mut Vec<Glyph>,
    selection: &[GsubLookup],
    script: Tag,
) -> Result<(), TextError> {
    apply_gsub_data_with_gdef(data, &[], buffer, selection, script)
}

#[cfg(test)]
fn apply_gsub_data_with_gdef(
    data: &[u8],
    gdef: &[u8],
    buffer: &mut Vec<Glyph>,
    selection: &[GsubLookup],
    script: Tag,
) -> Result<(), TextError> {
    let mut head = [0u8; 54];
    head[18..20].copy_from_slice(&1000u16.to_be_bytes());
    let mut hhea = [0u8; 36];
    hhea[34..36].copy_from_slice(&1u16.to_be_bytes());
    let maxp = [0, 0, 0x50, 0, 0, 100];
    let face = Face::from_raw_tables(ttf_parser::RawFaceTables {
        head: &head,
        hhea: &hhea,
        maxp: &maxp,
        gsub: Some(data),
        gdef: (!gdef.is_empty()).then_some(gdef),
        ..ttf_parser::RawFaceTables::default()
    })
    .map_err(|_| TextError::MalformedOpenType("test face"))?;
    apply_gsub_with_data(&face, Some(data), buffer, selection, script)
}

#[cfg(test)]
#[path = "gsub_tests.rs"]
mod tests;
