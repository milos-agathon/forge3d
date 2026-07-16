pub mod arabic;
pub mod bidi;
pub mod devanagari;
pub mod gpos;
pub mod gsub;
pub mod layout;
pub mod linebreak;
mod linebreak_emoji;
mod linebreak_rules;
pub mod ot;
mod types;

pub use layout::{FeatureSetting, LayoutTable};
pub use types::{Direction, ShapedGlyph, ShapedRun, ShapedText, TextError};

use crate::labels::font::FontCollection;
use crate::labels::unicode::{script as unicode_script, Script};
use gpos::PositioningGlyph;
use gsub::{Glyph, GsubLookup};
use ot::Reader;
use std::ops::Range;
use std::sync::Arc;
use ttf_parser::Tag;

#[derive(Clone, Copy)]
struct Character {
    byte: usize,
    end: usize,
    level: u8,
    script: Tag,
    font_index: Option<usize>,
    render: bool,
    mandatory_break: bool,
}

fn is_mandatory_break(ch: char) -> bool {
    matches!(ch, '\n' | '\r' | '\u{0085}' | '\u{2028}' | '\u{2029}')
}

fn is_default_ignorable(ch: char) -> bool {
    matches!(
        ch,
        '\u{00AD}'
            | '\u{034F}'
            | '\u{061C}'
            | '\u{115F}'..='\u{1160}'
            | '\u{17B4}'..='\u{17B5}'
            | '\u{180B}'..='\u{180F}'
            | '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{206F}'
            | '\u{3164}'
            | '\u{FE00}'..='\u{FE0F}'
            | '\u{FEFF}'
            | '\u{FFA0}'
            | '\u{1BCA0}'..='\u{1BCA3}'
            | '\u{1D173}'..='\u{1D17A}'
            | '\u{E0000}'..='\u{E0FFF}'
    )
}

fn script_tag(script: Script) -> Result<Tag, TextError> {
    let bytes = match script {
        Script::Latin => b"latn",
        Script::Arabic => b"arab",
        Script::Hebrew => b"hebr",
        Script::Devanagari => b"deva",
        Script::Han => b"hani",
        other => return Err(TextError::UnsupportedScript(format!("{other:?}"))),
    };
    Ok(Tag::from_bytes(bytes))
}

fn parse_tag(value: &str, kind: &'static str) -> Result<Tag, TextError> {
    let bytes: [u8; 4] = value
        .as_bytes()
        .try_into()
        .map_err(|_| TextError::MalformedOpenType(kind))?;
    Ok(Tag::from_bytes(&bytes))
}

fn resolved_scripts(text: &str, override_tag: Option<Tag>) -> Result<Vec<Tag>, TextError> {
    if let Some(tag) = override_tag {
        return Ok(vec![tag; text.chars().count()]);
    }
    let raw: Vec<_> = text.chars().map(unicode_script).collect();
    let mut output = Vec::with_capacity(raw.len());
    for index in 0..raw.len() {
        let script = match raw[index] {
            Script::Common | Script::Inherited => raw[..index]
                .iter()
                .rev()
                .copied()
                .find(|item| !matches!(item, Script::Common | Script::Inherited))
                .or_else(|| {
                    raw[index + 1..]
                        .iter()
                        .copied()
                        .find(|item| !matches!(item, Script::Common | Script::Inherited))
                })
                .unwrap_or(Script::Latin),
            script => script,
        };
        output.push(script_tag(script)?);
    }
    Ok(output)
}

fn characters(
    text: &str,
    fonts: &FontCollection,
    levels: &[u8],
    scripts: &[Tag],
) -> Result<Vec<Character>, TextError> {
    let mut output = text
        .char_indices()
        .enumerate()
        .map(|(index, (byte, ch))| {
            let mandatory_break = is_mandatory_break(ch);
            let render = !mandatory_break && !is_default_ignorable(ch);
            let font_index = render
                .then(|| fonts.glyph_for(ch).map(|glyph| glyph.font_index))
                .transpose()
                .map_err(TextError::Font)?;
            Ok(Character {
                byte,
                end: byte + ch.len_utf8(),
                level: levels[index],
                script: scripts[index],
                font_index,
                render,
                mandatory_break,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for index in 0..output.len() {
        if output[index].font_index.is_some() || output[index].mandatory_break {
            continue;
        }
        output[index].font_index = output[..index]
            .iter()
            .rev()
            .take_while(|item| !item.mandatory_break)
            .find_map(|item| item.font_index)
            .or_else(|| {
                output[index + 1..]
                    .iter()
                    .take_while(|item| !item.mandatory_break)
                    .find_map(|item| item.font_index)
            });
    }
    Ok(output)
}

fn spans(characters: &[Character]) -> Vec<Range<usize>> {
    let mut output = Vec::new();
    let mut start = 0;
    while start < characters.len() {
        while start < characters.len()
            && (characters[start].mandatory_break || characters[start].font_index.is_none())
        {
            start += 1;
        }
        if start == characters.len() {
            break;
        }
        let first = characters[start];
        let mut end = start + 1;
        while end < characters.len()
            && !characters[end].mandatory_break
            && characters[end].font_index == first.font_index
            && characters[end].script == first.script
            && characters[end].level & 1 == first.level & 1
        {
            end += 1;
        }
        if characters[start..end].iter().any(|item| item.render) {
            output.push(start..end);
        }
        start = end;
    }
    output
}

fn selected_lookups(
    table: &[u8],
    script: Tag,
    language: Option<Tag>,
    features: &[FeatureSetting],
) -> Result<Vec<(u16, Vec<Tag>)>, TextError> {
    let layout_script = if script == Tag::from_bytes(b"deva") {
        Tag::from_bytes(b"dev2")
    } else {
        script
    };
    LayoutTable::parse(table)?.selected_feature_lookups(layout_script, language, features)
}

fn default_feature(tag: Tag, script: Tag, language: Option<Tag>) -> bool {
    let common = [b"ccmp", b"rlig", b"liga", b"clig", b"calt"];
    let positioning = [
        b"kern", b"mark", b"mkmk", b"curs", b"dist", b"abvm", b"blwm",
    ];
    let arabic = [b"isol", b"fina", b"medi", b"init"];
    let devanagari = [
        b"nukt", b"akhn", b"rphf", b"rkrf", b"pref", b"blwf", b"half", b"pstf", b"vatu", b"cjct",
        b"abvs", b"blws", b"psts", b"haln", b"pres",
    ];
    (tag == Tag::from_bytes(b"locl") && language.is_some())
        || common
            .iter()
            .chain(positioning.iter())
            .chain(
                (script == Tag::from_bytes(b"arab"))
                    .then_some(arabic.iter())
                    .into_iter()
                    .flatten(),
            )
            .chain(
                (script == Tag::from_bytes(b"deva"))
                    .then_some(devanagari.iter())
                    .into_iter()
                    .flatten(),
            )
            .any(|candidate| tag == Tag::from_bytes(candidate))
}

fn effective_features(
    table: &[u8],
    script: Tag,
    language: Option<Tag>,
    requested: &[FeatureSetting],
) -> Result<Vec<FeatureSetting>, TextError> {
    let reader = Reader::new(table);
    let feature_list = usize::from(reader.u16(6)?);
    let feature_count = usize::from(reader.u16(feature_list)?);
    let tags = (0..feature_count)
        .map(|index| {
            let offset = feature_list + 2 + index * 6;
            let bytes: [u8; 4] = reader
                .slice_at(offset, 4)?
                .try_into()
                .map_err(|_| TextError::MalformedOpenType("feature tag"))?;
            Ok(Tag::from_bytes(&bytes))
        })
        .collect::<Result<Vec<_>, TextError>>()?;
    Ok(tags
        .into_iter()
        .filter_map(|tag| {
            let setting = requested
                .iter()
                .copied()
                .find(|setting| setting.tag == tag)
                .unwrap_or_else(|| {
                    FeatureSetting::new(tag, default_feature(tag, script, language))
                });
            (!(language.is_some() && tag == Tag::from_bytes(b"locl") && setting.enabled))
                .then_some(setting)
        })
        .collect())
}

fn shape_span(
    text: &str,
    range: Range<usize>,
    chars: &[Character],
    fonts: &FontCollection,
    language: Option<Tag>,
    features: &[FeatureSetting],
) -> Result<ShapedRun, TextError> {
    let first = chars[range.start];
    let byte_start = first.byte;
    let byte_end = chars[range.end - 1].end;
    let run_text = &text[byte_start..byte_end];
    let font_index = first.font_index.expect("spans exclude unassigned controls");
    let face = fonts.face(font_index).map_err(TextError::Font)?;
    let mut glyphs: Vec<Glyph> = if first.script == Tag::from_bytes(b"deva") {
        devanagari::reorder_devanagari(run_text)
            .into_iter()
            .filter(|item| !is_default_ignorable(item.ch) && !is_mandatory_break(item.ch))
            .map(|item| {
                let id = face
                    .glyph_index(item.ch)
                    .ok_or(TextError::MalformedOpenType("cmap glyph"))?;
                Ok(Glyph::new(id, item.cluster))
            })
            .collect::<Result<_, _>>()?
    } else {
        run_text
            .char_indices()
            .filter(|(_, ch)| !is_default_ignorable(*ch) && !is_mandatory_break(*ch))
            .map(|(cluster, ch)| {
                let id = face
                    .glyph_index(ch)
                    .ok_or(TextError::MalformedOpenType("cmap glyph"))?;
                Ok(Glyph::new(id, cluster as u32))
            })
            .collect::<Result<_, _>>()?
    };

    if first.script == Tag::from_bytes(b"arab") {
        arabic::apply_feature_masks(run_text, &mut glyphs);
    } else if first.script == Tag::from_bytes(b"deva") {
        devanagari::apply_feature_masks(run_text, &mut glyphs);
    }
    if let Some(table) = face.raw_face().table(Tag::from_bytes(b"GSUB")) {
        let settings = effective_features(table, first.script, language, features)?;
        let selection = selected_lookups(table, first.script, language, &settings)?
            .into_iter()
            .map(|(index, features)| GsubLookup { index, features })
            .collect::<Vec<_>>();
        gsub::apply_gsub(&face, &mut glyphs, &selection, first.script)?;
    }
    if first.script == Tag::from_bytes(b"deva") {
        devanagari::finish_reordering(run_text, &mut glyphs);
        normalize_devanagari_mark_clusters(run_text, &mut glyphs);
    }
    normalize_zwnj_clusters(run_text, &mut glyphs);

    let mut positioned: Vec<PositioningGlyph> = glyphs.iter().map(PositioningGlyph::from).collect();
    for glyph in &mut positioned {
        glyph.x_advance = i32::from(
            face.glyph_hor_advance(glyph.id)
                .ok_or(TextError::MalformedOpenType("hmtx advance"))?,
        );
    }
    if let Some(table) = face.raw_face().table(Tag::from_bytes(b"GPOS")) {
        let settings = effective_features(table, first.script, language, features)?;
        let layout_script = if first.script == Tag::from_bytes(b"deva") {
            Tag::from_bytes(b"dev2")
        } else {
            first.script
        };
        let selection = LayoutTable::parse(table)?.selected_lookup_indices(
            layout_script,
            language,
            &settings,
        )?;
        gpos::apply_gpos(&face, &mut positioned, &selection, first.script)?;
    } else {
        gpos::apply_legacy_kern(&face, &mut positioned)?;
    }
    let units_per_em = face.units_per_em();
    let shaped = positioned
        .into_iter()
        .map(|mut glyph| {
            glyph.cluster += byte_start as u32;
            glyph.normalize(font_index, units_per_em)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let direction = if first.level & 1 == 1 {
        Direction::RightToLeft
    } else {
        Direction::LeftToRight
    };
    Ok(ShapedRun {
        text_range: byte_start..byte_end,
        glyphs: shaped,
        bidi_levels: chars[range].iter().map(|item| item.level).collect(),
        direction,
        script: first.script,
        language,
    })
}

fn normalize_zwnj_clusters(text: &str, glyphs: &mut [Glyph]) {
    for (cluster, ch) in text.char_indices() {
        if ch != '\u{200C}' {
            continue;
        }
        if let Some(glyph) = glyphs
            .iter_mut()
            .find(|glyph| glyph.cluster > cluster as u32)
        {
            glyph.cluster = cluster as u32;
        }
    }
}

fn normalize_devanagari_mark_clusters(text: &str, glyphs: &mut [Glyph]) {
    let source_char = |cluster: u32| {
        text.get(cluster as usize..)
            .and_then(|tail| tail.chars().next())
    };
    for index in 0..glyphs.len() {
        let source_chars = glyphs[index]
            .component_clusters
            .iter()
            .filter_map(|&cluster| source_char(cluster))
            .collect::<Vec<_>>();
        if source_chars.is_empty()
            || !source_chars
                .iter()
                .all(|ch| matches!(ch, '\u{093A}'..='\u{094D}' | '\u{0951}'..='\u{0957}'))
        {
            continue;
        }
        let prebase = source_chars.contains(&'\u{093F}');
        let base = if prebase {
            glyphs.get(index + 1)
        } else {
            index
                .checked_sub(1)
                .and_then(|previous| glyphs.get(previous))
        };
        if let Some(base) = base {
            glyphs[index].cluster = base.cluster;
        }
    }
}

pub fn shape(
    text: &str,
    fonts: Arc<FontCollection>,
    size: f32,
    script: Option<&str>,
    language: Option<&str>,
    features: &[FeatureSetting],
) -> Result<ShapedText, TextError> {
    if !size.is_finite() || size <= 0.0 {
        return Err(TextError::InvalidSize);
    }
    let bidi =
        bidi::resolve_bidi(text, None).map_err(|error| TextError::Bidi(error.to_string()))?;
    let script = script
        .map(|value| {
            let tag = parse_tag(value, "script tag")?;
            if [b"latn", b"arab", b"hebr", b"deva", b"hani"]
                .iter()
                .any(|supported| tag == Tag::from_bytes(supported))
            {
                Ok(tag)
            } else {
                Err(TextError::UnsupportedScript(value.to_owned()))
            }
        })
        .transpose()?;
    let language = language
        .map(|value| parse_tag(value, "language tag"))
        .transpose()?;
    let scripts = resolved_scripts(text, script)?;
    let characters = characters(text, &fonts, &bidi.levels, &scripts)?;
    let mut runs = Vec::new();
    for span in spans(&characters) {
        runs.push(shape_span(
            text,
            span,
            &characters,
            &fonts,
            language,
            features,
        )?);
    }
    Ok(ShapedText {
        text: text.to_owned(),
        runs,
        levels: bidi.levels,
        legal_breaks: linebreak::line_breaks(text),
        face_descriptors: fonts.descriptors(),
        fonts,
        size,
    })
}
