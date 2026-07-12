use super::{apply_gpos_with_data, apply_legacy_kern, PositioningGlyph};
use crate::labels::shape::TextError;
use ttf_parser::{Face, Tag};

pub(super) fn coverage(glyph: u16) -> Vec<u8> {
    [1u16, 1, glyph]
        .into_iter()
        .flat_map(u16::to_be_bytes)
        .collect()
}

pub(super) fn lookup(kind: u16, mut subtable: Vec<u8>) -> Vec<u8> {
    let mut out: Vec<u8> = [kind, 0, 1, 8]
        .into_iter()
        .flat_map(u16::to_be_bytes)
        .collect();
    out.append(&mut subtable);
    out
}

pub(super) fn lookup_with_subtables(kind: u16, subtables: Vec<Vec<u8>>) -> Vec<u8> {
    let mut out = vec![];
    out.extend_from_slice(&kind.to_be_bytes());
    out.extend_from_slice(&0u16.to_be_bytes());
    out.extend_from_slice(&(subtables.len() as u16).to_be_bytes());
    let mut offset = 6 + subtables.len() * 2;
    for subtable in &subtables {
        out.extend_from_slice(&(offset as u16).to_be_bytes());
        offset += subtable.len();
    }
    for subtable in subtables {
        out.extend_from_slice(&subtable);
    }
    out
}

pub(super) fn gpos(lookups: Vec<Vec<u8>>) -> Vec<u8> {
    let mut list = (lookups.len() as u16).to_be_bytes().to_vec();
    let mut offset = 2 + lookups.len() * 2;
    for lookup in &lookups {
        list.extend_from_slice(&(offset as u16).to_be_bytes());
        offset += lookup.len();
    }
    for lookup in lookups {
        list.extend_from_slice(&lookup);
    }
    let mut out = vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 10];
    out.extend_from_slice(&list);
    out
}

pub(super) fn glyphs(ids: &[u16]) -> Vec<PositioningGlyph> {
    ids.iter()
        .enumerate()
        .map(|(cluster, id)| PositioningGlyph::new(ttf_parser::GlyphId(*id), cluster as u32))
        .collect()
}

pub(super) fn apply_positioning_data(
    gpos: Option<&[u8]>,
    kern: Option<&[u8]>,
    buffer: &mut [PositioningGlyph],
    selection: &[u16],
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
        gpos,
        kern,
        ..ttf_parser::RawFaceTables::default()
    })
    .map_err(|_| TextError::MalformedOpenType("test face"))?;
    if gpos.is_some() {
        apply_gpos_with_data(&face, gpos, buffer, selection, script)?;
        apply_legacy_kern(&face, buffer)
    } else {
        apply_legacy_kern(&face, buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_positioning_data, coverage, glyphs, gpos, lookup, lookup_with_subtables,
        PositioningGlyph,
    };
    use ttf_parser::{GlyphId, Tag};

    #[test]
    fn single_and_both_pair_formats_accumulate_value_records() {
        let mut single = vec![0, 1, 0, 14, 0, 15, 0, 10, 0, 20, 0, 30, 0, 40];
        single.extend_from_slice(&coverage(10));
        let mut pair = vec![0, 1, 0, 18, 0, 4, 0, 0, 0, 1, 0, 12];
        pair.extend_from_slice(&[0, 1, 0, 20, 0xFF, 0xCE]);
        pair.extend_from_slice(&coverage(10));
        let mut buffer = glyphs(&[10, 20]);
        apply_positioning_data(
            Some(&gpos(vec![lookup(1, single), lookup(2, pair)])),
            None,
            &mut buffer,
            &[0, 1],
            Tag::from_bytes(b"latn"),
        )
        .unwrap();
        assert_eq!((buffer[0].x_offset, buffer[0].y_offset), (10, 20));
        assert_eq!((buffer[0].x_advance, buffer[0].y_advance), (-20, 40));

        let mut class_pair = vec![0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2];
        class_pair.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0xFF, 0xB0]);
        let coverage_at = class_pair.len();
        class_pair.extend_from_slice(&coverage(10));
        let class1_at = class_pair.len();
        class_pair.extend_from_slice(&[0, 1, 0, 10, 0, 1, 0, 1]);
        let class2_at = class_pair.len();
        class_pair.extend_from_slice(&[0, 1, 0, 20, 0, 1, 0, 1]);
        for (at, value) in [(2, coverage_at), (8, class1_at), (10, class2_at)] {
            class_pair[at..at + 2].copy_from_slice(&(value as u16).to_be_bytes());
        }
        let mut buffer = glyphs(&[10, 20]);
        apply_positioning_data(
            Some(&gpos(vec![lookup(2, class_pair)])),
            None,
            &mut buffer,
            &[0],
            Tag::from_bytes(b"latn"),
        )
        .unwrap();
        assert_eq!(buffer[0].x_advance, -80);
    }

    fn mark_fixture(kind: u16) -> Vec<u8> {
        let mut table = vec![0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0];
        let mark_cov = table.len();
        table.extend_from_slice(&coverage(20));
        let parent_cov = table.len();
        table.extend_from_slice(&coverage(10));
        let marks = table.len();
        table.extend_from_slice(&[0, 1, 0, 0, 0, 6, 0, 1, 0, 30, 0, 40]);
        let parents = table.len();
        if kind == 5 {
            table.extend_from_slice(&[0, 1, 0, 4, 0, 2, 0, 6, 0, 12]);
            table.extend_from_slice(&[0, 1, 0, 100, 0, 200, 0, 1, 0, 150, 1, 124]);
        } else {
            table.extend_from_slice(&[0, 1, 0, 4, 0, 1, 0, 150, 1, 124]);
        }
        for (at, value) in [(2, mark_cov), (4, parent_cov), (8, marks), (10, parents)] {
            table[at..at + 2].copy_from_slice(&(value as u16).to_be_bytes());
        }
        table
    }

    #[test]
    fn mark_to_base_and_mark_to_mark_attach_without_advancing() {
        for kind in [4, 6] {
            let mut buffer = glyphs(&[10, 20]);
            buffer[1].x_advance = 50;
            apply_positioning_data(
                Some(&gpos(vec![lookup(kind, mark_fixture(kind))])),
                None,
                &mut buffer,
                &[0],
                Tag::from_bytes(b"latn"),
            )
            .unwrap();
            assert_eq!((buffer[1].x_offset, buffer[1].y_offset), (120, 340));
            assert_eq!(buffer[1].x_advance, 0);
            assert_eq!(buffer[1].attached_to, Some(0));
        }

        let mut buffer = glyphs(&[10, 20]);
        buffer[0].x_advance = 500;
        apply_positioning_data(
            Some(&gpos(vec![lookup(4, mark_fixture(4))])),
            None,
            &mut buffer,
            &[0],
            Tag::from_bytes(b"latn"),
        )
        .unwrap();
        assert_eq!(buffer[1].x_offset, -380);

        for kind in [4, 5, 6] {
            let mut buffer = glyphs(&[10, 30, 20]);
            apply_positioning_data(
                Some(&gpos(vec![lookup(kind, mark_fixture(kind))])),
                None,
                &mut buffer,
                &[0],
                Tag::from_bytes(b"latn"),
            )
            .unwrap();
            assert_eq!(buffer[2].attached_to, None);
        }
    }

    #[test]
    fn mark_to_ligature_selects_component_from_cluster() {
        let mut ligature = crate::labels::shape::gsub::Glyph::new(GlyphId(10), 4);
        ligature.component_clusters = vec![8, 4];
        let mut mark = PositioningGlyph::new(GlyphId(20), 4);
        mark.x_advance = 50;
        let mut buffer = vec![PositioningGlyph::from(&ligature), mark];
        apply_positioning_data(
            Some(&gpos(vec![lookup(5, mark_fixture(5))])),
            None,
            &mut buffer,
            &[0],
            Tag::from_bytes(b"latn"),
        )
        .unwrap();
        assert_eq!((buffer[1].x_offset, buffer[1].y_offset), (120, 340));
        assert_eq!(buffer[1].ligature_component, Some(1));
    }

    #[test]
    fn lookup_subtables_are_first_match_alternatives() {
        let single = |value: i16| {
            let mut table = vec![0, 1, 0, 8, 0, 4];
            table.extend_from_slice(&value.to_be_bytes());
            table.extend_from_slice(&coverage(10));
            table
        };
        let mut buffer = glyphs(&[10]);
        let lookup = lookup_with_subtables(1, vec![single(10), single(20)]);
        apply_positioning_data(
            Some(&gpos(vec![lookup])),
            None,
            &mut buffer,
            &[0],
            Tag::from_bytes(b"latn"),
        )
        .unwrap();
        assert_eq!(buffer[0].x_advance, 10);

        let pair = |value: i16| {
            let mut table = vec![0, 1, 0, 18, 0, 4, 0, 0, 0, 1, 0, 12];
            table.extend_from_slice(&[0, 1, 0, 20]);
            table.extend_from_slice(&value.to_be_bytes());
            table.extend_from_slice(&coverage(10));
            table
        };
        let mut buffer = glyphs(&[10, 20]);
        let lookup = lookup_with_subtables(2, vec![pair(30), pair(40)]);
        apply_positioning_data(
            Some(&gpos(vec![lookup])),
            None,
            &mut buffer,
            &[0],
            Tag::from_bytes(b"latn"),
        )
        .unwrap();
        assert_eq!(buffer[0].x_advance, 30);
    }

    #[test]
    fn extension_dispatches_and_kern_runs_only_without_gpos() {
        let mut single = vec![0, 1, 0, 8, 0, 1, 0, 25];
        single.extend_from_slice(&coverage(10));
        let mut extension = vec![0, 1, 0, 1, 0, 0, 0, 8];
        extension.extend_from_slice(&single);
        let kern = [
            0, 0, 0, 1, 0, 0, 0, 20, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10, 0, 20, 0xFF, 0xD8,
        ];
        let mut buffer = glyphs(&[10, 20]);
        apply_positioning_data(
            Some(&gpos(vec![lookup(9, extension)])),
            Some(&kern),
            &mut buffer,
            &[0],
            Tag::from_bytes(b"latn"),
        )
        .unwrap();
        assert_eq!((buffer[0].x_offset, buffer[0].x_advance), (25, 0));
        let mut buffer = glyphs(&[10, 20]);
        apply_positioning_data(
            None,
            Some(&kern),
            &mut buffer,
            &[],
            Tag::from_bytes(b"latn"),
        )
        .unwrap();
        assert_eq!(buffer[0].x_advance, -40);
    }
}
