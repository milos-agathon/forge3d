use super::gpos_tests::apply_positioning_data;
use super::PositioningGlyph;
use crate::labels::shape::TextError;
use ttf_parser::{GlyphId, Tag};

mod tests {
    use super::*;

    fn coverage(glyph: u16) -> Vec<u8> {
        [1u16, 1, glyph]
            .into_iter()
            .flat_map(u16::to_be_bytes)
            .collect()
    }

    fn lookup(kind: u16, mut subtable: Vec<u8>) -> Vec<u8> {
        let mut out: Vec<u8> = [kind, 0, 1, 8]
            .into_iter()
            .flat_map(u16::to_be_bytes)
            .collect();
        out.append(&mut subtable);
        out
    }

    fn gpos(lookups: Vec<Vec<u8>>) -> Vec<u8> {
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

    fn glyphs(ids: &[u16]) -> Vec<PositioningGlyph> {
        ids.iter()
            .enumerate()
            .map(|(cluster, id)| PositioningGlyph::new(GlyphId(*id), cluster as u32))
            .collect()
    }

    fn mark_fixture() -> Vec<u8> {
        let mut table = vec![0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0];
        let mark_cov = table.len();
        table.extend_from_slice(&coverage(20));
        let parent_cov = table.len();
        table.extend_from_slice(&coverage(10));
        let marks = table.len();
        table.extend_from_slice(&[0, 1, 0, 0, 0, 6, 0, 1, 0, 30, 0, 40]);
        let parents = table.len();
        table.extend_from_slice(&[0, 1, 0, 4, 0, 1, 0, 150, 1, 124]);
        for (at, value) in [(2, mark_cov), (4, parent_cov), (8, marks), (10, parents)] {
            table[at..at + 2].copy_from_slice(&(value as u16).to_be_bytes());
        }
        table
    }

    #[test]
    fn rejects_unknown_and_truncated_extension_lookups() {
        let mut buffer = glyphs(&[10]);
        let error = apply_positioning_data(
            Some(&gpos(vec![lookup(3, vec![0, 1])])),
            None,
            &mut buffer,
            &[0],
            Tag::from_bytes(b"latn"),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            TextError::UnsupportedLookup {
                table: "GPOS",
                lookup_type: 3,
                ..
            }
        ));

        let extension = vec![0, 1, 0, 1, 0, 0, 1, 0];
        let error = apply_positioning_data(
            Some(&gpos(vec![lookup(9, extension)])),
            None,
            &mut buffer,
            &[0],
            Tag::from_bytes(b"latn"),
        )
        .unwrap_err();
        assert!(matches!(error, TextError::OutOfBounds { .. }));
    }

    #[test]
    fn normalizes_signed_font_units_once_with_half_away_rounding() {
        let mut glyph = PositioningGlyph::new(GlyphId(10), 0);
        glyph.x_advance = 1;
        glyph.x_offset = -1;
        let shaped = glyph.normalize(2, 128).unwrap();
        assert_eq!(shaped.font_index, 2);
        assert_eq!(shaped.x_advance, 1);
        assert_eq!(shaped.x_offset, -1);
        let glyph = PositioningGlyph::new(GlyphId(10), 0);
        assert!(matches!(
            glyph.normalize(0, 0),
            Err(TextError::MalformedOpenType("head unitsPerEm"))
        ));
    }

    #[test]
    fn positioning_overflow_is_a_structured_error() {
        let mut single = vec![0, 1, 0, 8, 0, 4, 0, 1];
        single.extend_from_slice(&coverage(10));
        let mut buffer = glyphs(&[10]);
        buffer[0].x_advance = i32::MAX;
        let error = apply_positioning_data(
            Some(&gpos(vec![lookup(1, single)])),
            None,
            &mut buffer,
            &[0],
            Tag::from_bytes(b"latn"),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            TextError::MalformedOpenType("GPOS positioning overflow")
        ));

        let mut buffer = glyphs(&[30, 10, 20]);
        buffer[0].x_advance = i32::MAX;
        buffer[1].x_advance = 1;
        let error = apply_positioning_data(
            Some(&gpos(vec![lookup(4, mark_fixture())])),
            None,
            &mut buffer,
            &[0],
            Tag::from_bytes(b"latn"),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            TextError::MalformedOpenType("GPOS positioning overflow")
        ));
    }
}
