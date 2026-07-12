use super::{apply_gsub_data, apply_gsub_data_with_gdef, Glyph, GsubLookup};
use ttf_parser::{GlyphId, Tag};

fn glyphs(ids: &[u16]) -> Vec<Glyph> {
    ids.iter()
        .enumerate()
        .map(|(cluster, id)| Glyph::new(GlyphId(*id), cluster as u32))
        .collect()
}

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

fn gsub(lookups: Vec<Vec<u8>>) -> Vec<u8> {
    let lookup_list_offset = 10u16;
    let mut list = (lookups.len() as u16).to_be_bytes().to_vec();
    let mut offset = 2 + lookups.len() * 2;
    for item in &lookups {
        list.extend_from_slice(&(offset as u16).to_be_bytes());
        offset += item.len();
    }
    for item in lookups {
        list.extend_from_slice(&item);
    }
    let mut out = Vec::from([0, 1, 0, 0, 0, 0, 0, 0]);
    out.extend_from_slice(&lookup_list_offset.to_be_bytes());
    out.extend_from_slice(&list);
    out
}

fn apply(table: &[u8], ids: &[u16], selection: &[u16]) -> Vec<Glyph> {
    let mut buffer = glyphs(ids);
    let selection: Vec<_> = selection
        .iter()
        .copied()
        .map(GsubLookup::unmasked)
        .collect();
    apply_gsub_data(table, &mut buffer, &selection, Tag::from_bytes(b"latn")).unwrap();
    buffer
}

#[test]
fn single_multiple_alternate_and_ligature_substitute() {
    let mut single = Vec::from([0, 2, 0, 8, 0, 1, 0, 99]);
    single.extend_from_slice(&coverage(10));

    let mut multiple = Vec::from([0, 1, 0, 8, 0, 1, 0, 14]);
    multiple.extend_from_slice(&coverage(20));
    multiple.extend_from_slice(&[0, 2, 0, 21, 0, 22]);

    let mut alternate = Vec::from([0, 1, 0, 8, 0, 1, 0, 14]);
    alternate.extend_from_slice(&coverage(30));
    alternate.extend_from_slice(&[0, 2, 0, 31, 0, 32]);

    let mut ligature = Vec::from([0, 1, 0, 8, 0, 1, 0, 14]);
    ligature.extend_from_slice(&coverage(40));
    ligature.extend_from_slice(&[0, 1, 0, 4, 0, 77, 0, 2, 0, 41]);

    let table = gsub(vec![
        lookup(1, single),
        lookup(2, multiple),
        lookup(3, alternate),
        lookup(4, ligature),
    ]);
    let out = apply(&table, &[10, 20, 30, 40, 41], &[0, 1, 2, 3]);
    assert_eq!(
        out.iter().map(|glyph| glyph.id.0).collect::<Vec<_>>(),
        vec![99, 21, 22, 31, 77]
    );
    assert_eq!(
        out.iter().map(|glyph| glyph.cluster).collect::<Vec<_>>(),
        vec![0, 1, 1, 2, 3]
    );
}

#[test]
fn extension_lookup_dispatches_to_nested_type() {
    let mut nested = Vec::from([0, 2, 0, 8, 0, 1, 0, 88]);
    nested.extend_from_slice(&coverage(10));
    let mut extension = Vec::from([0, 1, 0, 1, 0, 0, 0, 8]);
    extension.extend_from_slice(&nested);
    assert_eq!(
        apply(&gsub(vec![lookup(7, extension)]), &[10], &[0])[0]
            .id
            .0,
        88
    );
}

#[test]
fn chained_context_applies_nested_lookup() {
    let mut chain = Vec::from([
        0, 3, 0, 1, 0, 20, 0, 1, 0, 26, 0, 1, 0, 32, 0, 1, 0, 0, 0, 1,
    ]);
    chain.extend_from_slice(&coverage(10));
    chain.extend_from_slice(&coverage(20));
    chain.extend_from_slice(&coverage(30));
    let mut single = Vec::from([0, 2, 0, 8, 0, 1, 0, 99]);
    single.extend_from_slice(&coverage(20));
    let table = gsub(vec![lookup(6, chain), lookup(1, single)]);
    let out = apply(&table, &[10, 20, 30], &[0]);
    assert_eq!(
        out.iter().map(|glyph| glyph.id.0).collect::<Vec<_>>(),
        vec![10, 99, 30]
    );
}

#[test]
fn chained_context_formats_one_and_two_apply_nested_lookup() {
    let mut format_one = Vec::from([0, 1, 0, 8, 0, 1, 0, 14]);
    format_one.extend_from_slice(&coverage(20));
    format_one.extend_from_slice(&[0, 1, 0, 4]);
    format_one.extend_from_slice(&[0, 1, 0, 10, 0, 1, 0, 1, 0, 30, 0, 1, 0, 0, 0, 1]);

    let class = |glyph: u16| {
        let mut bytes = Vec::from([0, 1]);
        bytes.extend_from_slice(&glyph.to_be_bytes());
        bytes.extend_from_slice(&[0, 1, 0, 1]);
        bytes
    };
    let mut format_two = vec![0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0];
    let coverage_offset = format_two.len() as u16;
    format_two.extend_from_slice(&coverage(20));
    let backtrack_offset = format_two.len() as u16;
    format_two.extend_from_slice(&class(10));
    let input_offset = format_two.len() as u16;
    format_two.extend_from_slice(&class(20));
    let lookahead_offset = format_two.len() as u16;
    format_two.extend_from_slice(&class(30));
    let set_offset = format_two.len() as u16;
    format_two.extend_from_slice(&[0, 1, 0, 4]);
    format_two.extend_from_slice(&[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]);
    for (at, offset) in [
        (2, coverage_offset),
        (4, backtrack_offset),
        (6, input_offset),
        (8, lookahead_offset),
        (14, set_offset),
    ] {
        format_two[at..at + 2].copy_from_slice(&offset.to_be_bytes());
    }

    for (format, chain) in [format_one, format_two].into_iter().enumerate() {
        let mut single = Vec::from([0, 2, 0, 8, 0, 1, 0, 99]);
        single.extend_from_slice(&coverage(20));
        let out = apply(
            &gsub(vec![lookup(6, chain), lookup(1, single)]),
            &[10, 20, 30],
            &[0],
        );
        assert_eq!(
            out.iter().map(|glyph| glyph.id.0).collect::<Vec<_>>(),
            vec![10, 99, 30],
            "format {}",
            format + 1
        );
    }
}

#[test]
fn chained_records_execute_in_stored_order_and_adjust_deleted_positions() {
    let mut chain = Vec::from([
        0, 3, 0, 0, 0, 2, 0, 22, 0, 28, 0, 0, 0, 2, 0, 0, 0, 1, 0, 1, 0, 2,
    ]);
    chain.extend_from_slice(&coverage(10));
    chain.extend_from_slice(&coverage(20));

    let mut ligature = Vec::from([0, 1, 0, 8, 0, 1, 0, 14]);
    ligature.extend_from_slice(&coverage(10));
    ligature.extend_from_slice(&[0, 1, 0, 4, 0, 50, 0, 2, 0, 20]);
    let mut single = Vec::from([0, 2, 0, 8, 0, 1, 0, 99]);
    single.extend_from_slice(&coverage(20));
    let out = apply(
        &gsub(vec![
            lookup(6, chain),
            lookup(4, ligature),
            lookup(1, single),
        ]),
        &[10, 20],
        &[0],
    );
    assert_eq!(
        out.iter().map(|glyph| glyph.id.0).collect::<Vec<_>>(),
        vec![50]
    );
}

#[test]
fn lam_alef_ligature_is_selected_through_rlig() {
    let mut ligature = Vec::from([0, 1, 0, 8, 0, 1, 0, 14]);
    ligature.extend_from_slice(&coverage(10));
    ligature.extend_from_slice(&[0, 1, 0, 4, 0, 99, 0, 2, 0, 11]);
    let lookup_list = gsub(vec![lookup(4, ligature)])[10..].to_vec();

    let mut table = vec![0, 1, 0, 0, 0, 10, 0, 28, 0, 42];
    table.extend_from_slice(&[0, 1]);
    table.extend_from_slice(b"arab");
    table.extend_from_slice(&[0, 8, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0]);
    table.extend_from_slice(&[0, 1]);
    table.extend_from_slice(b"rlig");
    table.extend_from_slice(&[0, 8, 0, 0, 0, 1, 0, 0]);
    table.extend_from_slice(&lookup_list);

    let layout = crate::labels::shape::LayoutTable::parse(&table).unwrap();
    let selection = GsubLookup::selected(&layout, Tag::from_bytes(b"arab"), None, &[]).unwrap();
    assert_eq!(
        selection,
        vec![GsubLookup::for_feature(0, Tag::from_bytes(b"rlig"))]
    );
    let mut glyphs = glyphs(&[10, 11]);
    apply_gsub_data(&table, &mut glyphs, &selection, Tag::from_bytes(b"arab")).unwrap();
    assert_eq!(
        glyphs.iter().map(|glyph| glyph.id.0).collect::<Vec<_>>(),
        vec![99]
    );
}

#[test]
fn malformed_and_unknown_required_lookup_is_diagnostic() {
    let mut buffer = glyphs(&[10]);
    let error = apply_gsub_data(
        &gsub(vec![lookup(9, Vec::new())]),
        &mut buffer,
        &[GsubLookup::unmasked(0)],
        Tag::from_bytes(b"latn"),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::labels::shape::TextError::UnsupportedLookup { lookup_type: 9, .. }
    ));
}

#[path = "gsub_tests/feature.rs"]
mod feature;
