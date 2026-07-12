use super::*;

#[test]
fn arabic_form_lookups_apply_only_to_matching_glyph_masks() {
    let mut initial = Vec::from([0, 2, 0, 8, 0, 1, 0, 11]);
    initial.extend_from_slice(&coverage(10));
    let mut final_form = Vec::from([0, 2, 0, 8, 0, 1, 0, 12]);
    final_form.extend_from_slice(&coverage(10));
    let table = gsub(vec![lookup(1, initial), lookup(1, final_form)]);
    let mut buffer = vec![Glyph::new(GlyphId(10), 0), Glyph::new(GlyphId(10), 2)];
    crate::labels::shape::arabic::apply_feature_masks("بب", &mut buffer);
    apply_gsub_data(
        &table,
        &mut buffer,
        &[
            GsubLookup::for_feature(0, Tag::from_bytes(b"init")),
            GsubLookup::for_feature(1, Tag::from_bytes(b"fina")),
        ],
        Tag::from_bytes(b"arab"),
    )
    .unwrap();
    assert_eq!(
        buffer.iter().map(|glyph| glyph.id.0).collect::<Vec<_>>(),
        vec![11, 12]
    );
}

#[test]
fn ignore_marks_allows_ligature_across_transparent_glyph() {
    let mut ligature = Vec::from([0, 1, 0, 8, 0, 1, 0, 14]);
    ligature.extend_from_slice(&coverage(10));
    ligature.extend_from_slice(&[0, 1, 0, 4, 0, 99, 0, 2, 0, 11]);
    let mut marked_lookup = lookup(4, ligature);
    marked_lookup[2..4].copy_from_slice(&8u16.to_be_bytes());
    let gdef = [
        0, 1, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 20, 0, 20, 0, 3,
    ];
    let mut buffer = glyphs(&[10, 20, 11]);
    apply_gsub_data_with_gdef(
        &gsub(vec![marked_lookup]),
        &gdef,
        &mut buffer,
        &[GsubLookup::unmasked(0)],
        Tag::from_bytes(b"arab"),
    )
    .unwrap();
    assert_eq!(
        buffer.iter().map(|glyph| glyph.id.0).collect::<Vec<_>>(),
        vec![99, 20]
    );
    assert_eq!(buffer[0].component_clusters, vec![0, 2]);
}

#[test]
fn chained_context_uses_lookup_filtering_for_input_positions() {
    let mut chain = Vec::from([0, 3, 0, 0, 0, 2, 0, 18, 0, 24, 0, 0, 0, 1, 0, 1, 0, 1]);
    chain.extend_from_slice(&coverage(10));
    chain.extend_from_slice(&coverage(11));
    let mut contextual = lookup(6, chain);
    contextual[2..4].copy_from_slice(&8u16.to_be_bytes());
    let mut single = Vec::from([0, 2, 0, 8, 0, 1, 0, 99]);
    single.extend_from_slice(&coverage(11));
    let gdef = [
        0, 1, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 20, 0, 20, 0, 3,
    ];
    let mut buffer = glyphs(&[10, 20, 11]);
    apply_gsub_data_with_gdef(
        &gsub(vec![contextual, lookup(1, single)]),
        &gdef,
        &mut buffer,
        &[GsubLookup::unmasked(0)],
        Tag::from_bytes(b"arab"),
    )
    .unwrap();
    assert_eq!(
        buffer.iter().map(|glyph| glyph.id.0).collect::<Vec<_>>(),
        vec![10, 20, 99]
    );
}

fn ligature_subtable() -> Vec<u8> {
    let mut ligature = Vec::from([0, 1, 0, 8, 0, 1, 0, 14]);
    ligature.extend_from_slice(&coverage(10));
    ligature.extend_from_slice(&[0, 1, 0, 4, 0, 99, 0, 2, 0, 11]);
    ligature
}

#[test]
fn mark_filtering_set_controls_which_marks_are_skipped() {
    let mut lookup = Vec::from([0, 4, 0, 16, 0, 1, 0, 10, 0, 0]);
    lookup.extend_from_slice(&ligature_subtable());
    let gdef = [
        0, 1, 0, 2, 0, 14, 0, 0, 0, 0, 0, 0, 0, 24, 0, 2, 0, 1, 0, 20, 0, 21, 0, 3, 0, 1, 0, 1, 0,
        0, 0, 8, 0, 1, 0, 1, 0, 20,
    ];
    for (mark, expected) in [(20, vec![10, 20, 11]), (21, vec![99, 21])] {
        let mut buffer = glyphs(&[10, mark, 11]);
        apply_gsub_data_with_gdef(
            &gsub(vec![lookup.clone()]),
            &gdef,
            &mut buffer,
            &[GsubLookup::unmasked(0)],
            Tag::from_bytes(b"arab"),
        )
        .unwrap();
        assert_eq!(
            buffer.iter().map(|glyph| glyph.id.0).collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn mark_attachment_type_filters_other_mark_classes() {
    let mut lookup = lookup(4, ligature_subtable());
    lookup[2..4].copy_from_slice(&0x0100u16.to_be_bytes());
    let gdef = [
        0, 1, 0, 0, 0, 12, 0, 0, 0, 0, 0, 22, 0, 2, 0, 1, 0, 20, 0, 21, 0, 3, 0, 2, 0, 2, 0, 20, 0,
        20, 0, 1, 0, 21, 0, 21, 0, 2,
    ];
    for (mark, expected) in [(20, vec![10, 20, 11]), (21, vec![99, 21])] {
        let mut buffer = glyphs(&[10, mark, 11]);
        apply_gsub_data_with_gdef(
            &gsub(vec![lookup.clone()]),
            &gdef,
            &mut buffer,
            &[GsubLookup::unmasked(0)],
            Tag::from_bytes(b"arab"),
        )
        .unwrap();
        assert_eq!(
            buffer.iter().map(|glyph| glyph.id.0).collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn nested_and_extension_unknown_lookup_types_are_diagnostic() {
    let mut chain = Vec::from([0, 3, 0, 0, 0, 1, 0, 16, 0, 0, 0, 1, 0, 0, 0, 1]);
    chain.extend_from_slice(&coverage(10));
    let table = gsub(vec![lookup(6, chain), lookup(9, Vec::new())]);
    let mut buffer = glyphs(&[10]);
    let error = apply_gsub_data(
        &table,
        &mut buffer,
        &[GsubLookup::unmasked(0)],
        Tag::from_bytes(b"arab"),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::labels::shape::TextError::UnsupportedLookup { lookup_type: 9, .. }
    ));

    let extension = Vec::from([0, 1, 0, 9, 0, 0, 0, 8]);
    let mut buffer = glyphs(&[10]);
    let error = apply_gsub_data(
        &gsub(vec![lookup(7, extension)]),
        &mut buffer,
        &[GsubLookup::unmasked(0)],
        Tag::from_bytes(b"arab"),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::labels::shape::TextError::UnsupportedLookup { lookup_type: 9, .. }
    ));
}

#[test]
fn devanagari_reph_half_and_pref_masks_drive_gsub_and_reordering() {
    fn shaped(text: &str, ids: &[u16], feature: &[u8; 4], first: u16, second: u16) -> Vec<u16> {
        let mut ligature = Vec::from([0, 1, 0, 8, 0, 1, 0, 14]);
        ligature.extend_from_slice(&coverage(first));
        ligature.extend_from_slice(&[0, 1, 0, 4, 0, 90, 0, 2]);
        ligature.extend_from_slice(&second.to_be_bytes());
        let table = gsub(vec![lookup(4, ligature)]);
        let mut buffer: Vec<_> = text
            .char_indices()
            .zip(ids.iter().copied())
            .map(|((cluster, _), id)| Glyph::new(GlyphId(id), cluster as u32))
            .collect();
        crate::labels::shape::devanagari::apply_feature_masks(text, &mut buffer);
        apply_gsub_data(
            &table,
            &mut buffer,
            &[GsubLookup::for_feature(0, Tag::from_bytes(feature))],
            Tag::from_bytes(b"dev2"),
        )
        .unwrap();
        crate::labels::shape::devanagari::finish_reordering(text, &mut buffer);
        buffer.iter().map(|glyph| glyph.id.0).collect()
    }

    assert_eq!(shaped("र्क", &[10, 11, 12], b"rphf", 10, 11), vec![12, 90]);
    assert_eq!(shaped("क्त", &[10, 11, 12], b"half", 10, 11), vec![90, 12]);
    assert_eq!(shaped("क्र", &[10, 11, 12], b"pref", 11, 12), vec![90, 10]);
}

#[test]
fn shared_feature_lookup_executes_once_with_union_mask() {
    fn push(bytes: &mut Vec<u8>, value: u16) {
        bytes.extend_from_slice(&value.to_be_bytes());
    }
    fn patch(bytes: &mut [u8], at: usize, value: usize) {
        bytes[at..at + 2].copy_from_slice(&(value as u16).to_be_bytes());
    }

    let mut single = Vec::from([0, 1, 0, 6, 0, 1]);
    single.extend_from_slice(&[0, 2, 0, 1, 0, 10, 0, 11, 0, 0]);
    let lookup_list = gsub(vec![lookup(1, single)])[10..].to_vec();

    let mut table = vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
    let scripts = table.len();
    push(&mut table, 1);
    table.extend_from_slice(b"DFLT");
    let script_record = table.len();
    push(&mut table, 0);
    let script = table.len();
    let default_record = table.len();
    push(&mut table, 0);
    push(&mut table, 0);
    let language = table.len();
    for value in [0, u16::MAX, 2, 0, 1] {
        push(&mut table, value);
    }

    let features = table.len();
    push(&mut table, 2);
    table.extend_from_slice(b"init");
    let init_record = table.len();
    push(&mut table, 0);
    table.extend_from_slice(b"fina");
    let fina_record = table.len();
    push(&mut table, 0);
    let init = table.len();
    for value in [0, 1, 0] {
        push(&mut table, value);
    }
    let fina = table.len();
    for value in [0, 1, 0] {
        push(&mut table, value);
    }
    let lookups = table.len();
    table.extend_from_slice(&lookup_list);
    patch(&mut table, 4, scripts);
    patch(&mut table, 6, features);
    patch(&mut table, 8, lookups);
    patch(&mut table, script_record, script - scripts);
    patch(&mut table, default_record, language - script);
    patch(&mut table, init_record, init - features);
    patch(&mut table, fina_record, fina - features);

    let layout = crate::labels::shape::LayoutTable::parse(&table).unwrap();
    let selection = GsubLookup::selected(&layout, Tag::from_bytes(b"DFLT"), None, &[]).unwrap();
    assert_eq!(selection.len(), 1);
    let mut buffer = vec![Glyph::new(GlyphId(10), 0)];
    buffer[0].enable_feature(Tag::from_bytes(b"init"));
    buffer[0].enable_feature(Tag::from_bytes(b"fina"));
    apply_gsub_data(&table, &mut buffer, &selection, Tag::from_bytes(b"DFLT")).unwrap();
    assert_eq!(buffer[0].id.0, 11);
}
