use super::*;

#[test]
fn invalid_context_sequence_index_is_malformed() {
    let mut chain = Vec::from([0, 3, 0, 0, 0, 1, 0, 16, 0, 0, 0, 1, 0, 1, 0, 1]);
    chain.extend_from_slice(&coverage(10));
    let mut buffer = glyphs(&[10]);
    let error = apply_gsub_data(
        &gsub(vec![lookup(6, chain), lookup(1, Vec::new())]),
        &mut buffer,
        &[GsubLookup::unmasked(0)],
        Tag::from_bytes(b"latn"),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::labels::shape::TextError::MalformedOpenType("GSUB contextual sequence index")
    ));
}

#[test]
fn extension_target_must_be_in_bounds() {
    let extension = Vec::from([0, 1, 0, 1, 0x7F, 0xFF, 0xFF, 0xFF]);
    let mut buffer = glyphs(&[10]);
    assert!(apply_gsub_data(
        &gsub(vec![lookup(7, extension)]),
        &mut buffer,
        &[GsubLookup::unmasked(0)],
        Tag::from_bytes(b"latn"),
    )
    .is_err());
}

#[test]
fn contextual_remap_does_not_retarget_consumed_glyph_to_ignored_mark() {
    let mut chain = Vec::from([
        0, 3, 0, 0, 0, 2, 0, 22, 0, 28, 0, 0, 0, 2, 0, 0, 0, 1, 0, 1, 0, 2,
    ]);
    chain.extend_from_slice(&coverage(10));
    chain.extend_from_slice(&coverage(11));
    let mut contextual = lookup(6, chain);
    contextual[2..4].copy_from_slice(&8u16.to_be_bytes());
    let mut ligature = Vec::from([0, 1, 0, 8, 0, 1, 0, 14]);
    ligature.extend_from_slice(&coverage(10));
    ligature.extend_from_slice(&[0, 1, 0, 4, 0, 99, 0, 2, 0, 11]);
    let mut ligature_lookup = lookup(4, ligature);
    ligature_lookup[2..4].copy_from_slice(&8u16.to_be_bytes());
    let mut mark_substitution = Vec::from([0, 2, 0, 8, 0, 1, 0, 77]);
    mark_substitution.extend_from_slice(&coverage(20));
    let gdef = [
        0, 1, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 20, 0, 20, 0, 3,
    ];
    let mut buffer = glyphs(&[10, 20, 11]);
    apply_gsub_data_with_gdef(
        &gsub(vec![
            contextual,
            ligature_lookup,
            lookup(1, mark_substitution),
        ]),
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
}

#[test]
fn multiple_substitution_outputs_have_distinct_context_identities() {
    let mut multiple = Vec::from([0, 1, 0, 8, 0, 1, 0, 14]);
    multiple.extend_from_slice(&coverage(10));
    multiple.extend_from_slice(&[0, 2, 0, 20, 0, 21]);
    let mut chain = Vec::from([0, 3, 0, 0, 0, 1, 0, 20, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3]);
    chain.extend_from_slice(&coverage(21));
    let mut first = Vec::from([0, 2, 0, 8, 0, 1, 0, 22]);
    first.extend_from_slice(&coverage(21));
    let mut second = Vec::from([0, 2, 0, 8, 0, 1, 0, 23]);
    second.extend_from_slice(&coverage(22));
    let table = gsub(vec![
        lookup(2, multiple),
        lookup(6, chain),
        lookup(1, first),
        lookup(1, second),
    ]);
    let mut buffer = glyphs(&[10]);
    apply_gsub_data(
        &table,
        &mut buffer,
        &[GsubLookup::unmasked(0), GsubLookup::unmasked(1)],
        Tag::from_bytes(b"latn"),
    )
    .unwrap();
    assert_eq!(
        buffer.iter().map(|glyph| glyph.id.0).collect::<Vec<_>>(),
        vec![20, 23]
    );
}

#[test]
fn reused_buffer_assigns_appended_glyph_a_fresh_identity() {
    let mut deletion = Vec::from([0, 1, 0, 8, 0, 1, 0, 14]);
    deletion.extend_from_slice(&coverage(10));
    deletion.extend_from_slice(&[0, 0]);
    let mut chain = Vec::from([
        0, 3, 0, 0, 0, 2, 0, 22, 0, 28, 0, 0, 0, 2, 0, 1, 0, 2, 0, 1, 0, 3,
    ]);
    chain.extend_from_slice(&coverage(11));
    chain.extend_from_slice(&coverage(12));
    let mut first = Vec::from([0, 2, 0, 8, 0, 1, 0, 13]);
    first.extend_from_slice(&coverage(12));
    let mut second = Vec::from([0, 2, 0, 8, 0, 1, 0, 14]);
    second.extend_from_slice(&coverage(13));
    let table = gsub(vec![
        lookup(2, deletion),
        lookup(6, chain),
        lookup(1, first),
        lookup(1, second),
    ]);
    let mut buffer = glyphs(&[10, 11]);
    apply_gsub_data(
        &table,
        &mut buffer,
        &[GsubLookup::unmasked(0)],
        Tag::from_bytes(b"latn"),
    )
    .unwrap();
    buffer.push(Glyph::new(GlyphId(12), 2));
    apply_gsub_data(
        &table,
        &mut buffer,
        &[GsubLookup::unmasked(1)],
        Tag::from_bytes(b"latn"),
    )
    .unwrap();
    assert_eq!(
        buffer.iter().map(|glyph| glyph.id.0).collect::<Vec<_>>(),
        vec![11, 14]
    );
}
