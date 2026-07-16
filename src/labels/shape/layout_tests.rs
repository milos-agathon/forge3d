use super::{FeatureSetting, LayoutTable};
use ttf_parser::Tag;

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn patch_u16(bytes: &mut [u8], offset: usize, value: usize) {
    bytes[offset..offset + 2].copy_from_slice(&(value as u16).to_be_bytes());
}

fn fixture() -> Vec<u8> {
    let mut bytes = vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
    let scripts = bytes.len();
    push_u16(&mut bytes, 2);
    bytes.extend_from_slice(b"DFLT");
    let dflt_record = bytes.len();
    push_u16(&mut bytes, 0);
    bytes.extend_from_slice(b"arab");
    let arab_record = bytes.len();
    push_u16(&mut bytes, 0);

    let dflt_script = bytes.len();
    for value in [4, 0, 0, 0, 1, 1] {
        push_u16(&mut bytes, value);
    }

    let arab_script = bytes.len();
    let arab_default = bytes.len();
    push_u16(&mut bytes, 0);
    push_u16(&mut bytes, 1);
    bytes.extend_from_slice(b"URD ");
    let urd_record = bytes.len();
    push_u16(&mut bytes, 0);
    let arab_default_table = bytes.len();
    for value in [0, 0, 2, 1, 2] {
        push_u16(&mut bytes, value);
    }
    let urd_table = bytes.len();
    for value in [0, 0, 2, 2, 4] {
        push_u16(&mut bytes, value);
    }
    patch_u16(&mut bytes, arab_default, arab_default_table - arab_script);
    patch_u16(&mut bytes, urd_record, urd_table - arab_script);

    let features = bytes.len();
    push_u16(&mut bytes, 5);
    let mut feature_records = Vec::new();
    for tag in [b"rlig", b"liga", b"kern", b"locl", b"locl"] {
        bytes.extend_from_slice(tag);
        feature_records.push(bytes.len());
        push_u16(&mut bytes, 0);
    }
    for (record, lookup) in feature_records.into_iter().zip([3, 7, 9, 11, 13]) {
        let table = bytes.len();
        for value in [0, 1, lookup] {
            push_u16(&mut bytes, value);
        }
        patch_u16(&mut bytes, record, table - features);
    }

    let lookups = bytes.len();
    push_u16(&mut bytes, 14);
    for _ in 0..14 {
        push_u16(&mut bytes, 0);
    }
    patch_u16(&mut bytes, 4, scripts);
    patch_u16(&mut bytes, 6, features);
    patch_u16(&mut bytes, 8, lookups);
    patch_u16(&mut bytes, dflt_record, dflt_script - scripts);
    patch_u16(&mut bytes, arab_record, arab_script - scripts);
    bytes
}

#[test]
fn required_feature_precedes_optional_features() {
    let bytes = fixture();
    let table = LayoutTable::parse(&bytes).unwrap();
    assert_eq!(
        table
            .selected_lookup_indices(Tag::from_bytes(b"arab"), None, &[])
            .unwrap(),
        vec![3, 7, 9]
    );
}

#[test]
fn script_and_language_fall_back_to_dflt() {
    let bytes = fixture();
    let table = LayoutTable::parse(&bytes).unwrap();
    assert_eq!(
        table
            .selected_lookup_indices(
                Tag::from_bytes(b"hebr"),
                Some(Tag::from_bytes(b"URD ")),
                &[],
            )
            .unwrap(),
        vec![3, 7]
    );
    assert_eq!(
        table
            .selected_lookup_indices(
                Tag::from_bytes(b"arab"),
                Some(Tag::from_bytes(b"dflt")),
                &[],
            )
            .unwrap(),
        vec![3, 7, 9]
    );
    assert_eq!(
        table
            .selected_lookup_indices(
                Tag::from_bytes(b"arab"),
                Some(Tag::from_bytes(b"FAR ")),
                &[],
            )
            .unwrap(),
        vec![3, 7, 9]
    );
}

#[test]
fn selected_language_uses_its_langsys_record() {
    let bytes = fixture();
    let table = LayoutTable::parse(&bytes).unwrap();
    assert_eq!(
        table
            .selected_lookup_indices(
                Tag::from_bytes(b"arab"),
                Some(Tag::from_bytes(b"URD ")),
                &[],
            )
            .unwrap(),
        vec![3, 9, 13]
    );
}

#[test]
fn language_does_not_append_first_global_locl_record() {
    let bytes = fixture();
    let table = LayoutTable::parse(&bytes).unwrap();
    let language = Some(Tag::from_bytes(b"URD "));
    let requested = [FeatureSetting::new(Tag::from_bytes(b"locl"), true)];
    let enabled =
        super::super::effective_features(&bytes, Tag::from_bytes(b"arab"), language, &requested)
            .unwrap();
    assert_eq!(
        table
            .selected_lookup_indices(Tag::from_bytes(b"arab"), language, &enabled)
            .unwrap(),
        vec![3, 9, 13, 7]
    );

    let requested = [FeatureSetting::new(Tag::from_bytes(b"locl"), false)];
    let disabled =
        super::super::effective_features(&bytes, Tag::from_bytes(b"arab"), language, &requested)
            .unwrap();
    assert_eq!(
        table
            .selected_lookup_indices(Tag::from_bytes(b"arab"), language, &disabled)
            .unwrap(),
        vec![3, 9, 7]
    );
}

#[test]
fn feature_overrides_disable_optional_but_not_required() {
    let bytes = fixture();
    let table = LayoutTable::parse(&bytes).unwrap();
    let settings = [
        FeatureSetting::new(Tag::from_bytes(b"rlig"), false),
        FeatureSetting::new(Tag::from_bytes(b"liga"), false),
    ];
    assert_eq!(
        table
            .selected_lookup_indices(Tag::from_bytes(b"arab"), None, &settings)
            .unwrap(),
        vec![3, 9]
    );
}

#[test]
fn explicit_enable_adds_known_feature() {
    let bytes = fixture();
    let table = LayoutTable::parse(&bytes).unwrap();
    let settings = [FeatureSetting::new(Tag::from_bytes(b"kern"), true)];
    assert_eq!(
        table
            .selected_lookup_indices(Tag::from_bytes(b"DFLT"), None, &settings)
            .unwrap(),
        vec![3, 7, 9]
    );
}

#[test]
fn malformed_layout_offsets_are_rejected() {
    let mut bytes = fixture();
    bytes[4..6].copy_from_slice(&u16::MAX.to_be_bytes());
    assert!(LayoutTable::parse(&bytes).is_err());
}

#[test]
fn version_one_one_requires_feature_variations_field_and_nonzero_lists() {
    let truncated = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0];
    assert!(LayoutTable::parse(&truncated).is_err());
}

#[test]
fn parse_rejects_truncated_script_records() {
    let mut bytes = fixture();
    let scripts = u16::from_be_bytes(bytes[4..6].try_into().unwrap()) as usize;
    bytes[scripts..scripts + 2].copy_from_slice(&u16::MAX.to_be_bytes());
    assert!(LayoutTable::parse(&bytes).is_err());
}
