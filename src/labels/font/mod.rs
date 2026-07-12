mod face;
mod fvar;
pub(crate) mod outline;
mod variation;

pub use face::{FaceDescriptor, FaceMetrics, FontCollection, FontGlyph, FontRequest, TextError};
pub(crate) use fvar::parse_named_instances;

pub(crate) fn to_q26_6(value: i32, units_per_em: u16) -> i32 {
    let numerator = i64::from(value) * 64;
    let denominator = i64::from(units_per_em);
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    (quotient
        + if remainder.abs() * 2 >= denominator {
            numerator.signum()
        } else {
            0
        }) as i32
}

#[cfg(test)]
mod tests {
    use super::{parse_named_instances, to_q26_6, FontCollection, FontRequest, TextError};

    const DEMO_FONT_HEX: &str = "000100000007004000020030636d617000090076000001000000002c676c7966f1cb6698000001340000005c68656164f235ddf80000007c0000003668686561066100ca000000b400000024686d74780474006a000000f8000000086c6f6361002e00140000012c000000066d6178700005000b000000d8000000200001000000010000f59c29445f0f3cf5000203e800000000b492f40000000000dc2fa65c00060000025802bc000000030002000000000000000100000400fe70000002580006ffff0258000100000000000000000000000000000002000100000002000b00020000000000000000000000000000000000000000000002580064021c000600000001000000030000000c00040020000000040004000100000041ffff00000041ffffffc000010000000000000014002e0000000200640000025802bc00030007000033112111252111216401f4fe3401a4fe5c02bcfd4428026c000200060000021d02900002000a00001333030113331323272307adc463fef8da60dd593eef42010b0140fdb50290fd70c8c800";

    fn demo_font() -> Vec<u8> {
        DEMO_FONT_HEX
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                u8::from_str_radix(std::str::from_utf8(pair).expect("ASCII hex"), 16)
                    .expect("valid hex")
            })
            .collect()
    }

    fn with_tables(extra: &[([u8; 4], Vec<u8>)]) -> Vec<u8> {
        let source = demo_font();
        let old_count = u16::from_be_bytes(source[4..6].try_into().unwrap()) as usize;
        let mut tables = Vec::new();
        for index in 0..old_count {
            let record = 12 + index * 16;
            let tag = source[record..record + 4].try_into().unwrap();
            let offset =
                u32::from_be_bytes(source[record + 8..record + 12].try_into().unwrap()) as usize;
            let length =
                u32::from_be_bytes(source[record + 12..record + 16].try_into().unwrap()) as usize;
            tables.push((tag, source[offset..offset + length].to_vec()));
        }
        tables.extend(extra.iter().cloned());
        tables.sort_by_key(|(tag, _)| *tag);

        let count = tables.len();
        let max_power = 1usize << (usize::BITS - count.leading_zeros() - 1);
        let mut output = vec![0u8; 12 + count * 16];
        output[0..4].copy_from_slice(&0x0001_0000u32.to_be_bytes());
        output[4..6].copy_from_slice(&(count as u16).to_be_bytes());
        output[6..8].copy_from_slice(&((max_power * 16) as u16).to_be_bytes());
        output[8..10].copy_from_slice(&(max_power.trailing_zeros() as u16).to_be_bytes());
        output[10..12].copy_from_slice(&((count * 16 - max_power * 16) as u16).to_be_bytes());
        for (index, (tag, data)) in tables.into_iter().enumerate() {
            while !output.len().is_multiple_of(4) {
                output.push(0);
            }
            let offset = output.len();
            output.extend_from_slice(&data);
            let record = 12 + index * 16;
            output[record..record + 4].copy_from_slice(&tag);
            output[record + 8..record + 12].copy_from_slice(&(offset as u32).to_be_bytes());
            output[record + 12..record + 16].copy_from_slice(&(data.len() as u32).to_be_bytes());
        }
        output
    }

    fn variable_demo_font() -> Vec<u8> {
        let mut fvar = Vec::new();
        fvar.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        fvar.extend_from_slice(&16u16.to_be_bytes());
        fvar.extend_from_slice(&2u16.to_be_bytes());
        fvar.extend_from_slice(&1u16.to_be_bytes());
        fvar.extend_from_slice(&20u16.to_be_bytes());
        fvar.extend_from_slice(&1u16.to_be_bytes());
        fvar.extend_from_slice(&8u16.to_be_bytes());
        fvar.extend_from_slice(b"wght");
        for value in [100i32, 400, 900] {
            fvar.extend_from_slice(&(value << 16).to_be_bytes());
        }
        fvar.extend_from_slice(&0u16.to_be_bytes());
        fvar.extend_from_slice(&256u16.to_be_bytes());
        fvar.extend_from_slice(&300u16.to_be_bytes());
        fvar.extend_from_slice(&0u16.to_be_bytes());
        fvar.extend_from_slice(&(700i32 << 16).to_be_bytes());

        let utf16: Vec<u8> = "Bold".encode_utf16().flat_map(u16::to_be_bytes).collect();
        let mut name = Vec::new();
        for value in [0u16, 1, 18, 3, 1, 0x0409, 300, utf16.len() as u16, 0] {
            name.extend_from_slice(&value.to_be_bytes());
        }
        name.extend_from_slice(&utf16);
        with_tables(&[(*b"fvar", fvar), (*b"name", name)])
    }

    fn multi_axis_variable_demo_font() -> Vec<u8> {
        let mut fvar = Vec::new();
        fvar.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        fvar.extend_from_slice(&16u16.to_be_bytes());
        fvar.extend_from_slice(&2u16.to_be_bytes());
        fvar.extend_from_slice(&2u16.to_be_bytes());
        fvar.extend_from_slice(&20u16.to_be_bytes());
        fvar.extend_from_slice(&1u16.to_be_bytes());
        fvar.extend_from_slice(&12u16.to_be_bytes());
        for (tag, min, default, max, name_id) in [
            (*b"wght", 100i32, 400i32, 900i32, 256u16),
            (*b"wdth", 50i32, 100i32, 200i32, 257u16),
        ] {
            fvar.extend_from_slice(&tag);
            for value in [min, default, max] {
                fvar.extend_from_slice(&(value << 16).to_be_bytes());
            }
            fvar.extend_from_slice(&0u16.to_be_bytes());
            fvar.extend_from_slice(&name_id.to_be_bytes());
        }
        fvar.extend_from_slice(&300u16.to_be_bytes());
        fvar.extend_from_slice(&0u16.to_be_bytes());
        fvar.extend_from_slice(&(700i32 << 16).to_be_bytes());
        fvar.extend_from_slice(&(150i32 << 16).to_be_bytes());

        let mut avar = Vec::new();
        avar.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        avar.extend_from_slice(&0u16.to_be_bytes());
        avar.extend_from_slice(&2u16.to_be_bytes());
        for positive_end in [8192i16, 4096i16] {
            avar.extend_from_slice(&3u16.to_be_bytes());
            for (from, to) in [(-16384i16, -16384i16), (0, 0), (16384, positive_end)] {
                avar.extend_from_slice(&from.to_be_bytes());
                avar.extend_from_slice(&to.to_be_bytes());
            }
        }

        let utf16: Vec<u8> = "Bold Condensed"
            .encode_utf16()
            .flat_map(u16::to_be_bytes)
            .collect();
        let mut name = Vec::new();
        for value in [0u16, 1, 18, 3, 1, 0x0409, 300, utf16.len() as u16, 0] {
            name.extend_from_slice(&value.to_be_bytes());
        }
        name.extend_from_slice(&utf16);
        with_tables(&[(*b"avar", avar), (*b"fvar", fvar), (*b"name", name)])
    }

    fn table_offset(font: &[u8], wanted: [u8; 4]) -> usize {
        let count = u16::from_be_bytes(font[4..6].try_into().unwrap()) as usize;
        for index in 0..count {
            let record = 12 + index * 16;
            if font[record..record + 4] == wanted {
                return u32::from_be_bytes(font[record + 8..record + 12].try_into().unwrap())
                    as usize;
            }
        }
        panic!("missing table")
    }

    fn demo_collection() -> Vec<u8> {
        fn relocated(mut font: Vec<u8>, base: u32) -> Vec<u8> {
            let count = u16::from_be_bytes(font[4..6].try_into().unwrap()) as usize;
            for index in 0..count {
                let record = 12 + index * 16 + 8;
                let offset = u32::from_be_bytes(font[record..record + 4].try_into().unwrap());
                font[record..record + 4].copy_from_slice(&(offset + base).to_be_bytes());
            }
            font
        }

        let first_offset = 20u32;
        let second_offset = first_offset + demo_font().len() as u32;
        let first = relocated(demo_font(), first_offset);
        let second = relocated(demo_font(), second_offset);
        let mut ttc = Vec::new();
        ttc.extend_from_slice(b"ttcf");
        ttc.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        ttc.extend_from_slice(&2u32.to_be_bytes());
        ttc.extend_from_slice(&first_offset.to_be_bytes());
        ttc.extend_from_slice(&second_offset.to_be_bytes());
        ttc.extend_from_slice(&first);
        ttc.extend_from_slice(&second);
        ttc
    }

    #[test]
    fn q26_6_rounds_half_away_from_zero() {
        assert_eq!(to_q26_6(1, 128), 1);
        assert_eq!(to_q26_6(-1, 128), -1);
        assert_eq!(to_q26_6(1, 256), 0);
        assert_eq!(to_q26_6(-1, 256), 0);
        assert_eq!(to_q26_6(1000, 1000), 64);
    }

    #[test]
    fn collection_uses_ordered_cmap_fallback_and_exact_face() {
        let mut only_b = demo_font();
        only_b[283] = b'B';
        only_b[289] = b'B';
        let collection = FontCollection::load(&[
            FontRequest::from_bytes("only-b.ttf", only_b),
            FontRequest::from_bytes("only-a.ttf", demo_font()),
        ])
        .unwrap();

        let glyph = collection.glyph_for('A').unwrap();
        assert_eq!(glyph.font_index, 1);
        assert_eq!(glyph.glyph_id.0, 1);
        assert_eq!(collection.horizontal_advance(glyph).unwrap(), 540);
        assert_eq!(collection.horizontal_advance_q26_6(glyph).unwrap(), 35);
        assert_ne!(
            collection.descriptors()[0].sha256,
            collection.descriptors()[1].sha256
        );
        let exact = collection.descriptors()[1]
            .sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            exact,
            "cb40a3b0aed56dbd2465355ff5ac53ea5e6b567877132844d8f780fd600bdade"
        );
    }

    #[test]
    fn missing_glyph_names_codepoint_and_chain() {
        let collection = FontCollection::load(&[
            FontRequest::from_bytes("first.ttf", demo_font()),
            FontRequest::from_bytes("second.ttf", demo_font()),
        ])
        .unwrap();

        let message = collection.glyph_for('\u{10FFFF}').unwrap_err().to_string();
        assert!(message.contains("U+10FFFF"));
        assert!(message.contains("first.ttf"));
        assert!(message.contains("second.ttf"));
    }

    #[test]
    fn outline_and_metrics_come_from_selected_face() {
        let collection =
            FontCollection::load(&[FontRequest::from_bytes("demo.ttf", demo_font())]).unwrap();
        let glyph = collection.glyph_for('A').unwrap();
        let metrics = collection.metrics(glyph.font_index).unwrap();
        let path = collection
            .outline(glyph.font_index, glyph.glyph_id)
            .unwrap();

        assert_eq!(metrics.units_per_em, 1000);
        assert_eq!(metrics.ascender, 1024);
        assert_eq!(metrics.descender, -400);
        assert_eq!(collection.vertical_advance(glyph).unwrap(), None);
        assert_eq!(collection.glyph_class(glyph).unwrap(), None);
        assert_eq!(collection.mark_attachment_class(glyph).unwrap(), 0);
        assert!(path.iter().count() > 4);
    }

    #[test]
    fn missing_horizontal_advance_has_truthful_error() {
        let collection =
            FontCollection::load(&[FontRequest::from_bytes("demo.ttf", demo_font())]).unwrap();
        let err = collection
            .horizontal_advance(super::FontGlyph {
                font_index: 0,
                glyph_id: ttf_parser::GlyphId(500),
            })
            .unwrap_err();
        assert!(matches!(err, TextError::MissingHorizontalAdvance { .. }));
    }

    #[test]
    fn raw_fvar_named_instance_records_are_parsed() {
        let mut table = Vec::new();
        table.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        table.extend_from_slice(&16u16.to_be_bytes());
        table.extend_from_slice(&2u16.to_be_bytes());
        table.extend_from_slice(&1u16.to_be_bytes());
        table.extend_from_slice(&20u16.to_be_bytes());
        table.extend_from_slice(&1u16.to_be_bytes());
        table.extend_from_slice(&8u16.to_be_bytes());
        table.extend_from_slice(b"wght");
        table.extend_from_slice(&100i32.to_be_bytes());
        table.extend_from_slice(&400i32.to_be_bytes());
        table.extend_from_slice(&900i32.to_be_bytes());
        table.extend_from_slice(&0u16.to_be_bytes());
        table.extend_from_slice(&256u16.to_be_bytes());
        table.extend_from_slice(&300u16.to_be_bytes());
        table.extend_from_slice(&0u16.to_be_bytes());
        table.extend_from_slice(&700i32.to_be_bytes());

        let instances = parse_named_instances(&table).unwrap();
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].name_id, 300);
        assert_eq!(instances[0].coordinates, vec![(b"wght".to_owned(), 700)]);
    }

    #[test]
    fn named_instance_is_resolved_and_bound_to_descriptor() {
        let original = variable_demo_font();
        let collection =
            FontCollection::load(&[FontRequest::from_bytes("variable.ttf", original.clone())
                .with_named_instance("Bold")])
            .unwrap();
        assert_eq!(
            collection.descriptors()[0].variations,
            vec![(*b"wght", 700i32 << 16)]
        );
        assert_eq!(collection.glyph_for('A').unwrap().glyph_id.0, 1);
        assert!(collection
            .face(0)
            .unwrap()
            .has_non_default_variation_coordinates());
        assert_eq!(collection.font_bytes(0).unwrap(), original);
    }

    #[test]
    fn multi_axis_avar_is_applied_once_to_the_complete_vector() {
        let collection = FontCollection::load(&[FontRequest::from_bytes(
            "multi-axis.ttf",
            multi_axis_variable_demo_font(),
        )
        .with_named_instance("Bold Condensed")])
        .unwrap();
        let coordinates: Vec<i16> = collection
            .face(0)
            .unwrap()
            .variation_coordinates()
            .iter()
            .map(|coordinate| coordinate.get())
            .collect();
        assert_eq!(coordinates, vec![4915, 2048]);
    }

    #[test]
    fn fvar_rejects_truncated_declared_records() {
        let mut truncated_axis = vec![0, 1, 0, 0, 0, 16, 0, 2, 0, 1, 0, 24, 0, 0, 0, 4];
        truncated_axis.extend_from_slice(&[0; 20]);
        assert_eq!(
            parse_named_instances(&truncated_axis).unwrap_err(),
            TextError::MalformedFvar
        );

        let mut truncated_instance = vec![0, 1, 0, 0, 0, 16, 0, 2, 0, 1, 0, 20, 0, 1, 0, 12];
        truncated_instance.extend_from_slice(&[0; 20]);
        truncated_instance.extend_from_slice(&[0; 8]);
        assert_eq!(
            parse_named_instances(&truncated_instance).unwrap_err(),
            TextError::MalformedFvar
        );
    }

    #[test]
    fn load_rejects_malformed_avar_before_returning_collection() {
        let mut font = multi_axis_variable_demo_font();
        let avar = table_offset(&font, *b"avar");
        font[avar + 6..avar + 8].copy_from_slice(&1u16.to_be_bytes());
        let result = FontCollection::load(&[
            FontRequest::from_bytes("bad-avar.ttf", font).with_named_instance("Bold Condensed")
        ]);
        assert!(matches!(result, Err(TextError::MalformedFvar)));
    }

    #[test]
    fn nonzero_ttc_face_index_is_preserved() {
        let mut request = FontRequest::from_bytes("demo.ttc", demo_collection());
        request.face_index = 1;
        let collection = FontCollection::load(&[request]).unwrap();
        assert_eq!(collection.descriptors()[0].face_index, 1);
        assert_eq!(collection.glyph_for('A').unwrap().glyph_id.0, 1);
    }
}
