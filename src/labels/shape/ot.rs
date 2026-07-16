use super::TextError;
use ttf_parser::GlyphId;

#[derive(Clone, Copy)]
pub struct Reader<'a> {
    data: &'a [u8],
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    pub fn slice_at(&self, offset: usize, length: usize) -> Result<&'a [u8], TextError> {
        let end = offset
            .checked_add(length)
            .ok_or(TextError::OutOfBounds { offset, length })?;
        self.data
            .get(offset..end)
            .ok_or(TextError::OutOfBounds { offset, length })
    }

    pub fn u16(&self, offset: usize) -> Result<u16, TextError> {
        Ok(u16::from_be_bytes(
            self.slice_at(offset, 2)?
                .try_into()
                .map_err(|_| TextError::OutOfBounds { offset, length: 2 })?,
        ))
    }

    pub fn i16(&self, offset: usize) -> Result<i16, TextError> {
        Ok(i16::from_be_bytes(
            self.slice_at(offset, 2)?
                .try_into()
                .map_err(|_| TextError::OutOfBounds { offset, length: 2 })?,
        ))
    }

    pub fn u32(&self, offset: usize) -> Result<u32, TextError> {
        Ok(u32::from_be_bytes(
            self.slice_at(offset, 4)?
                .try_into()
                .map_err(|_| TextError::OutOfBounds { offset, length: 4 })?,
        ))
    }
}

#[derive(Clone, Copy)]
pub struct Coverage<'a> {
    data: &'a [u8],
    format: u16,
    count: usize,
}

impl<'a> Coverage<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, TextError> {
        let reader = Reader::new(data);
        let format = reader.u16(0)?;
        let count = usize::from(reader.u16(2)?);
        let record_size = match format {
            1 => 2,
            2 => 6,
            _ => return Err(TextError::MalformedOpenType("coverage")),
        };
        reader.slice_at(4, count * record_size)?;
        let coverage = Self {
            data,
            format,
            count,
        };
        coverage.validate()?;
        Ok(coverage)
    }

    fn validate(&self) -> Result<(), TextError> {
        let reader = Reader::new(self.data);
        let mut previous_end = None;
        let mut expected_coverage_index = 0u32;
        for index in 0..self.count {
            let offset = 4 + index * if self.format == 1 { 2 } else { 6 };
            let start = reader.u16(offset)?;
            let end = if self.format == 1 {
                start
            } else {
                reader.u16(offset + 2)?
            };
            if start > end || previous_end.is_some_and(|previous| start <= previous) {
                return Err(TextError::MalformedOpenType("coverage"));
            }
            if self.format == 2 {
                if u32::from(reader.u16(offset + 4)?) != expected_coverage_index {
                    return Err(TextError::MalformedOpenType("coverage"));
                }
                expected_coverage_index += u32::from(end - start) + 1;
                if expected_coverage_index > u32::from(u16::MAX) + 1 {
                    return Err(TextError::MalformedOpenType("coverage"));
                }
            }
            previous_end = Some(end);
        }
        Ok(())
    }

    pub fn index(&self, glyph: GlyphId) -> Option<u16> {
        let reader = Reader::new(self.data);
        if self.format == 1 {
            (0..self.count).find_map(|index| {
                (reader.u16(4 + index * 2).ok()? == glyph.0).then_some(index as u16)
            })
        } else {
            (0..self.count).find_map(|index| {
                let offset = 4 + index * 6;
                let start = reader.u16(offset).ok()?;
                let end = reader.u16(offset + 2).ok()?;
                let base = reader.u16(offset + 4).ok()?;
                (start <= glyph.0 && glyph.0 <= end)
                    .then(|| base.checked_add(glyph.0 - start))
                    .flatten()
            })
        }
    }
}

#[derive(Clone, Copy)]
pub struct ClassDef<'a> {
    data: &'a [u8],
    format: u16,
    count: usize,
}

impl<'a> ClassDef<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, TextError> {
        let reader = Reader::new(data);
        let format = reader.u16(0)?;
        let (count_offset, records_offset, record_size) = match format {
            1 => (4, 6, 2),
            2 => (2, 4, 6),
            _ => return Err(TextError::MalformedOpenType("class definition")),
        };
        let count = usize::from(reader.u16(count_offset)?);
        reader.slice_at(records_offset, count * record_size)?;
        let classes = Self {
            data,
            format,
            count,
        };
        classes.validate()?;
        Ok(classes)
    }

    fn validate(&self) -> Result<(), TextError> {
        if self.format == 1 {
            let start = u32::from(Reader::new(self.data).u16(2)?);
            if start + self.count as u32 > u32::from(u16::MAX) + 1 {
                return Err(TextError::MalformedOpenType("ClassDef glyph range"));
            }
            return Ok(());
        }
        let reader = Reader::new(self.data);
        let mut previous_end = None;
        for index in 0..self.count {
            let offset = 4 + index * 6;
            let start = reader.u16(offset)?;
            let end = reader.u16(offset + 2)?;
            if start > end || previous_end.is_some_and(|previous| start <= previous) {
                return Err(TextError::MalformedOpenType("class definition"));
            }
            previous_end = Some(end);
        }
        Ok(())
    }

    pub fn get(&self, glyph: GlyphId) -> u16 {
        let reader = Reader::new(self.data);
        if self.format == 1 {
            let start = reader.u16(2).unwrap_or(0);
            let Some(index) = glyph.0.checked_sub(start).map(usize::from) else {
                return 0;
            };
            if index >= self.count {
                return 0;
            }
            reader.u16(6 + index * 2).unwrap_or(0)
        } else {
            (0..self.count)
                .find_map(|index| {
                    let offset = 4 + index * 6;
                    let start = reader.u16(offset).ok()?;
                    let end = reader.u16(offset + 2).ok()?;
                    (start <= glyph.0 && glyph.0 <= end)
                        .then(|| reader.u16(offset + 4).ok())
                        .flatten()
                })
                .unwrap_or(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ClassDef, Coverage, Reader};
    use ttf_parser::GlyphId;

    #[test]
    fn reader_rejects_out_of_bounds_offsets() {
        assert!(Reader::new(&[0, 1]).slice_at(4, 2).is_err());
        assert!(Reader::new(&[0, 1]).u32(0).is_err());
    }

    #[test]
    fn coverage_format_one_returns_sorted_index() {
        let bytes = [0, 1, 0, 3, 0, 2, 0, 5, 0, 9];
        let coverage = Coverage::parse(&bytes).unwrap();
        assert_eq!(coverage.index(GlyphId(2)), Some(0));
        assert_eq!(coverage.index(GlyphId(9)), Some(2));
        assert_eq!(coverage.index(GlyphId(6)), None);
    }

    #[test]
    fn coverage_format_two_uses_range_start_index() {
        let bytes = [0, 2, 0, 2, 0, 3, 0, 5, 0, 0, 0, 10, 0, 11, 0, 3];
        let coverage = Coverage::parse(&bytes).unwrap();
        assert_eq!(coverage.index(GlyphId(4)), Some(1));
        assert_eq!(coverage.index(GlyphId(11)), Some(4));
        assert_eq!(coverage.index(GlyphId(9)), None);
    }

    #[test]
    fn class_definitions_support_both_formats() {
        let format_one = [0, 1, 0, 5, 0, 3, 0, 1, 0, 2, 0, 0];
        let classes = ClassDef::parse(&format_one).unwrap();
        assert_eq!(classes.get(GlyphId(5)), 1);
        assert_eq!(classes.get(GlyphId(6)), 2);
        assert_eq!(classes.get(GlyphId(9)), 0);

        let format_two = [0, 2, 0, 2, 0, 2, 0, 4, 0, 7, 0, 8, 0, 8, 0, 3];
        let classes = ClassDef::parse(&format_two).unwrap();
        assert_eq!(classes.get(GlyphId(3)), 7);
        assert_eq!(classes.get(GlyphId(8)), 3);
        assert_eq!(classes.get(GlyphId(7)), 0);
    }

    #[test]
    fn malformed_counts_are_rejected() {
        assert!(Coverage::parse(&[0, 1, 0, 2, 0, 4]).is_err());
        assert!(ClassDef::parse(&[0, 2, 0, 1, 0, 4]).is_err());
    }

    #[test]
    fn coverage_range_indices_must_be_contiguous() {
        let bytes = [0, 2, 0, 2, 0, 3, 0, 5, 0, 0, 0, 10, 0, 11, 0, 9];
        assert!(Coverage::parse(&bytes).is_err());
    }

    #[test]
    fn class_definition_format_one_rejects_glyph_range_overflow() {
        let bytes = [0, 1, 0xFF, 0xFF, 0, 2, 0, 1, 0, 2];
        assert!(ClassDef::parse(&bytes).is_err());
    }
}
