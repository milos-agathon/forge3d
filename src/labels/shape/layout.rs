use super::ot::Reader;
use super::TextError;
use ttf_parser::Tag;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeatureSetting {
    pub tag: Tag,
    pub enabled: bool,
}

impl FeatureSetting {
    pub fn new(tag: Tag, enabled: bool) -> Self {
        Self { tag, enabled }
    }
}

pub struct LayoutTable<'a> {
    data: &'a [u8],
    script_list: usize,
    feature_list: usize,
    lookup_list: usize,
    feature_count: usize,
    lookup_count: usize,
}

impl<'a> LayoutTable<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, TextError> {
        let reader = Reader::new(data);
        let header_size = match reader.u32(0)? {
            0x0001_0000 => 10,
            0x0001_0001 => {
                reader.u32(10)?;
                14
            }
            _ => return Err(TextError::MalformedOpenType("layout header")),
        };
        let script_list = usize::from(reader.u16(4)?);
        let feature_list = usize::from(reader.u16(6)?);
        let lookup_list = usize::from(reader.u16(8)?);
        if [script_list, feature_list, lookup_list]
            .iter()
            .any(|offset| *offset < header_size)
        {
            return Err(TextError::MalformedOpenType("layout header"));
        }
        let feature_count = usize::from(reader.u16(feature_list)?);
        let lookup_count = usize::from(reader.u16(lookup_list)?);
        let script_count = usize::from(reader.u16(script_list)?);
        reader.slice_at(script_list + 2, script_count * 6)?;
        reader.slice_at(feature_list + 2, feature_count * 6)?;
        reader.slice_at(lookup_list + 2, lookup_count * 2)?;
        Ok(Self {
            data,
            script_list,
            feature_list,
            lookup_list,
            feature_count,
            lookup_count,
        })
    }

    fn tag_at(&self, offset: usize) -> Result<Tag, TextError> {
        let bytes: [u8; 4] = Reader::new(self.data)
            .slice_at(offset, 4)?
            .try_into()
            .map_err(|_| TextError::MalformedOpenType("tag"))?;
        Ok(Tag::from_bytes(&bytes))
    }

    fn tagged_offset(
        &self,
        list: usize,
        count: usize,
        tag: Tag,
    ) -> Result<Option<usize>, TextError> {
        let reader = Reader::new(self.data);
        for index in 0..count {
            let record = list + 2 + index * 6;
            if self.tag_at(record)? == tag {
                return Ok(Some(list + usize::from(reader.u16(record + 4)?)));
            }
        }
        Ok(None)
    }

    fn script_offset(&self, script: Tag) -> Result<usize, TextError> {
        let reader = Reader::new(self.data);
        let count = usize::from(reader.u16(self.script_list)?);
        self.tagged_offset(self.script_list, count, script)?
            .or(self.tagged_offset(self.script_list, count, Tag::from_bytes(b"DFLT"))?)
            .ok_or(TextError::MalformedOpenType("ScriptList"))
    }

    fn lang_sys_offset(&self, script: usize, language: Option<Tag>) -> Result<usize, TextError> {
        let reader = Reader::new(self.data);
        let default_offset = usize::from(reader.u16(script)?);
        let count = usize::from(reader.u16(script + 2)?);
        if let Some(language) = language {
            for index in 0..count {
                let record = script + 4 + index * 6;
                if self.tag_at(record)? == language {
                    return Ok(script + usize::from(reader.u16(record + 4)?));
                }
            }
        }
        if default_offset == 0 {
            return Err(TextError::MalformedOpenType("LangSys"));
        }
        Ok(script + default_offset)
    }

    fn feature_tag(&self, index: usize) -> Result<Tag, TextError> {
        if index >= self.feature_count {
            return Err(TextError::MalformedOpenType("FeatureList"));
        }
        self.tag_at(self.feature_list + 2 + index * 6)
    }

    fn feature_index(&self, tag: Tag) -> Result<Option<usize>, TextError> {
        for index in 0..self.feature_count {
            if self.feature_tag(index)? == tag {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }

    fn feature_lookups(&self, index: usize) -> Result<Vec<u16>, TextError> {
        let reader = Reader::new(self.data);
        if index >= self.feature_count {
            return Err(TextError::MalformedOpenType("FeatureList"));
        }
        let record = self.feature_list + 2 + index * 6;
        let table = self.feature_list + usize::from(reader.u16(record + 4)?);
        let count = usize::from(reader.u16(table + 2)?);
        let mut lookups = Vec::with_capacity(count);
        for position in 0..count {
            let lookup = reader.u16(table + 4 + position * 2)?;
            if usize::from(lookup) >= self.lookup_count {
                return Err(TextError::MalformedOpenType("LookupList"));
            }
            lookups.push(lookup);
        }
        Ok(lookups)
    }

    pub fn selected_lookup_indices(
        &self,
        script: Tag,
        language: Option<Tag>,
        settings: &[FeatureSetting],
    ) -> Result<Vec<u16>, TextError> {
        let mut lookups = Vec::new();
        for (lookup, _) in self.selected_feature_lookups(script, language, settings)? {
            if !lookups.contains(&lookup) {
                lookups.push(lookup);
            }
        }
        Ok(lookups)
    }

    pub fn selected_feature_lookups(
        &self,
        script: Tag,
        language: Option<Tag>,
        settings: &[FeatureSetting],
    ) -> Result<Vec<(u16, Vec<Tag>)>, TextError> {
        let reader = Reader::new(self.data);
        let script = self.script_offset(script)?;
        let lang_sys = self.lang_sys_offset(script, language)?;
        reader.u16(lang_sys)?;
        let required = reader.u16(lang_sys + 2)?;
        let count = usize::from(reader.u16(lang_sys + 4)?);
        reader.slice_at(lang_sys + 6, count * 2)?;

        let disabled = |tag| settings.iter().any(|item| item.tag == tag && !item.enabled);
        let mut features = Vec::new();
        if required != u16::MAX {
            features.push(usize::from(required));
        }
        for position in 0..count {
            let index = usize::from(reader.u16(lang_sys + 6 + position * 2)?);
            if !disabled(self.feature_tag(index)?) && !features.contains(&index) {
                features.push(index);
            }
        }
        for setting in settings.iter().filter(|setting| setting.enabled) {
            if let Some(index) = self.feature_index(setting.tag)? {
                if !features.contains(&index) {
                    features.push(index);
                }
            }
        }

        let mut lookups = Vec::new();
        for feature in features {
            let tag = self.feature_tag(feature)?;
            for lookup in self.feature_lookups(feature)? {
                if let Some((_, tags)) = lookups
                    .iter_mut()
                    .find(|(existing, _): &&mut (u16, Vec<Tag>)| *existing == lookup)
                {
                    if !tags.contains(&tag) {
                        tags.push(tag);
                    }
                } else {
                    lookups.push((lookup, vec![tag]));
                }
            }
        }
        Ok(lookups)
    }

    pub fn lookup_count(&self) -> usize {
        self.lookup_count
    }

    pub fn lookup_list_offset(&self) -> usize {
        self.lookup_list
    }
}

#[cfg(test)]
#[path = "layout_tests.rs"]
mod tests;
