use std::collections::HashMap;

#[cfg(feature = "extension-module")]
pub(super) struct VTSource {
    pub material_index: u32,
    pub virtual_size: (u32, u32),
    pub data: Vec<u8>,
    pub fallback_color: [f32; 4],
}

#[cfg(feature = "extension-module")]
pub(super) struct TerrainMaterialVT {
    pub sources: HashMap<(u32, String), VTSource>,  // (material_index, family)
    pub resident_tile_count: u32,
    pub total_tile_count: u32,
}

#[cfg(feature = "extension-module")]
impl TerrainMaterialVT {
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            resident_tile_count: 0,
            total_tile_count: 0,
        }
    }

    pub fn register_source(
        &mut self,
        material_index: u32,
        family: String,
        virtual_size_px: (u32, u32),
        data: Vec<u8>,
        fallback_color: [f32; 4],
    ) -> Result<(), String> {
        // Validation: check consistency if already registered for this family
        if let Some(existing) = self.sources.get(&(material_index, family.clone())) {
            if existing.virtual_size != virtual_size_px {
                return Err(format!(
                    "Virtual size mismatch: existing {:?}, new {:?}",
                    existing.virtual_size, virtual_size_px
                ));
            }
        }

        self.sources.insert(
            (material_index, family),
            VTSource {
                material_index,
                virtual_size: virtual_size_px,
                data,
                fallback_color,
            },
        );
        Ok(())
    }

    pub fn clear_sources(&mut self) {
        self.sources.clear();
    }

    pub fn get_stats(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();
        stats.insert("resident_pages".to_string(), self.resident_tile_count as f32);
        stats.insert("total_pages".to_string(), self.total_tile_count as f32);
        if self.total_tile_count > 0 {
            stats.insert(
                "miss_rate".to_string(),
                1.0 - (self.resident_tile_count as f32 / self.total_tile_count as f32),
            );
        }
        stats
    }
}

#[cfg(not(feature = "extension-module"))]
pub(super) struct TerrainMaterialVT;

#[cfg(not(feature = "extension-module"))]
impl TerrainMaterialVT {
    pub fn new() -> Self {
        Self
    }
}
