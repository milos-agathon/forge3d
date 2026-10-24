use pyo3::prelude::*;
use std::path::Path;

#[cfg(feature = "extension-module")]
use once_cell::sync::OnceCell;
#[cfg(feature = "extension-module")]
use std::sync::Arc;

/// Material set for terrain rendering
#[pyclass(module = "forge3d._forge3d", name = "MaterialSet")]
pub struct MaterialSet {
    pub(crate) materials: Vec<crate::core::material::PbrMaterial>,
    pub(crate) triplanar_scale: f32,
    pub(crate) normal_strength: f32,
    pub(crate) blend_sharpness: f32,
    pub(crate) _texture_paths: Vec<Option<String>>,
    #[cfg(feature = "extension-module")]
    pub(crate) gpu_cache: OnceCell<Arc<super::GpuMaterialSet>>,
    #[cfg(feature = "extension-module")]
    pub(crate) physical_gpu_cache: OnceCell<Arc<super::GpuMaterialSet>>,
}

impl Clone for MaterialSet {
    fn clone(&self) -> Self {
        Self {
            materials: self.materials.clone(),
            triplanar_scale: self.triplanar_scale,
            normal_strength: self.normal_strength,
            blend_sharpness: self.blend_sharpness,
            _texture_paths: self._texture_paths.clone(),
            #[cfg(feature = "extension-module")]
            gpu_cache: OnceCell::new(),
            #[cfg(feature = "extension-module")]
            physical_gpu_cache: OnceCell::new(),
        }
    }
}

impl MaterialSet {
    pub(crate) fn from_parts(
        materials: Vec<crate::core::material::PbrMaterial>,
        triplanar_scale: f32,
        normal_strength: f32,
        blend_sharpness: f32,
        texture_paths: Vec<Option<String>>,
    ) -> Self {
        Self {
            materials,
            triplanar_scale,
            normal_strength,
            blend_sharpness,
            _texture_paths: texture_paths,
            #[cfg(feature = "extension-module")]
            gpu_cache: OnceCell::new(),
            #[cfg(feature = "extension-module")]
            physical_gpu_cache: OnceCell::new(),
        }
    }

    /// Get reference to materials
    pub fn materials(&self) -> &[crate::core::material::PbrMaterial] {
        &self.materials
    }

    /// Get material at index
    pub fn get_material(&self, index: usize) -> Option<&crate::core::material::PbrMaterial> {
        self.materials.get(index)
    }

    /// Return the authored linear base color when this set is exactly one
    /// constant material. Physical terrain can then use the original f32
    /// reflectance without an sRGB8 texture round trip.
    pub(crate) fn single_untextured_base_color(&self) -> Option<[f32; 3]> {
        if self.materials.len() != 1
            || self._texture_paths.len() != 1
            || self._texture_paths[0].is_some()
        {
            return None;
        }

        let base_color = self.materials[0].base_color;
        Some([base_color[0], base_color[1], base_color[2]])
    }

    /// Identity of the immutable resources actually uploaded for this material set.
    #[cfg(feature = "extension-module")]
    pub(crate) fn anamnesis_bytes(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> std::io::Result<Vec<u8>> {
        let gpu = self
            .gpu(device, queue)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        let mut bytes = b"forge3d.terrain.material-set/v2\0".to_vec();
        bytes.extend_from_slice(bytemuck::cast_slice(&self.materials));
        bytes.extend_from_slice(&self.triplanar_scale.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.normal_strength.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.blend_sharpness.to_bits().to_le_bytes());
        bytes.extend_from_slice(&gpu.resident_identity);
        Ok(bytes)
    }
}

pub(super) fn resolve_default_texture(file_name: &str) -> Option<String> {
    let asset_root = std::env::var("FORGE3D_MATERIAL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("assets/materials"));
    let candidate = asset_root.join(file_name);
    if Path::new(&candidate).exists() {
        Some(candidate.to_string_lossy().to_string())
    } else {
        None
    }
}
