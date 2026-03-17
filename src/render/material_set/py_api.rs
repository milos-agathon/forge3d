use pyo3::prelude::*;

use super::core::resolve_default_texture;
use super::MaterialSet;

#[pymethods]
impl MaterialSet {
    /// Create default terrain material set
    ///
    /// Args:
    ///     triplanar_scale: Texture scaling for triplanar mapping (default: 6.0)
    ///     normal_strength: Normal map strength multiplier (default: 1.0)
    ///     blend_sharpness: Triplanar blend sharpness (default: 4.0)
    ///
    /// Returns:
    ///     MaterialSet with rock, grass, and snow materials
    #[staticmethod]
    #[pyo3(signature = (triplanar_scale=6.0, normal_strength=1.0, blend_sharpness=4.0))]
    pub fn terrain_default(
        triplanar_scale: f32,
        normal_strength: f32,
        blend_sharpness: f32,
    ) -> PyResult<Self> {
        validate_common_inputs(triplanar_scale, normal_strength, blend_sharpness)?;

        let mut materials = Vec::new();
        let mut texture_paths = Vec::new();

        materials.push(
            crate::core::material::PbrMaterial::dielectric(glam::Vec3::new(0.28, 0.26, 0.24), 0.50)
                .with_normal_scale(normal_strength * 1.5),
        );
        texture_paths.push(resolve_default_texture("rock_albedo.png"));

        materials.push(
            crate::core::material::PbrMaterial::dielectric(glam::Vec3::new(0.18, 0.38, 0.10), 0.85)
                .with_normal_scale(normal_strength * 0.8),
        );
        texture_paths.push(resolve_default_texture("grass_albedo.png"));

        materials.push(
            crate::core::material::PbrMaterial::dielectric(glam::Vec3::new(0.35, 0.25, 0.15), 0.50)
                .with_normal_scale(normal_strength * 1.2),
        );
        texture_paths.push(resolve_default_texture("dirt_albedo.png"));

        materials.push(
            crate::core::material::PbrMaterial::dielectric(glam::Vec3::new(0.95, 0.97, 1.0), 0.25)
                .with_normal_scale(normal_strength * 0.3),
        );
        texture_paths.push(resolve_default_texture("snow_albedo.png"));

        Ok(Self::from_parts(
            materials,
            triplanar_scale,
            normal_strength,
            blend_sharpness,
            texture_paths,
        ))
    }

    /// Create a material set with a single custom material
    #[staticmethod]
    #[pyo3(signature = (base_color, metallic, roughness, triplanar_scale=1.0, normal_strength=0.0, blend_sharpness=1.0))]
    pub fn custom(
        base_color: (f32, f32, f32),
        metallic: f32,
        roughness: f32,
        triplanar_scale: f32,
        normal_strength: f32,
        blend_sharpness: f32,
    ) -> PyResult<Self> {
        validate_common_inputs(triplanar_scale, normal_strength, blend_sharpness)?;
        if !(0.0..=1.0).contains(&metallic) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "metallic must be in [0, 1]",
            ));
        }
        if !(0.04..=1.0).contains(&roughness) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "roughness must be in [0.04, 1.0]",
            ));
        }

        let color = glam::Vec3::new(base_color.0, base_color.1, base_color.2);
        let material = if metallic > 0.5 {
            crate::core::material::PbrMaterial::metallic(color, roughness)
                .with_normal_scale(normal_strength)
        } else {
            crate::core::material::PbrMaterial::dielectric(color, roughness)
                .with_normal_scale(normal_strength)
        };

        Ok(Self::from_parts(
            vec![material],
            triplanar_scale,
            normal_strength,
            blend_sharpness,
            vec![None],
        ))
    }

    #[getter]
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    #[getter]
    pub fn triplanar_scale(&self) -> f32 {
        self.triplanar_scale
    }

    #[getter]
    pub fn normal_strength(&self) -> f32 {
        self.normal_strength
    }

    #[getter]
    pub fn blend_sharpness(&self) -> f32 {
        self.blend_sharpness
    }

    fn __repr__(&self) -> String {
        format!(
            "MaterialSet(materials={}, triplanar_scale={:.1}, normal_strength={:.1}, blend_sharpness={:.1})",
            self.materials.len(),
            self.triplanar_scale,
            self.normal_strength,
            self.blend_sharpness
        )
    }
}

fn validate_common_inputs(
    triplanar_scale: f32,
    normal_strength: f32,
    blend_sharpness: f32,
) -> PyResult<()> {
    if triplanar_scale <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "triplanar_scale must be > 0",
        ));
    }
    if normal_strength < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "normal_strength must be >= 0",
        ));
    }
    if blend_sharpness <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "blend_sharpness must be > 0",
        ));
    }
    Ok(())
}
