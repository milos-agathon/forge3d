use crate::lighting::types::{BrdfModel, MaterialShading};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "MaterialShading")]
#[derive(Clone)]
pub struct PyMaterialShading {
    #[pyo3(get, set)]
    pub brdf: String,
    #[pyo3(get, set)]
    pub metallic: f32,
    #[pyo3(get, set)]
    pub roughness: f32,
    #[pyo3(get, set)]
    pub sheen: f32,
    #[pyo3(get, set)]
    pub clearcoat: f32,
    #[pyo3(get, set)]
    pub subsurface: f32,
    #[pyo3(get, set)]
    pub anisotropy: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyMaterialShading {
    #[new]
    #[pyo3(signature = (brdf="CookTorranceGgx", roughness=0.5, metallic=0.0, sheen=0.0, clearcoat=0.0, subsurface=0.0, anisotropy=0.0))]
    pub fn new(
        brdf: &str,
        roughness: f32,
        metallic: f32,
        sheen: f32,
        clearcoat: f32,
        subsurface: f32,
        anisotropy: f32,
    ) -> PyResult<Self> {
        let mat = Self {
            brdf: brdf.to_string(),
            metallic,
            roughness,
            sheen,
            clearcoat,
            subsurface,
            anisotropy,
        };
        mat.to_native()?;
        Ok(mat)
    }

    #[staticmethod]
    pub fn lambert(roughness: f32) -> PyResult<Self> {
        Self::new("Lambert", roughness, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    #[staticmethod]
    pub fn phong(roughness: f32, metallic: f32) -> PyResult<Self> {
        Self::new("Phong", roughness, metallic, 0.0, 0.0, 0.0, 0.0)
    }

    #[staticmethod]
    pub fn disney(
        roughness: f32,
        metallic: f32,
        sheen: f32,
        clearcoat: f32,
        subsurface: f32,
    ) -> PyResult<Self> {
        Self::new(
            "DisneyPrincipled",
            roughness,
            metallic,
            sheen,
            clearcoat,
            subsurface,
            0.0,
        )
    }

    #[staticmethod]
    pub fn anisotropic(brdf: &str, roughness: f32, anisotropy: f32) -> PyResult<Self> {
        Self::new(brdf, roughness, 0.0, 0.0, 0.0, 0.0, anisotropy)
    }
}

impl PyMaterialShading {
    pub fn to_native(&self) -> PyResult<MaterialShading> {
        let brdf_model = match self.brdf.as_str() {
            "Lambert" => BrdfModel::Lambert,
            "Phong" => BrdfModel::Phong,
            "BlinnPhong" => BrdfModel::BlinnPhong,
            "OrenNayar" => BrdfModel::OrenNayar,
            "CookTorranceGgx" => BrdfModel::CookTorranceGgx,
            "CookTorranceBeckmann" => BrdfModel::CookTorranceBeckmann,
            "DisneyPrincipled" => BrdfModel::DisneyPrincipled,
            "AshikhminShirley" => BrdfModel::AshikhminShirley,
            "Ward" => BrdfModel::Ward,
            "Toon" => BrdfModel::Toon,
            "Minnaert" => BrdfModel::Minnaert,
            "Subsurface" => BrdfModel::Subsurface,
            "Hair" => BrdfModel::Hair,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown BRDF model: {}",
                    self.brdf
                )))
            }
        };

        let mat = MaterialShading {
            brdf: brdf_model.as_u32(),
            metallic: self.metallic,
            roughness: self.roughness,
            sheen: self.sheen,
            clearcoat: self.clearcoat,
            subsurface: self.subsurface,
            anisotropy: self.anisotropy,
            _pad: 0.0,
        };

        mat.validate().map_err(PyValueError::new_err)?;
        Ok(mat)
    }
}
