// src/geometry/array_convert.rs
// Python array conversion helpers for geometry module
// RELEVANT FILES: python/forge3d/geometry.py

use numpy::PyReadonlyArray2;

/// Convert fixed-size arrays to nested Vecs for PyArray2 construction.

pub fn to_vec2(data: &[[f32; 2]]) -> Vec<Vec<f32>> {
    data.iter().map(|row| row.to_vec()).collect()
}

pub fn to_vec3(data: &[[f32; 3]]) -> Vec<Vec<f32>> {
    data.iter().map(|row| row.to_vec()).collect()
}

pub fn to_vec4(data: &[[f32; 4]]) -> Vec<Vec<f32>> {
    data.iter().map(|row| row.to_vec()).collect()
}

pub fn read_vec2(array: PyReadonlyArray2<'_, f32>) -> Vec<[f32; 2]> {
    array
        .as_array()
        .outer_iter()
        .map(|row| [row[0], row[1]])
        .collect()
}

pub fn read_vec3(array: PyReadonlyArray2<'_, f32>) -> Vec<[f32; 3]> {
    array
        .as_array()
        .outer_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect()
}

pub fn read_vec4(array: PyReadonlyArray2<'_, f32>) -> Vec<[f32; 4]> {
    array
        .as_array()
        .outer_iter()
        .map(|row| [row[0], row[1], row[2], row[3]])
        .collect()
}
