from __future__ import annotations

import base64
import binascii
import json
import struct
import zlib

import numpy as np
import pytest

import forge3d as f3d
import forge3d.map_scene as map_scene
from forge3d import io


def _png_data_uri_rgba_1x1() -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\xff\xff\xff")
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _write_two_primitive_gltf(path) -> None:
    positions = struct.pack(
        "<18f",
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        2.0, 0.0, 0.0,
        3.0, 0.0, 0.0,
        2.0, 1.0, 0.0,
    )
    indices = struct.pack("<6H", 0, 1, 2, 0, 1, 2)
    blob = positions + indices
    uri = "data:application/octet-stream;base64," + base64.b64encode(blob).decode("ascii")
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(blob), "uri": uri}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices), "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 0.0],
            },
            {
                "bufferView": 0,
                "byteOffset": 36,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [2.0, 0.0, 0.0],
                "max": [3.0, 1.0, 0.0],
            },
            {"bufferView": 1, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 1, "byteOffset": 6, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 2},
                    {"attributes": {"POSITION": 1}, "indices": 3},
                ]
            }
        ],
    }
    path.write_text(json.dumps(gltf), encoding="utf-8")


def _write_material_gltf(path) -> None:
    positions = struct.pack(
        "<18f",
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        2.0, 0.0, 0.0,
        3.0, 0.0, 0.0,
        2.0, 1.0, 0.0,
    )
    indices = struct.pack("<6H", 0, 1, 2, 0, 1, 2)
    blob = positions + indices
    uri = "data:application/octet-stream;base64," + base64.b64encode(blob).decode("ascii")
    png_1x1 = _png_data_uri_rgba_1x1()
    gltf = {
        "asset": {"version": "2.0"},
        "extensionsUsed": ["KHR_materials_unlit", "KHR_texture_transform"],
        "buffers": [{"byteLength": len(blob), "uri": uri}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices), "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 0.0],
            },
            {
                "bufferView": 0,
                "byteOffset": 36,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [2.0, 0.0, 0.0],
                "max": [3.0, 1.0, 0.0],
            },
            {"bufferView": 1, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 1, "byteOffset": 6, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "images": [{"uri": png_1x1}],
        "textures": [{"source": 0}],
        "materials": [
            {
                "name": "mat_red",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 0.0, 0.0, 0.5],
                    "metallicFactor": 0.25,
                    "roughnessFactor": 0.75,
                    "baseColorTexture": {
                        "index": 0,
                        "texCoord": 1,
                        "extensions": {
                            "KHR_texture_transform": {
                                "offset": [0.25, 0.5],
                                "rotation": 0.125,
                                "scale": [2.0, 3.0],
                                "texCoord": 0,
                            }
                        },
                    },
                    "metallicRoughnessTexture": {"index": 0},
                },
                "normalTexture": {"index": 0, "scale": 0.8},
                "occlusionTexture": {"index": 0, "strength": 0.6},
                "emissiveTexture": {"index": 0},
                "alphaMode": "BLEND",
                "doubleSided": True,
                "emissiveFactor": [0.1, 0.2, 0.3],
                "extensions": {"KHR_materials_unlit": {}},
            },
            {
                "name": "mat_blue",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.0, 0.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.5,
                },
                "alphaMode": "MASK",
                "alphaCutoff": 0.4,
            },
        ],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 2, "material": 0},
                    {"attributes": {"POSITION": 1}, "indices": 3, "material": 1},
                ]
            }
        ],
    }
    path.write_text(json.dumps(gltf), encoding="utf-8")


def _write_material_glb(path) -> None:
    json_path = path.with_suffix(".gltf")
    _write_material_gltf(json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    json_path.unlink()
    json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - (len(json_bytes) % 4)) % 4)
    chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    header = struct.pack("<III", 0x46546C67, 2, 12 + len(chunk))
    path.write_bytes(header + chunk)


def _write_translated_node_gltf(path) -> None:
    positions = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    uri = "data:application/octet-stream;base64," + base64.b64encode(positions).decode("ascii")
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "translation": [10.0, 20.0, 30.0]}],
        "buffers": [{"byteLength": len(positions), "uri": uri}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 0.0],
            }
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
    }
    path.write_text(json.dumps(gltf), encoding="utf-8")


def test_import_gltf_concatenates_all_mesh_primitives(tmp_path) -> None:
    path = tmp_path / "two_primitives.gltf"
    _write_two_primitive_gltf(path)

    mesh = io.import_gltf(str(path))

    assert mesh.positions.shape == (6, 3)
    assert mesh.indices.tolist() == [[0, 1, 2], [3, 4, 5]]
    np.testing.assert_allclose(mesh.positions[3], np.array([2.0, 0.0, 0.0], dtype=np.float32))


def test_import_gltf_applies_node_translation(tmp_path) -> None:
    path = tmp_path / "translated_node.gltf"
    _write_translated_node_gltf(path)

    mesh = io.import_gltf(str(path))

    np.testing.assert_allclose(mesh.positions[0], np.array([10.0, 20.0, 30.0], dtype=np.float32))
    np.testing.assert_allclose(mesh.positions[1], np.array([11.0, 20.0, 30.0], dtype=np.float32))


def test_import_gltf_preserves_pbr_material_metadata(tmp_path) -> None:
    path = tmp_path / "materials.gltf"
    _write_material_gltf(path)

    mesh, materials, primitive_materials = io.import_gltf(str(path), with_materials=True)

    assert mesh.positions.shape == (6, 3)
    assert primitive_materials == [0, 1]
    assert [material.name for material in materials] == ["mat_red", "mat_blue"]
    np.testing.assert_allclose(
        materials[0].base_color_factor,
        np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32),
    )
    assert materials[0].metallic_factor == 0.25
    assert materials[0].roughness_factor == 0.75
    assert materials[0].alpha_mode == "BLEND"
    assert materials[0].double_sided is True
    assert materials[0].unlit is True
    assert materials[0].has_base_color_texture is True
    assert materials[0].has_metallic_roughness_texture is True
    assert materials[0].has_normal_texture is True
    assert materials[0].has_occlusion_texture is True
    assert materials[0].has_emissive_texture is True
    assert materials[0].base_color_texture is not None
    assert materials[0].base_color_texture.texture_index == 0
    assert materials[0].base_color_texture.image_index == 0
    assert materials[0].base_color_texture.tex_coord == 1
    assert materials[0].base_color_texture.transform is not None
    np.testing.assert_allclose(
        materials[0].base_color_texture.transform.offset,
        np.array([0.25, 0.5], dtype=np.float32),
    )
    np.testing.assert_allclose(
        materials[0].base_color_texture.transform.scale,
        np.array([2.0, 3.0], dtype=np.float32),
    )
    assert materials[0].base_color_texture.transform.rotation == pytest.approx(0.125)
    assert materials[0].base_color_texture.transform.tex_coord == 0
    np.testing.assert_allclose(
        materials[0].emissive_factor,
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    assert materials[1].alpha_mode == "MASK"
    assert materials[1].alpha_cutoff == pytest.approx(0.4)


def test_mapscene_composites_textured_gltf_landmark(tmp_path) -> None:
    gltf_path = tmp_path / "materials.glb"
    _write_material_glb(gltf_path)
    texture = np.zeros((8, 8, 4), dtype=np.uint8)
    texture[..., 0] = np.linspace(40, 240, 8, dtype=np.uint8)[None, :]
    texture[..., 1] = np.linspace(240, 40, 8, dtype=np.uint8)[:, None]
    texture[..., 2] = 90
    texture[..., 3] = 255
    texture_path = tmp_path / "landmark-albedo.png"
    f3d.numpy_to_png(texture_path, texture)
    layer = f3d.MapSceneBuildingLayer(
        layer_id="textured-landmark",
        source={"path": str(gltf_path), "source_format": "gltf"},
        support_level="supported",
        geometry_count=1,
        material_status="textured_pbr",
        metadata={
            "source_id": "textured-landmark",
            "gltf_path": str(gltf_path),
            "screen_rect": [0.25, 0.25, 0.75, 0.75],
            "textured_materials": [
                {
                    "material_id": "mat_red",
                    "object_id": "landmark",
                    "albedo_texture": str(texture_path),
                    "texture_format": "png",
                    "uv_available": True,
                }
            ],
        },
    )
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"source_id": "inline-dem", "width": 4, "height": 4},
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
        layers=[layer],
    )
    base = np.zeros((64, 64, 4), dtype=np.uint8)
    base[..., 3] = 255

    rendered, used, metadata = map_scene._composite_textured_landmark_layers(base, scene.recipe)

    assert used is True
    assert metadata["gltf_textured_backend"] == "mapscene_textured_landmark"
    assert metadata["gltf_material_count"] == 2
    assert metadata["gltf_primitive_count"] == 2
    assert rendered[20:44, 20:44, :3].max() > 0
