# tests/test_gltf_mr_channels.py
# Unit test for glTF metallic-roughness channel mapping.
# Exists to ensure we follow convention: G=roughness, B=metallic.
# RELEVANT FILES:python/forge3d/textures.py

import numpy as np

from forge3d.textures import gltf_mr_channels


def test_gltf_mr_channel_mapping():
    h, w = 4, 5
    img = np.zeros((h, w, 4), dtype=np.uint8)
    # encode roughness gradient in G and metallic constant in B
    img[..., 1] = np.arange(w, dtype=np.uint8)[None, :]
    img[..., 2] = 77
    rough, metal = gltf_mr_channels(img)
    assert rough.shape == (h, w)
    assert metal.shape == (h, w)
    assert int(metal.mean()) == 77
    assert rough[0, -1] == w - 1

