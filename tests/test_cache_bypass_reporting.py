"""Requested fresh paths must expose why replay is unavailable."""
import ast
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import forge3d as f3d


@pytest.mark.parametrize("options,output_changes,reason", [
    ({"certificate": True}, {}, "certificate"),
    ({"emit_provenance": True}, {}, "scene_provenance"),
    ({"provenance_signing_key": b"test-only"}, {}, "scene_provenance_signing"),
    ({}, {"format": "exr"}, "scene_output_format"),
    ({}, {"bit_depth": 16}, "scene_output_bit_depth"),
    ({}, {"aovs": ("depth",)}, "scene_aov"),
    ({}, {"hdr": True}, "scene_hdr"),
])
def test_mapscene_fresh_paths_report_reason(tmp_path, monkeypatch, options, output_changes, reason):
    from test_mapscene_render_cache import _scene
    import forge3d.certificate as certificate
    scene = _scene(tmp_path / "frame.png")
    scene.recipe.output = replace(scene.recipe.output, **output_changes)
    marker = object()
    monkeypatch.setattr(scene, "_render_impl", lambda *a, **kw: marker)
    monkeypatch.setattr(certificate, "_render_capture", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(certificate, "emit_render_certificate", lambda *a, **kw: None)
    assert scene.render(cache=tmp_path / "cache", **options) is marker
    assert scene.last_render_metadata["bypass_reason"] == reason
    assert scene.last_render_metadata["cache_hit"] is False
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("certificate", [False, True])
def test_chronos_exposes_sequence_and_certificate_bypasses(tmp_path, monkeypatch, certificate):
    import forge3d.anamnesis as anamnesis
    from forge3d.chronos import _render_frames
    scene = SimpleNamespace(last_render_metadata={}, render=lambda *a, **kw: None)
    result = SimpleNamespace(frame_blobs=[b"png"], bypass_reason="scene_furniture")
    monkeypatch.setattr(anamnesis, "render_sequence", lambda *a, **kw: result)
    reason = "certificate" if certificate else "scene_furniture"
    with pytest.warns(RuntimeWarning, match=reason):
        _render_frames({0: scene}, {}, {0: tmp_path / "frame.png"},
                       certificate_paths={0: tmp_path / "certificate.json"} if certificate else None,
                       cache=tmp_path / "cache")
    if not certificate:
        assert scene.last_render_metadata["bypass_reason"] == reason


def test_unsupported_renderers_report_cache_requests_and_none_is_quiet():
    from forge3d._cache_policy import report_cache_bypass
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        report_cache_bypass(None, "render")
    assert not caught
    with pytest.warns(RuntimeWarning, match="render_cache_unsupported"):
        report_cache_bypass("cache", "render")


# These delegates reach a Python cache owner or the native terrain cache.
# A bare forward to any other native function is insufficient evidence.
_CACHE_OWNERS = {
    ("anamnesis.py", "render_sequence"),
    ("_anamnesis_scene.py", "try_render_sequence"),
}
_CACHE_DELEGATES = {
    ("__init__.py", "render_triangle_png", "self.render_triangle_rgba"),
    ("chronos.py", "render_flythrough", "_render_frames"),
    ("map_scene.py", "render", "render_sequence"),
    ("map_scene.py", "render_passes", "self.render"),
    ("path_tracing.py", "render_terrain_poster", "hybrid_render_terrain_reference"),
    ("publication.py", "render_series", "self._render"),
    ("sdf.py", "render_simple_scene", "renderer.render_sdf_scene"),
    ("smoke.py", "render_rgba", "self.render_projection_rgba"),
    ("terrain_demo.py", "render_sunrise_to_noon_sequence", "renderer_native.render_terrain_pbr_pom"),
}


def _handles_cache(module, node):
    if (module, node.name) in _CACHE_OWNERS:
        return True
    for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
        target = ast.unparse(call.func)
        if (target == "report_cache_bypass" and call.args
                and isinstance(call.args[0], ast.Name) and call.args[0].id == "cache"):
            return True
        if (module, node.name, target) in _CACHE_DELEGATES and any(
            k.arg == "cache" and any(isinstance(n, ast.Name) and n.id in {"cache", "scene_root"} for n in ast.walk(k.value))
            for k in call.keywords
        ):
            return True
    return False


def test_every_python_render_cache_request_has_an_owner_or_bypass():
    root = Path(f3d.__file__).parent
    for path in root.rglob("*.py"):
        module = path.relative_to(root).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_") or "render" not in node.name:
                continue
            if "cache" not in {a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}:
                continue
            assert _handles_cache(module, node), (module, node.name, node.lineno)


def test_bare_native_forward_is_not_cache_handling():
    node = ast.parse("def render(cache=None): return native(cache=cache)").body[0]
    assert not _handles_cache("new_module.py", node)


@pytest.mark.parametrize("entrypoint", ["render_oit", "render_snapshot", "render_pick_map", "render_oit_and_pick"])
@pytest.mark.parametrize("cache", [None, "cache"])
def test_vector_cache_bypass_runtime(monkeypatch, entrypoint, cache):
    import warnings
    from forge3d.vector import VectorScene
    marker = object()
    for name in ("vector_render_oit_py", "vector_render_pick_map_py", "vector_render_oit_and_pick_py"):
        monkeypatch.setattr(f3d, name, lambda *a, **kw: marker)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert getattr(VectorScene(), entrypoint)(4, 4, cache=cache) is marker
    assert len(caught) == (0 if cache is None else 1)
    if caught:
        assert "render_cache_unsupported" in str(caught[0].message)


@pytest.mark.parametrize("entrypoint", ["render_fused", "render_fused_sequence", "render_fused_reference"])
@pytest.mark.parametrize("cache", [None, "cache"])
def test_splat_cache_bypass_precedes_native_dispatch(monkeypatch, entrypoint, cache):
    import warnings
    import forge3d.splat as splat
    class ReachedNative(Exception):
        pass
    def native(name):
        assert name == entrypoint
        raise ReachedNative
    monkeypatch.setattr(splat, "_native", native)
    import inspect
    signature = inspect.signature(getattr(splat, entrypoint))
    required = {name: None for name, p in signature.parameters.items()
                if p.default is inspect.Parameter.empty and p.kind != inspect.Parameter.VAR_KEYWORD}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ReachedNative):
            getattr(splat, entrypoint)(**required, cache=cache)
    assert len(caught) == (0 if cache is None else 1)
    if caught:
        assert "render_cache_unsupported" in str(caught[0].message)


@pytest.mark.parametrize("cache", [None, "cache"])
def test_offscreen_native_scene_cache_bypass(monkeypatch, cache):
    import warnings
    import forge3d.helpers.offscreen as offscreen
    marker = object()
    monkeypatch.setattr(offscreen, "_is_native_scene", lambda scene: True)
    calls = []
    scene = SimpleNamespace(render_rgba=lambda **kw: calls.append(kw) or marker)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert offscreen.render_offscreen_rgba(4, 4, scene=scene, cache=cache) is marker
    assert calls == [{"certificate": False, "cache": None}]
    assert len(caught) == (0 if cache is None else 1)
