from forge3d.bench import run_benchmark, run_vt_frame_time_comparison
from forge3d.diagnostics import memory_budget_validation_report, memory_tracking_completeness_report


def test_benchmark_result_includes_memory_and_timing_sections():
    result = run_benchmark("numpy_to_png", 4, 4, iterations=1, warmup=0)

    assert "memory" in result
    assert set(result["memory"]) == {"before", "after", "delta", "tracking"}
    assert "host_visible_bytes" in result["memory"]["after"]
    assert "budget_policy" in result["memory"]["after"]
    assert result["memory"]["tracking"]["expected_bytes"] == 4 * 4 * 4
    assert result["memory"]["tracking"]["status"] in {"supported", "underdeveloped"}

    assert "gpu_timings" in result
    assert result["gpu_timings"]["terrain_main_pass_ms"] is None
    assert result["gpu_timings"]["vt_upload_avg_ms"] is None
    assert result["gpu_timings"]["offline_accumulation_ms"] is None


def test_mapscene_benchmark_surfaces_vt_upload_timing(monkeypatch):
    import forge3d as f3d

    class FakeMapScene:
        def __init__(self, **_kwargs):
            self.last_render_metadata = None

        def render(self, path):
            self.last_render_metadata = {
                "material_vt_stats": {
                    "avg_upload_ms": 1.25,
                    "feedback_requests": 4.0,
                },
                "terrain_main_pass_ms": 2.5,
                "offline_accumulation_ms": 7.0,
            }
            with open(path, "wb") as handle:
                handle.write(b"PNG")

    monkeypatch.setattr(f3d, "TerrainSource", lambda **kwargs: kwargs)
    monkeypatch.setattr(f3d, "OrbitCamera", lambda **kwargs: kwargs)
    monkeypatch.setattr(f3d, "LightingPreset", lambda **kwargs: kwargs)
    monkeypatch.setattr(f3d, "OutputSpec", lambda **kwargs: kwargs)
    monkeypatch.setattr(f3d, "MapScene", FakeMapScene)

    result = run_benchmark("mapscene_terrain_png", 16, 16, iterations=1, warmup=0)

    assert result["gpu_timings"]["available"] is True
    assert result["gpu_timings"]["vt_upload_avg_ms"] == 1.25
    assert result["gpu_timings"]["terrain_main_pass_ms"] == 2.5
    assert result["gpu_timings"]["offline_accumulation_ms"] == 7.0


def test_mapscene_vt_benchmark_builds_vt_active_scene(monkeypatch):
    import forge3d as f3d

    captured = {}

    class FakeMapScene:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.last_render_metadata = {
                "material_vt_stats": {
                    "avg_upload_ms": 0.75,
                    "source_count": 4.0,
                    "tiles_streamed": 2.0,
                }
            }

        def render(self, path):
            with open(path, "wb") as handle:
                handle.write(b"PNG")

    monkeypatch.setattr(f3d, "TerrainSource", lambda **kwargs: kwargs)
    monkeypatch.setattr(f3d, "OrbitCamera", lambda **kwargs: kwargs)
    monkeypatch.setattr(f3d, "LightingPreset", lambda **kwargs: kwargs)
    monkeypatch.setattr(f3d, "OutputSpec", lambda **kwargs: kwargs)
    monkeypatch.setattr(f3d, "MapScene", FakeMapScene)

    result = run_benchmark("mapscene_terrain_vt_png", 16, 16, iterations=1, warmup=0)

    vt_config = captured["terrain"]["metadata"]["virtual_texture"]
    assert vt_config["enabled"] is True
    assert vt_config["use_feedback"] is True
    assert vt_config["procedural_sources"] is True
    assert captured["lighting"]["settings"]["albedo_mode"] == "material"
    assert result["gpu_timings"]["available"] is True
    assert result["gpu_timings"]["vt_upload_avg_ms"] == 0.75


def test_vt_frame_time_comparison_reports_delta(monkeypatch):
    import forge3d.bench as bench

    def fake_run_benchmark(op, width, height, *, iterations=100, warmup=10, **_kwargs):
        mean = 10.0 if op == "mapscene_terrain_png" else 12.5
        return {
            "op": op,
            "width": width,
            "height": height,
            "iterations": iterations,
            "warmup": warmup,
            "stats": {"mean_ms": mean},
            "gpu_timings": {
                "available": op == "mapscene_terrain_vt_png",
                "vt_upload_avg_ms": 0.5 if op == "mapscene_terrain_vt_png" else None,
            },
        }

    monkeypatch.setattr(bench, "run_benchmark", fake_run_benchmark)

    result = run_vt_frame_time_comparison(32, 24, iterations=3, warmup=1)

    assert result["baseline"]["op"] == "mapscene_terrain_png"
    assert result["vt_active"]["op"] == "mapscene_terrain_vt_png"
    assert result["delta_ms"] == 2.5
    assert result["delta_pct"] == 25.0
    assert result["vt_upload_avg_ms"] == 0.5
    assert result["vt_gpu_timings_available"] is True


def test_memory_budget_validation_report_contains_policy_details():
    report = memory_budget_validation_report(
        {
            "host_visible_bytes": 2048,
            "limit_bytes": 1024,
            "within_budget": False,
            "budget_policy": "warn",
            "buffer_bytes": 2048,
            "texture_bytes": 0,
        }
    )

    data = report.to_dict()
    assert data["status"] == "warning"
    assert data["diagnostics"][0]["code"] == "estimated_gpu_memory"
    assert data["diagnostics"][0]["details"]["budget_policy"] == "warn"


def test_memory_tracking_completeness_report_marks_coverage():
    report = memory_tracking_completeness_report(
        1000,
        {"host_visible_bytes": 980},
        min_coverage=0.95,
    )

    data = report.to_dict()
    assert data["status"] == "ok"
    assert data["supported_features"]["memory.tracking_completeness"] == "supported"
    details = data["diagnostics"][0]["details"]
    assert details["coverage_ratio"] == 0.98


def test_memory_tracking_completeness_report_warns_when_undertracked():
    report = memory_tracking_completeness_report(
        1000,
        {"host_visible_bytes": 500},
        min_coverage=0.95,
    )

    data = report.to_dict()
    assert data["status"] == "warning"
    assert data["supported_features"]["memory.tracking_completeness"] == "underdeveloped"
    assert data["diagnostics"][0]["code"] == "memory_tracking_completeness"


def test_benchmark_memory_tracking_uses_peak_total_bytes():
    import forge3d.bench as bench

    tracking = bench._memory_tracking_snapshot(
        64,
        64,
        {"total_bytes": 0, "peak_total_bytes": 64 * 64 * 4},
    )

    assert tracking["tracked_bytes"] == 64 * 64 * 4
    assert tracking["coverage_ratio"] == 1.0
    assert tracking["status"] == "supported"
