"""TV12: offline controller behavior, OIDN wrapper, and denoiser contract."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from forge3d.denoise_oidn import oidn_available, oidn_denoise
from forge3d.offline import render_offline
from forge3d.terrain_params import DenoiseSettings, OfflineQualitySettings


class _FakeHdrFrame:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.ascontiguousarray(data, dtype=np.float32)
        self.size = (self._data.shape[1], self._data.shape[0])

    def to_numpy_f32(self) -> np.ndarray:
        return self._data


class _FakeAovFrame:
    def __init__(self, albedo: np.ndarray, normal: np.ndarray) -> None:
        self._albedo = np.ascontiguousarray(albedo, dtype=np.float32)
        self._normal = np.ascontiguousarray(normal, dtype=np.float32)

    def albedo(self) -> np.ndarray:
        return self._albedo

    def normal(self) -> np.ndarray:
        return self._normal


class _FakeRenderer:
    def __init__(
        self,
        hdr_frame: _FakeHdrFrame,
        aov_frame: _FakeAovFrame,
        *,
        metrics_sequence: list[dict[str, float]] | None = None,
        fail_stage: str | None = None,
    ) -> None:
        self._hdr_frame = hdr_frame
        self._aov_frame = aov_frame
        self._samples = 0
        self.active = False
        self.uploaded: np.ndarray | None = None
        self.begin_kwargs: dict[str, object] = {}
        self._metrics_sequence = metrics_sequence or [
            {
                "mean_delta": 0.0,
                "p95_delta": 0.0,
                "max_tile_delta": 0.0,
                "converged_tile_ratio": 1.0,
            }
        ]
        self._metric_index = 0
        self._fail_stage = fail_stage

    def begin_offline_accumulation(self, **kwargs: object) -> None:
        self.begin_kwargs = dict(kwargs)
        self.active = True
        self._samples = 0
        self._metric_index = 0

    def accumulate_batch(self, sample_count: int) -> dict[str, float]:
        if self._fail_stage == "accumulate_batch":
            raise RuntimeError("accumulate_batch failed")
        self._samples += int(sample_count)
        return {"total_samples": self._samples, "batch_time_ms": 0.25}

    def read_accumulation_metrics(self, _target_variance: float, _tile_size: int) -> dict[str, float]:
        if self._fail_stage == "read_accumulation_metrics":
            raise RuntimeError("read_accumulation_metrics failed")
        index = min(self._metric_index, len(self._metrics_sequence) - 1)
        template = self._metrics_sequence[index]
        self._metric_index += 1
        return {
            "total_samples": self._samples,
            "mean_delta": float(template["mean_delta"]),
            "p95_delta": float(template["p95_delta"]),
            "max_tile_delta": float(template["max_tile_delta"]),
            "converged_tile_ratio": float(template["converged_tile_ratio"]),
        }

    def resolve_offline_hdr(self) -> tuple[_FakeHdrFrame, _FakeAovFrame]:
        if self._fail_stage == "resolve_offline_hdr":
            raise RuntimeError("resolve_offline_hdr failed")
        return self._hdr_frame, self._aov_frame

    def upload_hdr_frame(self, data: np.ndarray, size: tuple[int, int]) -> _FakeHdrFrame:
        self.uploaded = np.ascontiguousarray(data, dtype=np.float32)
        rgba = np.concatenate(
            [self.uploaded, np.ones((*self.uploaded.shape[:2], 1), dtype=np.float32)],
            axis=-1,
        )
        assert size == (rgba.shape[1], rgba.shape[0])
        return _FakeHdrFrame(rgba)

    def tonemap_offline_hdr(self, hdr_frame: _FakeHdrFrame) -> str:
        if self._fail_stage == "tonemap_offline_hdr":
            raise RuntimeError("tonemap_offline_hdr failed")
        self.active = False
        return f"tonemapped:{hdr_frame.size}"

    def end_offline_accumulation(self) -> None:
        self.active = False


def test_offline_controller_falls_back_to_atrous_when_oidn_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    beauty = np.ones((4, 4, 4), dtype=np.float32) * 0.25
    albedo = np.ones((4, 4, 3), dtype=np.float32) * 0.5
    normal = np.dstack(
        [
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
            np.ones((4, 4), dtype=np.float32),
        ]
    )
    renderer = _FakeRenderer(_FakeHdrFrame(beauty), _FakeAovFrame(albedo, normal))
    params = SimpleNamespace(
        aa_samples=4,
        denoise=DenoiseSettings(enabled=True, method="oidn", iterations=2),
    )

    monkeypatch.setattr("forge3d.offline.oidn_available", lambda: False)
    monkeypatch.setattr(
        "forge3d.offline.atrous_denoise",
        lambda color, **_: np.ascontiguousarray(color + 0.1, dtype=np.float32),
    )

    with pytest.warns(RuntimeWarning, match="oidn package not installed"):
        result = render_offline(
            renderer,
            material_set=object(),
            env_maps=object(),
            params=params,
            heightmap=np.zeros((4, 4), dtype=np.float32),
            settings=OfflineQualitySettings(enabled=True, adaptive=False, batch_size=4),
        )

    assert result.metadata["denoiser_used"] == "atrous"
    assert renderer.uploaded is not None
    np.testing.assert_allclose(renderer.uploaded, np.full((4, 4, 3), 0.35, dtype=np.float32))


def test_offline_controller_reports_none_when_denoiser_disabled() -> None:
    beauty = np.ones((3, 3, 4), dtype=np.float32) * 0.2
    albedo = np.ones((3, 3, 3), dtype=np.float32) * 0.4
    normal = np.dstack(
        [
            np.zeros((3, 3), dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
            np.ones((3, 3), dtype=np.float32),
        ]
    )
    renderer = _FakeRenderer(_FakeHdrFrame(beauty), _FakeAovFrame(albedo, normal))
    params = SimpleNamespace(aa_samples=2, denoise=DenoiseSettings(enabled=False, method="none"))

    result = render_offline(
        renderer,
        material_set=object(),
        env_maps=object(),
        params=params,
        heightmap=np.zeros((3, 3), dtype=np.float32),
        settings=OfflineQualitySettings(enabled=True, adaptive=False, batch_size=2),
    )

    assert result.metadata["denoiser_used"] == "none"
    assert renderer.uploaded is None


def test_offline_controller_requires_explicit_enabled_opt_in() -> None:
    beauty = np.ones((3, 3, 4), dtype=np.float32) * 0.2
    albedo = np.ones((3, 3, 3), dtype=np.float32) * 0.4
    normal = np.dstack(
        [
            np.zeros((3, 3), dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
            np.ones((3, 3), dtype=np.float32),
        ]
    )
    renderer = _FakeRenderer(_FakeHdrFrame(beauty), _FakeAovFrame(albedo, normal))
    params = SimpleNamespace(aa_samples=2, denoise=DenoiseSettings(enabled=False, method="none"))

    with pytest.raises(ValueError, match=r"OfflineQualitySettings\(enabled=True\)"):
        render_offline(
            renderer,
            material_set=object(),
            env_maps=object(),
            params=params,
            heightmap=np.zeros((3, 3), dtype=np.float32),
            settings=OfflineQualitySettings(enabled=False, adaptive=False, batch_size=2),
        )

    assert renderer.begin_kwargs == {}
    assert renderer.active is False


def test_offline_controller_method_none_matches_non_denoised_resolve() -> None:
    beauty = np.ones((5, 4, 4), dtype=np.float32) * 0.3
    albedo = np.ones((5, 4, 3), dtype=np.float32) * 0.45
    normal = np.zeros((5, 4, 3), dtype=np.float32)
    normal[..., 2] = 1.0

    disabled_renderer = _FakeRenderer(_FakeHdrFrame(beauty), _FakeAovFrame(albedo, normal))
    none_renderer = _FakeRenderer(_FakeHdrFrame(beauty), _FakeAovFrame(albedo, normal))

    disabled = render_offline(
        disabled_renderer,
        material_set=object(),
        env_maps=object(),
        params=SimpleNamespace(aa_samples=2, denoise=DenoiseSettings(enabled=False, method="none")),
        heightmap=np.zeros((5, 4), dtype=np.float32),
        settings=OfflineQualitySettings(enabled=True, adaptive=False, batch_size=2),
    )
    method_none = render_offline(
        none_renderer,
        material_set=object(),
        env_maps=object(),
        params=SimpleNamespace(aa_samples=2, denoise=DenoiseSettings(enabled=True, method="none")),
        heightmap=np.zeros((5, 4), dtype=np.float32),
        settings=OfflineQualitySettings(enabled=True, adaptive=False, batch_size=2),
    )

    np.testing.assert_allclose(disabled.hdr_frame.to_numpy_f32(), beauty)
    np.testing.assert_allclose(method_none.hdr_frame.to_numpy_f32(), beauty)
    assert disabled.frame == method_none.frame
    assert disabled.metadata["denoiser_used"] == method_none.metadata["denoiser_used"] == "none"
    assert disabled_renderer.uploaded is None
    assert none_renderer.uploaded is None


def test_offline_controller_passes_aovs_aligned_with_beauty_to_oidn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    beauty = np.ones((4, 6, 4), dtype=np.float32) * 0.2
    albedo = np.ones((4, 6, 3), dtype=np.float32) * 0.5
    normal = np.zeros((4, 6, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    renderer = _FakeRenderer(_FakeHdrFrame(beauty), _FakeAovFrame(albedo, normal))
    params = SimpleNamespace(
        aa_samples=4,
        denoise=DenoiseSettings(enabled=True, method="oidn", iterations=2),
    )
    captured: dict[str, tuple[int, ...]] = {}

    def _fake_oidn(
        beauty_hdr: np.ndarray,
        *,
        albedo: np.ndarray | None = None,
        normal: np.ndarray | None = None,
        **_: object,
    ) -> np.ndarray:
        captured["beauty"] = beauty_hdr.shape
        captured["albedo"] = () if albedo is None else albedo.shape
        captured["normal"] = () if normal is None else normal.shape
        return np.ascontiguousarray(beauty_hdr + 0.05, dtype=np.float32)

    monkeypatch.setattr("forge3d.offline.oidn_available", lambda: True)
    monkeypatch.setattr("forge3d.offline.oidn_denoise", _fake_oidn)

    result = render_offline(
        renderer,
        material_set=object(),
        env_maps=object(),
        params=params,
        heightmap=np.zeros((4, 6), dtype=np.float32),
        settings=OfflineQualitySettings(enabled=True, adaptive=False, batch_size=4),
    )

    assert captured["beauty"] == (4, 6, 3)
    assert captured["albedo"] == (4, 6, 3)
    assert captured["normal"] == (4, 6, 3)
    assert result.metadata["denoiser_used"] == "oidn"
    assert renderer.uploaded is not None
    np.testing.assert_allclose(renderer.uploaded, np.full((4, 6, 3), 0.25, dtype=np.float32))


@pytest.mark.parametrize(
    ("adaptive", "aa_samples", "max_samples", "expected_jitter_sequence_samples"),
    [
        (False, 6, 16, 6),
        (True, 6, 16, 16),
    ],
)
def test_offline_controller_passes_full_jitter_budget_to_offline_session(
    adaptive: bool,
    aa_samples: int,
    max_samples: int,
    expected_jitter_sequence_samples: int,
) -> None:
    beauty = np.ones((4, 4, 4), dtype=np.float32) * 0.2
    albedo = np.ones((4, 4, 3), dtype=np.float32) * 0.4
    normal = np.zeros((4, 4, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    renderer = _FakeRenderer(_FakeHdrFrame(beauty), _FakeAovFrame(albedo, normal))
    params = SimpleNamespace(aa_samples=aa_samples, denoise=DenoiseSettings(enabled=False, method="none"))

    render_offline(
        renderer,
        material_set=object(),
        env_maps=object(),
        params=params,
        heightmap=np.zeros((4, 4), dtype=np.float32),
        settings=OfflineQualitySettings(
            enabled=True,
            adaptive=adaptive,
            min_samples=4,
            max_samples=max_samples,
            batch_size=4,
        ),
    )

    assert renderer.begin_kwargs["jitter_sequence_samples"] == expected_jitter_sequence_samples


def test_offline_controller_rejects_misaligned_aovs_for_oidn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    beauty = np.ones((4, 4, 4), dtype=np.float32) * 0.2
    bad_albedo = np.ones((3, 4, 3), dtype=np.float32) * 0.5
    normal = np.zeros((4, 4, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    renderer = _FakeRenderer(_FakeHdrFrame(beauty), _FakeAovFrame(bad_albedo, normal))
    params = SimpleNamespace(
        aa_samples=4,
        denoise=DenoiseSettings(enabled=True, method="oidn", iterations=2),
    )

    monkeypatch.setattr("forge3d.offline.oidn_available", lambda: True)

    with pytest.raises(RuntimeError, match="albedo shape"):
        render_offline(
            renderer,
            material_set=object(),
            env_maps=object(),
            params=params,
            heightmap=np.zeros((4, 4), dtype=np.float32),
            settings=OfflineQualitySettings(enabled=True, adaptive=False, batch_size=4),
        )

    assert renderer.uploaded is None


def test_offline_controller_requires_three_metric_snapshots_before_early_stop_and_reports_metadata() -> None:
    beauty = np.ones((4, 4, 4), dtype=np.float32) * 0.2
    albedo = np.ones((4, 4, 3), dtype=np.float32) * 0.4
    normal = np.zeros((4, 4, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    renderer = _FakeRenderer(
        _FakeHdrFrame(beauty),
        _FakeAovFrame(albedo, normal),
        metrics_sequence=[
            {
                "mean_delta": 0.001,
                "p95_delta": 0.0,
                "max_tile_delta": 0.002,
                "converged_tile_ratio": 1.0,
            }
        ],
    )
    params = SimpleNamespace(aa_samples=16, denoise=DenoiseSettings(enabled=False, method="none"))
    progress_events = []

    result = render_offline(
        renderer,
        material_set=object(),
        env_maps=object(),
        params=params,
        heightmap=np.zeros((4, 4), dtype=np.float32),
        settings=OfflineQualitySettings(
            enabled=True,
            adaptive=True,
            min_samples=4,
            max_samples=16,
            batch_size=4,
            target_variance=0.001,
            convergence_ratio=0.95,
        ),
        progress_callback=progress_events.append,
    )

    assert [event.samples_so_far for event in progress_events] == [4, 8, 12]
    assert result.metadata == {
        "samples_used": 12,
        "denoiser_used": "none",
        "final_p95_delta": 0.0,
        "converged_ratio": 1.0,
        "target_samples": 16,
        "adaptive": True,
    }
    assert renderer.active is False
    assert renderer.uploaded is None


@pytest.mark.parametrize(
    "fail_stage",
    ["accumulate_batch", "read_accumulation_metrics", "resolve_offline_hdr", "tonemap_offline_hdr"],
)
def test_offline_controller_cleans_up_session_on_renderer_errors(fail_stage: str) -> None:
    beauty = np.ones((4, 4, 4), dtype=np.float32) * 0.2
    albedo = np.ones((4, 4, 3), dtype=np.float32) * 0.4
    normal = np.zeros((4, 4, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    renderer = _FakeRenderer(
        _FakeHdrFrame(beauty),
        _FakeAovFrame(albedo, normal),
        metrics_sequence=[
            {
                "mean_delta": 0.01,
                "p95_delta": 0.01,
                "max_tile_delta": 0.02,
                "converged_tile_ratio": 0.5,
            }
        ],
        fail_stage=fail_stage,
    )
    params = SimpleNamespace(aa_samples=8, denoise=DenoiseSettings(enabled=False, method="none"))

    with pytest.raises(RuntimeError, match=fail_stage):
        render_offline(
            renderer,
            material_set=object(),
            env_maps=object(),
            params=params,
            heightmap=np.zeros((4, 4), dtype=np.float32),
            settings=OfflineQualitySettings(
                enabled=True,
                adaptive=True,
                min_samples=4,
                max_samples=8,
                batch_size=4,
            ),
        )

    assert renderer.active is False


def test_oidn_runtime_wrapper_executes_when_backend_is_present() -> None:
    if not oidn_available():
        pytest.skip("OIDN runtime not installed in this environment")

    rng = np.random.default_rng(42)
    beauty = np.clip(0.5 + rng.normal(0.0, 0.12, (16, 16, 3)), 0.0, 1.0).astype(np.float32)
    albedo = np.ones((16, 16, 3), dtype=np.float32) * 0.5
    normal = np.zeros((16, 16, 3), dtype=np.float32)
    normal[..., 2] = 1.0

    denoised = oidn_denoise(beauty, albedo=albedo, normal=normal)
    assert denoised.shape == beauty.shape
    assert denoised.dtype == np.float32
    assert not np.allclose(denoised, beauty)
