"""Offline session ownership and failure behavior, without requiring a GPU."""

from types import SimpleNamespace

import numpy as np
import pytest

from forge3d import offline
from forge3d.terrain_params import DenoiseSettings, OfflineQualitySettings


class _Hdr:
    def __init__(self, pixels):
        self.pixels = pixels
        self.size = (pixels.shape[1], pixels.shape[0])

    def to_numpy_f32(self):
        return self.pixels


class _Renderer:
    def __init__(self, *, native_completion=False, attributes=False):
        self.active = False
        self.native_completion = native_completion
        self.attributes = attributes
        self.events = []
        self.failure_stage = None
        self.failure = RuntimeError("injected render failure")
        self.cleanup_failure = None
        self.batch_limit = None
        self.ratios = []
        self.aov_shape = (2, 3, 3)

    def _event(self, stage):
        self.events.append(stage)
        if self.failure_stage == stage:
            raise self.failure

    def _result(self, **values):
        return SimpleNamespace(**values) if self.attributes else values

    def begin_offline_accumulation(self, **kwargs):
        self._event("begin")
        if self.active:
            raise RuntimeError("already active")
        self.active = True
        self.begin_kwargs = kwargs
        self.total_samples = 0

    def accumulate_batch(self, count):
        assert self.active
        self._event("batch")
        self.total_samples += count if self.batch_limit is None else min(count, self.batch_limit)
        return self._result(total_samples=self.total_samples)

    def read_accumulation_metrics(self, target_variance, tile_size):
        assert self.active
        self._event("metrics")
        ratio = self.ratios.pop(0) if self.ratios else 1.0
        return self._result(
            total_samples=self.total_samples,
            mean_delta=0.0001,
            p95_delta=0.0002,
            max_tile_delta=0.0003,
            converged_tile_ratio=ratio,
        )

    def resolve_offline_hdr(self):
        assert self.active
        self._event("resolve")
        self.hdr = _Hdr(np.ones((2, 3, 4), dtype=np.float32))
        self.aov = SimpleNamespace(albedo=self._albedo, normal=self._normal)
        return self.hdr, self.aov

    def _albedo(self):
        self._event("albedo")
        return np.full(self.aov_shape, 0.5, dtype=np.float32)

    def _normal(self):
        self._event("normal")
        return np.ones(self.aov_shape, dtype=np.float32)

    def upload_hdr_frame(self, pixels, size):
        assert self.active
        self._event("upload")
        self.uploaded = _Hdr(pixels)
        assert self.uploaded.size == size
        return self.uploaded

    def tonemap_offline_hdr(self, hdr):
        assert self.active
        self._event("tonemap")
        self.frame = SimpleNamespace(hdr=hdr)
        if self.native_completion:
            self.active = False
        return self.frame

    def end_offline_accumulation(self):
        self.events.append("end")
        if self.cleanup_failure is not None:
            raise self.cleanup_failure
        self.active = False


def _render(renderer, *, settings=None, params=None, progress_callback=None, water_mask=None):
    return offline.render_offline(
        renderer,
        "materials",
        "environment",
        params or SimpleNamespace(aa_samples=6, denoise=None),
        np.zeros((2, 3), dtype=np.float32),
        settings=settings or OfflineQualitySettings(enabled=True, batch_size=4),
        progress_callback=progress_callback,
        water_mask=water_mask,
    )


@pytest.mark.parametrize("native_completion", [False, True])
@pytest.mark.parametrize("attributes", [False, True])
def test_success_ends_session_and_keeps_native_frames_and_actual_sample_counts(
    native_completion, attributes
):
    renderer = _Renderer(native_completion=native_completion, attributes=attributes)
    renderer.batch_limit = 2
    water_mask = np.ones((2, 3), dtype=np.float32)
    progress = []
    result = _render(renderer, progress_callback=progress.append, water_mask=water_mask)

    assert not renderer.active
    assert renderer.events[-3:] == ["resolve", "tonemap", "end"]
    assert renderer.events.count("end") == 1
    assert renderer.begin_kwargs["water_mask"] is water_mask
    assert renderer.begin_kwargs["jitter_sequence_samples"] == 6
    assert [update.samples_so_far for update in progress] == [2, 4, 6]
    assert all(update.max_samples == 6 for update in progress)
    assert result.frame is renderer.frame
    assert result.hdr_frame is renderer.hdr
    assert result.aov_frame is renderer.aov
    assert result.metadata == {
        "samples_used": 6,
        "target_samples": 6,
        "adaptive": False,
        "denoiser_used": "none",
        "final_p95_delta": 0.0002,
        "converged_ratio": 1.0,
    }

    next_result = _render(renderer)
    assert next_result.frame is not result.frame
    assert next_result.hdr_frame is not result.hdr_frame
    assert next_result.aov_frame is not result.aov_frame


@pytest.mark.parametrize("stage", ["batch", "metrics", "resolve", "upload", "tonemap", "atrous", "oidn"])
def test_failure_ends_session_preserves_error_and_allows_next_render(monkeypatch, stage):
    renderer = _Renderer()
    renderer.failure_stage = stage

    def denoise(beauty, **kwargs):
        renderer._event("oidn" if stage == "oidn" else "atrous")
        return beauty

    monkeypatch.setattr(offline, "oidn_available", lambda: True)
    monkeypatch.setattr(offline, "oidn_denoise", denoise)
    monkeypatch.setattr(offline, "atrous_denoise", denoise)
    method = "oidn" if stage == "oidn" else "atrous"
    params = SimpleNamespace(aa_samples=6, denoise=DenoiseSettings(enabled=True, method=method))
    with pytest.raises(RuntimeError) as caught:
        _render(renderer, params=params, progress_callback=lambda update: None)

    assert caught.value is renderer.failure
    assert not renderer.active
    assert renderer.events[-1] == "end"
    assert renderer.events.count("end") == 1
    if stage in ("atrous", "oidn", "upload", "resolve"):
        assert "tonemap" not in renderer.events

    renderer.failure_stage = None
    result = _render(renderer, params=params)
    assert result.metadata["samples_used"] == 6
    assert not renderer.active


@pytest.mark.parametrize("error", [RuntimeError("callback failure"), KeyboardInterrupt(), SystemExit(1)])
def test_callback_failure_or_cancellation_ends_session(error):
    renderer = _Renderer()

    def callback(update):
        raise error

    with pytest.raises(type(error)) as caught:
        _render(renderer, progress_callback=callback)
    assert caught.value is error
    assert not renderer.active
    assert renderer.events == ["begin", "batch", "metrics", "end"]
    assert _render(renderer).metadata["samples_used"] == 6


def test_rejected_begin_preserves_existing_session():
    renderer = _Renderer()
    renderer.active = True
    with pytest.raises(RuntimeError, match="already active"):
        _render(renderer)
    assert renderer.active
    assert renderer.events == ["begin"]


def test_disabled_settings_do_not_begin_session():
    renderer = _Renderer()
    with pytest.raises(ValueError, match="enabled=True"):
        _render(renderer, settings=OfflineQualitySettings())
    assert renderer.events == []


@pytest.mark.parametrize("render_fails", [False, True])
def test_cleanup_failure_is_visible_without_masking_render_failure(render_fails):
    renderer = _Renderer()
    renderer.cleanup_failure = RuntimeError("injected cleanup failure")
    if render_fails:
        renderer.failure_stage = "batch"
    with pytest.raises(RuntimeError) as caught:
        _render(renderer)
    assert renderer.events[-1] == "end"
    if render_fails:
        assert caught.value is renderer.failure
        assert caught.value.__cause__ is renderer.cleanup_failure
    else:
        assert caught.value is renderer.cleanup_failure


@pytest.mark.parametrize("ratios,expected_samples", [([0.96, 0.98, 1.0], 12), ([1.0, 0.99, 0.98, 0.97], 16)])
def test_adaptive_stopping_keeps_trend_requirement_and_jitter_horizon(ratios, expected_samples):
    renderer = _Renderer()
    renderer.ratios = ratios.copy()
    settings = OfflineQualitySettings(enabled=True, adaptive=True, max_samples=16, min_samples=4)
    progress = []
    result = _render(renderer, settings=settings, progress_callback=progress.append)
    assert result.metadata["samples_used"] == expected_samples
    assert result.metadata["target_samples"] == 16
    assert renderer.begin_kwargs["jitter_sequence_samples"] == 16
    assert progress[-1].samples_so_far == expected_samples
    assert result.metadata["converged_ratio"] == ratios[-1]
    assert not renderer.active


@pytest.mark.parametrize("method,available,used", [("oidn", True, "oidn"), ("oidn", False, "atrous"), ("atrous", True, "atrous")])
def test_denoise_order_guidance_upload_and_fallback_warning(monkeypatch, method, available, used):
    renderer = _Renderer()

    def denoise(beauty, *, albedo, normal, **kwargs):
        renderer._event(used)
        assert beauty.shape == albedo.shape == normal.shape == (2, 3, 3)
        assert beauty.dtype == albedo.dtype == normal.dtype == np.float32
        return beauty * 0.5

    monkeypatch.setattr(offline, "oidn_available", lambda: available)
    monkeypatch.setattr(offline, "oidn_denoise", denoise)
    monkeypatch.setattr(offline, "atrous_denoise", denoise)
    # Native TerrainRenderParams exposes its Python configuration this way.
    params = SimpleNamespace(
        aa_samples=1,
        python_object=SimpleNamespace(denoise=DenoiseSettings(enabled=True, method=method)),
    )
    if method == "oidn" and not available:
        with pytest.warns(RuntimeWarning, match="oidn package not installed; falling back to atrous denoiser"):
            result = _render(renderer, params=params)
    else:
        result = _render(renderer, params=params)
    assert renderer.events == ["begin", "batch", "resolve", "albedo", "normal", used, "upload", "tonemap", "end"]
    assert result.metadata["denoiser_used"] == used
    assert result.hdr_frame is renderer.uploaded
    assert result.frame.hdr is result.hdr_frame
    assert result.aov_frame is renderer.aov
    np.testing.assert_array_equal(result.hdr_frame.pixels, np.full((2, 3, 3), 0.5, dtype=np.float32))


@pytest.mark.parametrize("denoise", [None, DenoiseSettings(enabled=False), DenoiseSettings(enabled=True, method="none")])
def test_disabled_denoise_keeps_resolved_hdr_without_guidance_or_upload(denoise):
    renderer = _Renderer()
    result = _render(renderer, params=SimpleNamespace(aa_samples=1, denoise=denoise))
    assert renderer.events == ["begin", "batch", "resolve", "tonemap", "end"]
    assert result.metadata["denoiser_used"] == "none"
    assert result.hdr_frame is renderer.hdr


def test_mismatched_guidance_fails_before_upload_and_cleans_up():
    renderer = _Renderer()
    renderer.aov_shape = (1, 3, 3)
    params = SimpleNamespace(aa_samples=1, denoise=DenoiseSettings(enabled=True))
    with pytest.raises(RuntimeError, match="albedo shape .* does not match beauty HDR shape"):
        _render(renderer, params=params)
    assert not renderer.active
    assert renderer.events == ["begin", "batch", "resolve", "albedo", "end"]
