"""Admission and observed process memory for the three MODUS render routes."""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from inspect import signature
import json
from operator import index
import sys
import weakref

import numpy as np

from . import _poster_memory
from ._native import get_native_symbol
from ._scene_memory import estimate_scene_job

_active = ContextVar("forge3d_process_memory", default=None)
# A weak reference to the allocating ndarray holds the native counter token.
# Views have no independent allocation. The last view keeps its owner alive.
_owners = {}


def owned_array(array, source=None):
    if _active.get() is None or not isinstance(array, np.ndarray):
        return array
    owner = array
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    if not owner.flags.owndata or id(owner) in _owners:
        return array
    if source is not None:
        if isinstance(source, (np.ndarray, memoryview)):
            if np.may_share_memory(array, source):
                return array
        elif not isinstance(source, (list, tuple, range)):
            # Opaque array providers can return their own pre-existing storage.
            return array
    counter = get_native_symbol("_OwnedArrayBytes")
    if counter is not None:
        key = id(owner)
        _owners[key] = (weakref.ref(owner, lambda ref: _owners.pop(key, None)),
                        counter(owner.nbytes))
    return array


def predict(route, options):
    if route == "render_terrain_poster":
        return _poster_memory.estimate_poster(**{
            k: v for k, v in options.items() if k in _POSTER_ARGS
        })
    return estimate_scene_job(route, options)


_POSTER_ARGS = signature(_poster_memory.estimate_poster).parameters


class Measurement:
    def __init__(self, report):
        self.report = report
        self.certificate = False
        self.digest_target = None
        self.digest_key = None
        self.flythrough = False
        factory = get_native_symbol("_ProcessMemoryCapture")
        self.sampler = factory(sys.getswitchinterval()) if factory else None
        if self.sampler is None:
            report["ledger_observation_unavailable"] = "native process sampler unavailable"

    def _observe(self):
        report = self.report
        samples = json.loads(self.sampler.finish()) if self.sampler is not None else None
        current, kernel_peak = _poster_memory.process_bytes()
        baseline = report["process_baseline_bytes"]
        previous = report["prior_peak_bytes"]
        sampled = max(baseline, current)
        if samples is not None:
            initial, at_peak = samples["baseline"], samples["at_peak"]
            sampled = at_peak["process"]
            delta = {k: at_peak[k] - initial[k] for k in
                     ("rust_heap", "python_arrays", "mapped_buffers", "host_visible", "device_local")}
            ledger_growth = delta["rust_heap"] + delta["python_arrays"] + delta["mapped_buffers"]
            if cpu := samples.get("at_cpu_peak"):
                report["sampled_cpu_peak_growth_bytes"] = (
                    cpu["rust_heap"] + cpu["python_arrays"] - initial["rust_heap"] - initial["python_arrays"]
                )
            report.update(ledger_samples=samples, ledger_growth_at_sampled_peak=delta,
                          sampled_unexplained_bytes=sampled - initial["process"] - ledger_growth)
        masked = kernel_peak <= previous and previous > sampled
        measured = sampled if masked else max(sampled, kernel_peak)
        report.update(measured_peak_bytes=measured, kernel_peak_bytes=kernel_peak,
                      peak_masked_by_history=masked, sampled_peak_bytes=sampled,
                      peak_observation="sampled_current" if masked else "kernel_high_water",
                      peak_upper_bound_bytes=max(measured, kernel_peak))
        if "predicted_peak_bytes" in report:
            report["prediction_ratio"] = report["predicted_peak_bytes"] / measured
            report["modeled_unexplained_bytes"] = measured - baseline - report["known_growth_bytes"]
        if samples is not None:
            report["unexplained_bytes"] = measured - initial["process"] - ledger_growth
            report["kernel_to_sample_gap_bytes"] = measured - sampled
            return self.sampler.attach(json.dumps(report, allow_nan=False), self.flythrough)
        return False

    def publish(self):
        from .certificate import _emit_current_render_certificate

        # The render frame has unwound. Its resource destructors, report
        # attachment and requested certificate export all belong to this call.
        # Republish only when that work advances the kernel high-water mark.
        while True:
            attached = self._observe()
            if self.certificate and (attached or not self.flythrough):
                digest = _emit_current_render_certificate(self.certificate)
                if self.digest_target is not None:
                    self.digest_target[self.digest_key] = digest
            if _poster_memory.process_bytes()[1] <= self.report["kernel_peak_bytes"]:
                return


def job_report(certificate=False, *, digest_target=None, digest_key=None, flythrough=False):
    """Keep the public report in its result while the wrapper finishes the call."""
    job = _active.get()
    job.certificate = certificate
    job.digest_target = digest_target
    job.digest_key = digest_key
    job.flythrough = flythrough
    return job.report


def scene_resource(phase, create, *args, **kwargs):
    job = _active.get()
    if job is None or job.sampler is None:
        return create(*args, **kwargs)
    job.sampler.scene_phase(phase)
    try:
        return create(*args, **kwargs)
    finally:
        job.sampler.scene_phase("render setup and publication")


def measured_render(route):
    def decorate(function):
        public = signature(function)

        @wraps(function)
        def wrapper(*args, **kwargs):
            options = public.bind(*args, **kwargs).arguments
            limit = options.get("process_limit_bytes")
            if limit is not None:
                if isinstance(limit, (bool, np.bool_)):
                    raise TypeError("process_limit_bytes must be an integer byte count")
                limit = index(limit)
                if limit < 0:
                    raise ValueError("process_limit_bytes must be non-negative")
            try:
                report = predict(route, options)
            except _poster_memory.UnknownInputShape as exc:
                if limit is not None:
                    exc.limit_bytes = limit
                    raise
                baseline, previous = _poster_memory.process_bytes()
                report = dict(estimate_unavailable=str(exc), process_baseline_bytes=baseline,
                              prior_peak_bytes=previous)
            if limit is not None and report["predicted_peak_bytes"] > limit:
                raise _poster_memory.MemoryAdmissionRefused(limit, report)
            job = Measurement(report)
            token = _active.set(job)
            try:
                result = function(*args, **kwargs)
            except BaseException:
                if job.sampler is not None:
                    job.sampler.finish()
                raise
            finally:
                _active.reset(token)
            job.publish()
            return result

        return wrapper
    return decorate
