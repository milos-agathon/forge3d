# tests/test_bundles_support.py
# Verify bundle capability detection toggles with native features.
# Ensures R7 shim uses _forge3d attribute presence for detection.
# RELEVANT FILES:python/forge3d/bundles.py,tests/test_bundles_support.py

from types import SimpleNamespace

import forge3d.bundles as bundles


def _restore_native(module):
    bundles._native = module
    bundles.refresh_bundles_support()


def test_bundles_support_false():
    original = getattr(bundles, "_native", None)
    try:
        bundles._native = None
        assert bundles.refresh_bundles_support(None) is False
        assert bundles.has_bundles_support() is False
    finally:
        _restore_native(original)


def test_bundles_support_true():
    original = getattr(bundles, "_native", None)
    fake_native = SimpleNamespace(render_bundle_compile=lambda *args, **kwargs: None)
    try:
        bundles._native = fake_native
        assert bundles.refresh_bundles_support(fake_native) is True
        assert bundles.has_bundles_support() is True
    finally:
        _restore_native(original)
