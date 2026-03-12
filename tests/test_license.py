"""Tests for forge3d._license."""

import datetime as dt
import inspect

import pytest

from forge3d._license import (
    LicenseError,
    _check_expiry_with_grace,
    _get_license_state,
    _reset_license_state,
    requires_pro,
    set_license_key,
)


@pytest.fixture(autouse=True)
def clear_license(monkeypatch):
    """Reset cached license state and remove env overrides between tests."""

    monkeypatch.delenv("FORGE3D_LICENSE_KEY", raising=False)
    _reset_license_state(allow_env_reload=True)
    yield
    set_license_key("")
    _reset_license_state(allow_env_reload=True)


def test_license_error_message_has_help_text():
    """LicenseError includes setup guidance and the Pro URL."""

    err = LicenseError("PDF export")
    message = str(err)
    assert "PDF export requires a Pro license." in message
    assert "forge3d.set_license_key" in message
    assert "https://forge3d.dev/pro" in message


def test_set_license_key_clears_state():
    """An empty key clears the active license state."""

    set_license_key("")
    state = _get_license_state()
    assert state["valid"] is False
    assert state["key"] is None


def test_set_license_key_rejects_invalid_format():
    """Malformed keys are rejected."""

    with pytest.raises(LicenseError, match="Invalid key format"):
        set_license_key("not-a-license")


def test_set_license_key_accepts_well_formed_key():
    """A well-formed key populates cached tier and expiry."""

    set_license_key("F3D-PRO-20991231-test-signature")
    state = _get_license_state()
    assert state["valid"] is True
    assert state["tier"] == "pro"
    assert state["expiry"] == dt.date(2099, 12, 31)


def test_license_key_can_load_from_environment(monkeypatch):
    """The first Pro check can load a key from FORGE3D_LICENSE_KEY."""

    monkeypatch.setenv("FORGE3D_LICENSE_KEY", "F3D-PRO-20991231-env-signature")
    _reset_license_state(allow_env_reload=True)
    state = _get_license_state()
    assert state["valid"] is True
    assert state["key"] == "F3D-PRO-20991231-env-signature"


def test_requires_pro_blocks_without_key():
    """The decorator raises LicenseError when no key is present."""

    @requires_pro(feature="Scene bundle save")
    def guarded() -> str:
        return "ok"

    with pytest.raises(LicenseError, match="Scene bundle save requires a Pro license"):
        guarded()


def test_requires_pro_preserves_signature():
    """Decorated callables keep their original inspect.signature output."""

    @requires_pro(feature="Test feature")
    def guarded(alpha: int, beta: str = "x") -> str:
        return beta * alpha

    sig = inspect.signature(guarded)
    assert str(sig) == "(alpha: int, beta: str = 'x') -> str"


def test_grace_period_status_active():
    """Non-expired licenses are reported as active."""

    today = dt.date(2026, 3, 12)
    status = _check_expiry_with_grace(dt.date(2026, 3, 20), today=today)
    assert status["status"] == "active"
    assert status["days_remaining"] == 8


def test_grace_period_status_grace():
    """Recently expired licenses remain usable during grace."""

    today = dt.date(2026, 3, 12)
    status = _check_expiry_with_grace(dt.date(2026, 3, 5), today=today)
    assert status["status"] == "grace"
    assert status["days_remaining"] == 7


def test_grace_period_status_expired():
    """Expired licenses eventually stop working."""

    today = dt.date(2026, 3, 12)
    status = _check_expiry_with_grace(dt.date(2026, 2, 20), today=today)
    assert status["status"] == "expired"
    assert status["days_remaining"] == 0


def test_requires_pro_warns_during_grace():
    """Calls still work during grace but emit a renewal warning."""

    expired_recently = (dt.date.today() - dt.timedelta(days=3)).strftime("%Y%m%d")
    set_license_key(f"F3D-PRO-{expired_recently}-grace-signature")

    @requires_pro(feature="Map plate composition")
    def guarded() -> str:
        return "ok"

    with pytest.warns(UserWarning, match="expired on"):
        assert guarded() == "ok"
