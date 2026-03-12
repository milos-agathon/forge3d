"""Offline license checks for Pro-gated forge3d features."""

import datetime as _dt
import functools
import inspect
import os
import threading
import warnings
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, cast, overload

_GRACE_PERIOD_DAYS = 14
_PRO_URL = "https://forge3d.dev/pro"

P = ParamSpec("P")
R = TypeVar("R")


def _empty_license_state(*, env_loaded: bool) -> dict[str, Any]:
    return {
        "key": None,
        "tier": None,
        "expiry": None,
        "signature": None,
        "valid": False,
        "env_loaded": env_loaded,
    }


_license_lock = threading.Lock()
_license_state: dict[str, Any] = _empty_license_state(env_loaded=False)


class LicenseError(RuntimeError):
    """Raised when a Pro feature is called without a valid license."""

    def __init__(self, feature: str = "", *, detail: str | None = None) -> None:
        if detail:
            message = detail
        elif feature:
            message = f"{feature} requires a Pro license."
        else:
            message = "Pro license required."

        message += (
            "\n  Set your key: forge3d.set_license_key('F3D-...')"
            f"\n  Get a key at: {_PRO_URL}"
        )
        super().__init__(message)


def _reset_license_state(*, allow_env_reload: bool = False) -> None:
    """Reset cached state.

    Private helper used by tests to exercise env-var loading paths.
    """

    global _license_state
    with _license_lock:
        _license_state = _empty_license_state(env_loaded=not allow_env_reload)


def set_license_key(key: str) -> None:
    """Set or clear the forge3d Pro license key for the current process."""

    global _license_state

    cleaned = key.strip()
    if not cleaned:
        with _license_lock:
            # Explicit clears should not silently re-read the environment.
            _license_state = _empty_license_state(env_loaded=True)
        return

    parsed = _parse_key(cleaned)
    parsed["env_loaded"] = True
    with _license_lock:
        _license_state = parsed


def _get_license_state() -> dict[str, Any]:
    """Return the cached license state, loading from env once if needed."""

    with _license_lock:
        current = dict(_license_state)

    if current["key"] is None and not current["env_loaded"]:
        env_key = os.environ.get("FORGE3D_LICENSE_KEY", "").strip()
        if env_key:
            try:
                set_license_key(env_key)
            except LicenseError:
                _reset_license_state()
        else:
            with _license_lock:
                _license_state["env_loaded"] = True
        with _license_lock:
            current = dict(_license_state)

    return current


def _verify_signature(tier: str, expiry_text: str, signature: str) -> bool:
    """Validate a key signature.

    Phase 3 ships the offline license shape and runtime boundary first. The
    signing tool and embedded public key live in a follow-up private workflow,
    so a non-empty signature placeholder is accepted for now.
    """

    del tier, expiry_text
    return bool(signature.strip())


def _parse_key(key: str) -> dict[str, Any]:
    """Parse and validate a license key string."""

    parts = key.split("-", 3)
    if len(parts) != 4 or parts[0] != "F3D":
        raise LicenseError(
            detail="Invalid key format. Expected: F3D-TIER-YYYYMMDD-signature"
        )

    tier = parts[1].upper()
    if tier not in {"PRO", "ENTERPRISE"}:
        raise LicenseError(detail=f"Invalid license tier: {tier}")

    expiry_text = parts[2]
    try:
        expiry = _dt.datetime.strptime(expiry_text, "%Y%m%d").date()
    except ValueError as exc:
        raise LicenseError(detail="Invalid expiry date in license key.") from exc

    signature = parts[3].strip()
    if not signature:
        raise LicenseError(detail="Invalid key format. Signature is missing.")

    if not _verify_signature(tier, expiry_text, signature):
        raise LicenseError(detail="License signature verification failed.")

    return {
        "key": key,
        "tier": tier.lower(),
        "expiry": expiry,
        "signature": signature,
        "valid": True,
    }


def _check_expiry_with_grace(
    expiry: _dt.date,
    today: _dt.date | None = None,
) -> dict[str, Any]:
    """Check expiry status with a fixed grace period."""

    current_day = today or _dt.date.today()
    days_since_expiry = (current_day - expiry).days
    grace_end = expiry + _dt.timedelta(days=_GRACE_PERIOD_DAYS)

    if days_since_expiry <= 0:
        return {
            "status": "active",
            "days_remaining": -days_since_expiry,
            "grace_end": grace_end,
        }
    if days_since_expiry <= _GRACE_PERIOD_DAYS:
        return {
            "status": "grace",
            "days_remaining": _GRACE_PERIOD_DAYS - days_since_expiry,
            "grace_end": grace_end,
        }
    return {
        "status": "expired",
        "days_remaining": 0,
        "grace_end": grace_end,
    }


def _check_pro_access(feature: str = "") -> None:
    """Raise LicenseError if the current process lacks Pro access."""

    state = _get_license_state()
    if not state["valid"] or state["tier"] is None:
        raise LicenseError(feature=feature)

    expiry = cast(_dt.date | None, state.get("expiry"))
    if expiry is None:
        return

    check = _check_expiry_with_grace(expiry)
    if check["status"] == "expired":
        feature_prefix = f"{feature} requires a Pro license. " if feature else ""
        raise LicenseError(
            detail=(
                f"{feature_prefix}Your forge3d Pro license expired on {expiry}. "
                f"Grace period ended on {check['grace_end']}."
            )
        )
    if check["status"] == "grace":
        warnings.warn(
            (
                f"Your forge3d Pro license expired on {expiry}. "
                f"Renew at {_PRO_URL}. "
                f"Pro features will stop working in {check['days_remaining']} days."
            ),
            UserWarning,
            stacklevel=4,
        )


@overload
def requires_pro(func: Callable[P, R], *, feature: str = "") -> Callable[P, R]:
    ...


@overload
def requires_pro(*, feature: str = "") -> Callable[[Callable[P, R]], Callable[P, R]]:
    ...


def requires_pro(
    func: Callable[P, R] | None = None,
    *,
    feature: str = "",
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a callable so it requires a valid Pro license."""

    def decorator(inner: Callable[P, R]) -> Callable[P, R]:
        feature_name = feature or inner.__qualname__

        @functools.wraps(inner)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _check_pro_access(feature_name)
            return inner(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(inner)
        return cast(Callable[P, R], wrapper)

    if func is not None:
        return decorator(func)
    return decorator


__all__ = ["LicenseError", "requires_pro", "set_license_key"]
