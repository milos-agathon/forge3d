from __future__ import annotations

import pytest

from forge3d import _forge3d as native


_EMITTER_VALUES = {
    "center": (1.0, 2.0, 3.0),
    "radius": 2.0,
    "density_rate": 4.0,
    "temperature_rate": 5.0,
    "fuel_rate": 6.0,
    "soot_rate": 7.0,
    "humidity_rate": 8.0,
    "emission_rate": 9.0,
    "velocity": (-1.0, 0.0, 1.0),
    "start_time": 1.0,
    "end_time": 4.0,
}
_VALID_UPDATES = {
    "center": (-2.5, 0.0, 4.5),
    "radius": 0.5,
    # Existing validation accepts finite negative rates.
    "density_rate": -0.5,
    "temperature_rate": -0.5,
    "fuel_rate": -0.5,
    "soot_rate": -0.5,
    "humidity_rate": -0.5,
    "emission_rate": -0.5,
    "velocity": (2.0, -3.0, 4.0),
    "start_time": -1.0,
    "end_time": 2.0,
}
_NONFINITE = (float("nan"), float("inf"), -float("inf"))


def _invalid_edits():
    for value in (0.0, -1.0, *_NONFINITE):
        yield pytest.param(
            "radius", value, "radius must be finite and > 0", id=f"radius-{value}"
        )
    for field in ("center", "velocity"):
        for axis in range(3):
            for value in _NONFINITE:
                vector = list(_EMITTER_VALUES[field])
                vector[axis] = value
                yield pytest.param(
                    field,
                    tuple(vector),
                    f"{field}[{axis}] must be finite",
                    id=f"{field}-{axis}-{value}",
                )
    for field in (
        "density_rate",
        "temperature_rate",
        "fuel_rate",
        "soot_rate",
        "humidity_rate",
        "emission_rate",
        "start_time",
        "end_time",
    ):
        for value in _NONFINITE:
            yield pytest.param(
                field, value, f"{field} must be finite", id=f"{field}-{value}"
            )
    yield pytest.param(
        "start_time", 5.0, "end_time must be >= start_time", id="start-after-end"
    )
    yield pytest.param(
        "end_time", 0.0, "end_time must be >= start_time", id="end-before-start"
    )


def _snapshot(emitter):
    return {field: getattr(emitter, field) for field in _EMITTER_VALUES}


@pytest.mark.parametrize("field,invalid,message", list(_invalid_edits()))
def test_native_smoke_emitter_rejects_edit_without_changing_state(
    field, invalid, message
) -> None:
    # Use the native class directly so fallback behavior cannot hide a regression.
    emitter = native.SmokeEmitter(**_EMITTER_VALUES)
    before = _snapshot(emitter)

    with pytest.raises(ValueError) as construction_error:
        native.SmokeEmitter(**{**_EMITTER_VALUES, field: invalid})
    with pytest.raises(ValueError) as update_error:
        setattr(emitter, field, invalid)

    assert str(construction_error.value) == message
    assert str(update_error.value) == message
    assert _snapshot(emitter) == before

    valid = _VALID_UPDATES[field]
    setattr(emitter, field, valid)
    assert _snapshot(emitter) == {**before, field: valid}


def test_native_smoke_emitter_accepts_equal_time_bounds() -> None:
    emitter = native.SmokeEmitter(**_EMITTER_VALUES)
    emitter.start_time = emitter.end_time
    assert _snapshot(emitter) == {**_EMITTER_VALUES, "start_time": 4.0}
    emitter.start_time = 1.0
    emitter.end_time = emitter.start_time
    assert _snapshot(emitter) == {**_EMITTER_VALUES, "end_time": 1.0}
