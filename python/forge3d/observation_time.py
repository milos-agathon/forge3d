"""C04 observation selection, independent of rendering and publication.

Validity intervals are half-open [valid_from, valid_until). An observation is
never available before acquisition. Overlaps select the latest acquisition;
equal acquisition times are ambiguous and fail. No interpolation occurs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Literal, Mapping, Sequence

__all__ = ["ObservationRecord", "ObservationSelection", "ObservationTrack"]


def _aware(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Observation times require an explicit timezone")
    return value


def _utc(value: datetime | str) -> datetime:
    return _aware(value).astimezone(timezone.utc)


def _stamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ObservationRecord:
    record_id: str
    observed_at: datetime | str
    valid_from: datetime | str
    valid_until: datetime | str
    payload: Mapping[str, Any]
    _payload_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise ValueError("Observation record ID must be nonempty")
        for name in ("observed_at", "valid_from", "valid_until"):
            object.__setattr__(self, name, _aware(getattr(self, name)))
        if _utc(self.valid_from) >= _utc(self.valid_until):
            raise ValueError("Observation validity must have positive duration")
        if not isinstance(self.payload, Mapping):
            raise ValueError("Observation payload must be a JSON object")
        object.__setattr__(self, "_payload_json", json.dumps(dict(self.payload), sort_keys=True, allow_nan=False))

    @property
    def observation_year(self) -> str:
        """Calendar year in the observation's declared timezone."""
        return str(self.observed_at.year)

    def to_dict(self) -> dict[str, Any]:
        return {"record_id": self.record_id, "observed_at": _stamp(_utc(self.observed_at)),
                "valid_from": _stamp(_utc(self.valid_from)), "valid_until": _stamp(_utc(self.valid_until)),
                "declared_observed_at": _stamp(self.observed_at),
                "declared_valid_from": _stamp(self.valid_from), "declared_valid_until": _stamp(self.valid_until),
                "payload": json.loads(self._payload_json)}


@dataclass(frozen=True)
class ObservationSelection:
    presentation_time: datetime
    state: Literal["observed", "held", "missing"]
    record: ObservationRecord | None

    @property
    def age_seconds(self) -> float | None:
        return None if self.record is None else (_utc(self.presentation_time) - _utc(self.record.observed_at)).total_seconds()

    @property
    def caption(self) -> str:
        prefix = f"Time {_stamp(self.presentation_time)}"
        if self.record is None:
            return f"{prefix} | No observation"
        state = "Held observation" if self.state == "held" else "Observation"
        # Preserve sub-day ages without displaying a false zero after rounding.
        age = format(self.age_seconds / 86400, ".12g")
        return f"{prefix} | {state} {_stamp(self.record.observed_at)} | Age {age} days"

    def to_dict(self) -> dict[str, Any]:
        return {"presentation_time": _stamp(_utc(self.presentation_time)),
                "declared_presentation_time": _stamp(self.presentation_time), "state": self.state,
                "record": None if self.record is None else self.record.to_dict(),
                "age_seconds": self.age_seconds, "caption": self.caption}


@dataclass(frozen=True)
class ObservationTrack:
    records: Sequence[ObservationRecord]
    policy: Literal["hold", "no-data"]

    def __post_init__(self) -> None:
        if self.policy not in {"hold", "no-data"}:
            raise ValueError("Observation policy must explicitly be hold or no-data")
        records = tuple(self.records)
        if any(not isinstance(r, ObservationRecord) for r in records):
            raise TypeError("Observation tracks require ObservationRecord values")
        if len({r.record_id for r in records}) != len(records):
            raise ValueError("Observation record IDs must be distinct")
        object.__setattr__(self, "records", records)

    def select(self, presentation_times: Sequence[datetime | str]) -> list[ObservationSelection]:
        result = []
        for value in presentation_times:
            declared_time, time = _aware(value), _utc(value)
            available = [r for r in self.records if _utc(r.observed_at) <= time and _utc(r.valid_from) <= time]
            active = [r for r in available if time < _utc(r.valid_until)]
            candidates = active or (available if self.policy == "hold" else [])
            if not candidates:
                result.append(ObservationSelection(declared_time, "missing", None))
                continue
            newest = max(_utc(r.observed_at) for r in candidates)
            selected = [r for r in candidates if _utc(r.observed_at) == newest]
            if len(selected) != 1:
                raise ValueError(f"Ambiguous observations at {_stamp(time)}: equal acquisition times")
            result.append(ObservationSelection(declared_time, "observed" if active else "held", selected[0]))
        return result

    def selection_table(self, presentation_times: Sequence[datetime | str]) -> dict[str, Any]:
        return {"policy": self.policy, "interval": "[valid_from, valid_until)",
                "overlap": "latest available acquisition; ties rejected",
                "records": [r.to_dict() for r in self.records],
                "selections": [s.to_dict() for s in self.select(presentation_times)]}
