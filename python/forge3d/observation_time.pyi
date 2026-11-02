from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping, Sequence

@dataclass(frozen=True)
class ObservationRecord:
    record_id: str
    observed_at: datetime | str
    valid_from: datetime | str
    valid_until: datetime | str
    payload: Mapping[str, Any]
    @property
    def observation_year(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...

@dataclass(frozen=True)
class ObservationSelection:
    presentation_time: datetime
    state: Literal["observed", "held", "missing"]
    record: ObservationRecord | None
    @property
    def age_seconds(self) -> float | None: ...
    @property
    def caption(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...

@dataclass(frozen=True)
class ObservationTrack:
    records: Sequence[ObservationRecord]
    policy: Literal["hold", "no-data"]
    def select(self, presentation_times: Sequence[datetime | str]) -> list[ObservationSelection]: ...
    def selection_table(self, presentation_times: Sequence[datetime | str]) -> dict[str, Any]: ...
