from datetime import datetime
from typing import Tuple

def body_position(
    body: str,
    datetime_utc: datetime,
    lat: float,
    lon: float,
    *,
    height_m: float = ...,
    refraction: bool = ...,
) -> Tuple[float, float, float]: ...

def moon_phase(datetime_utc: datetime) -> Tuple[float, float, float]: ...
