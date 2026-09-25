from datetime import datetime
from .viewer import ViewerHandle

def set_observation(datetime_utc: datetime, lat: float, lon: float, *, viewer: ViewerHandle | None = None) -> None: ...
