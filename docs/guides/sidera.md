# SIDERA observation sky

`forge3d.astro` returns offline topocentric positions for the Sun, Moon, and
Mercury through Saturn. Pass a timezone-aware `datetime`; positions outside UTC
2000–2050 raise `ValueError`. Azimuth and altitude are degrees and the returned
distance is kilometres. The altitude is airless; the Rust core also exposes a
separate standard-atmosphere Sæmundsson refraction reduction.

```python
from datetime import datetime, timezone
import forge3d

when = datetime(2026, 9, 25, 22, tzinfo=timezone.utc)
azimuth, altitude, distance_km = forge3d.astro.body_position(
    "Moon", when, 52.37, 4.9
)
fraction = forge3d.astro.moon_phase(when)
local_fraction = forge3d.astro.moon_phase(when, 52.37, 4.9)

with forge3d.open_viewer_async() as viewer:
    forge3d.sky.set_observation(when, 52.37, 4.9, viewer=viewer)
```

`set_observation` also targets the most recently opened asynchronous viewer when
`viewer=` is omitted. The viewer computes the actual Sun, Moon, five planets,
and J2000 Yale bright stars in one command. Stars use V magnitudes and B−V
color approximation; the lunar disc has topocentric size and Sun-facing bright
limb. The civil-to-astronomical twilight fade is a visual ramp from solar
depression 6° to 18°. Moonlight uses the Krisciunas–Schaefer (1991) phase
magnitude fit, representative V-band extinction, and an explicit display gain.
These night lighting choices are rendering models, not light-pollution or
sky-glow predictions.

The catalog display mapping uses [Ballesteros' B−V temperature relation](https://arxiv.org/abs/1201.1809)
and relative V-band flux. The Moon lighting phase fit is from the
[Krisciunas–Schaefer paper](https://articles.adsabs.harvard.edu/pdf/1991PASP..103.1033K).

Data sources, hashes, coefficient counts, observed error maxima, and
regeneration commands are in [`data/sidera/MANIFEST.md`](../../data/sidera/MANIFEST.md).
The independent JPL Horizons oracle covers 40 UTC/site combinations across
2000–2050. The fixed night image and its model certificate live in
`tests/golden/sidera/`.
