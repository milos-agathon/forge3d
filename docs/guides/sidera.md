# SIDERA observation sky

`forge3d.astro` returns offline topocentric positions for the Sun, Moon, and
Mercury through Saturn. Pass a timezone-aware `datetime`; positions outside UTC
2000–2050 raise `ValueError`. Azimuth and altitude are degrees and the returned
distance is kilometres. `body_position` returns airless altitude for comparison
with the Horizons oracle. `body_position_refracted` returns standard-atmosphere
apparent altitude using the Sæmundsson refraction reduction.
Azimuth is clockwise from north (0° north, 90° east). In the live viewer,
the right-handed terrain frame is +x east, +y up, +z south, so northward
celestial and light directions from `set_observation` point toward −z.
`SetSunDirection` keeps its legacy convention (azimuth 0 maps to +z); SIDERA
converts its azimuths internally before driving the terrain sun.

```python
from datetime import datetime, timezone
import forge3d

when = datetime(2026, 9, 25, 22, tzinfo=timezone.utc)
azimuth, altitude, distance_km = forge3d.astro.body_position(
    "Moon", when, 52.37, 4.9
)
azimuth, apparent_altitude, distance_km = forge3d.astro.body_position_refracted(
    "Moon", when, 52.37, 4.9
)
fraction = forge3d.astro.moon_phase(when)
local_fraction = forge3d.astro.moon_phase(when, 52.37, 4.9)

with forge3d.open_viewer_async() as viewer:
    forge3d.sky.set_observation(when, 52.37, 4.9, viewer=viewer)
```

`set_observation` also targets the most recently opened asynchronous viewer when
`viewer=` is omitted. The viewer computes the actual Sun, Moon, five planets,
and J2000 Yale bright stars in one command. Celestial sprites and the live
terrain light use standard-atmosphere apparent directions; the sky model uses
the airless Sun direction to select day, night, and twilight. Stars use V
magnitudes and B−V colour
approximation; the lunar disc has topocentric size, a Sun-facing bright limb,
and a packed texture derived from NASA SVS's LRO nearside mosaic. The IAU WGCCRE
lunar north pole rotates the surface image with observer position and camera
orientation; the nearside markings remain fixed, without a libration model.
The civil-to-astronomical twilight fade is a smoothstep visual ramp from solar
depression 6° to 18°, shared by the sky and celestial sprites. Moonlight uses
the Krisciunas–Schaefer (1991) phase-magnitude and air-mass fits,
representative V-band extinction, and an explicit relative display gain.
These night lighting choices are rendering models, not light-pollution or
sky-glow predictions.

The catalog display mapping uses [Ballesteros' B−V temperature relation](https://arxiv.org/abs/1201.1809)
and relative V-band flux: `10^(-0.4 V)` uses the V=0 reference (about 3630 Jy
in the [UKIRT calibration table](https://about.ifa.hawaii.edu/ukirt/calibration-and-standards/astronomical-utilities/zero-mag-fluxes-and-conversions/)).
Missing catalogue B−V values use a declared display default of 0.65. The
renderer is not an absolute radiometric calibration. The Moon lighting fit is from the
[Krisciunas–Schaefer paper](https://articles.adsabs.harvard.edu/pdf/1991PASP..103.1033K).

Data sources, hashes, coefficient counts, observed error maxima, and
regeneration commands are in [`data/sidera/MANIFEST.md`](../../data/sidera/MANIFEST.md).
The independent JPL Horizons oracle covers 40 UTC/site combinations across
2000–2050. The fixed night image and its model certificate live in
`tests/golden/sidera/`.
