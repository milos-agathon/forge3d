# Europe Wildfire Smoke - Plan 1 Results

- Build verdict: **PASS**
- Delivered axis: `2026-07-26` through `2026-08-04`; NOW `2026-08-04T12:00:00`; run `2026-08-04 12:00`
- CAMS steps: **63**
- Population: domain **1257393922**; display **1037658908**
- Advection gain: **0.0**
- Basemap: **4000x2250**, frame `[0.004542151162790609, 0.0, 0.9954578488372094, 1.0]`
- Measured h1: **530.0 m**; highland share **21.000%**
- Hillshade latitude-uniformity rel_diff: **0.0005**
- WebP sizes: q86 **229738 B**, q90 **301794 B**

## Gates

```json
{
  "10": {
    "covers_basemap_window": true,
    "covers_display_window": true,
    "frame_rect": [
      0.004542151162790609,
      0.0,
      0.9954578488372094,
      1.0
    ],
    "frame_rect_contained": true,
    "name": "frame_rect containment and the \u00a76.6 rail invariant",
    "out_of_data_px_at_1920": {
      "left": 455.94812164579605,
      "right": 462.2629695885508
    },
    "rail_ceiling_px_at_1920": 455.94812164579605,
    "render_aspect": 1.7777777777777777,
    "rule": "Plan 2 must satisfy rail_width < rail_ceiling_px_at_1920 * (W/1920). The ceiling is measured here, not taken from \u00a76.6's rounded 450.",
    "spec_rail_reference_px": 450.0,
    "verdict": "PASS"
  },
  "11": {
    "a_static": {
      "edges": {
        "east": {
          "delivered_margin_deg": 13.0,
          "ok": true,
          "required_deg": 12.8,
          "slack_deg": 0.1999999999999993
        },
        "north": {
          "delivered_margin_deg": 6.0,
          "ok": true,
          "required_deg": 5.8,
          "slack_deg": 0.20000000000000018
        },
        "south": {
          "delivered_margin_deg": 6.0,
          "ok": true,
          "required_deg": 5.8,
          "slack_deg": 0.20000000000000018
        },
        "west": {
          "delivered_margin_deg": 13.0,
          "ok": true,
          "required_deg": 12.8,
          "slack_deg": 0.1999999999999993
        }
      },
      "min_slack_deg": 0.1999999999999993,
      "name": "static sample containment",
      "terms": {
        "D_lat": 4.6,
        "D_lon": 11.6,
        "blur_tap": 0.8,
        "curl_clamp": 0.4
      },
      "verdict": "PASS"
    },
    "b_empirical": {
      "D": {
        "lat": 4.6,
        "lon": 11.6
      },
      "display_block": [
        106,
        175
      ],
      "display_block_expected": [
        106,
        175
      ],
      "display_block_shape": [
        106,
        175
      ],
      "frac_over_0.8D": 0.0,
      "frac_over_D": 0.0,
      "k": 0.0,
      "k_over_validated_cap": false,
      "n_steps": 63,
      "pass": true,
      "per_lag": {
        "12h": {
          "frac_over_0.8D": 0.0,
          "frac_over_D": 0.0
        },
        "6h": {
          "frac_over_0.8D": 0.0,
          "frac_over_D": 0.0
        }
      },
      "thresholds": {
        "over_0.8D": 0.02,
        "over_D": 0.005
      },
      "weighting": "mercator screen-area share over display pixels (sums to 1) x kernel weight renormalised over the engageable lags {6h, 12h} x uniform over time steps"
    },
    "name": "sample containment",
    "verdict": "PASS"
  },
  "16": {
    "advection_gain": {
      "cap": 1.5,
      "clamped": false,
      "k": 0.0,
      "k_unclamped": 0.0,
      "ok": true
    },
    "artifacts_and_credentials": {
      "findings": [
        {
          "detail": "verified 293372 B",
          "level": "ok",
          "name": "engine",
          "sha256": "20f302223ed9282965921b80a076ccff208640e8105ae8a9214bf2be816c48bc",
          "status": "verified"
        },
        {
          "detail": "verified 384285897 B",
          "level": "ok",
          "name": "ghsl",
          "sha256": "6a3dac929afc5f5f77893c81b6e2a0c4771d76e267d2a2a19ade526f0e25ddc2",
          "status": "verified"
        },
        {
          "detail": "verified 24451300 B",
          "level": "ok",
          "name": "osm_land",
          "sha256": "4ac8f10a30bcb8ee11da574b704bf37a1e37c9de85d5d2b2f9d2c34869d5272a",
          "status": "verified"
        },
        {
          "detail": "verified 4930492 B",
          "level": "ok",
          "name": "natural_earth",
          "sha256": "ce1ac7036499a0edd641fbc093cd209a98f96a49d2eca8480aaacad35138a7f6",
          "status": "verified"
        },
        {
          "detail": "live API, no local artifact; gated by cdsapirc",
          "level": "ok",
          "name": "cams",
          "sha256": null,
          "status": "remote"
        },
        {
          "detail": "live API, no local artifact; gated by firms_key",
          "level": "ok",
          "name": "firms",
          "sha256": null,
          "status": "remote"
        },
        {
          "detail": "present",
          "level": "ok",
          "name": "cdsapirc",
          "sha256": null,
          "status": "present"
        },
        {
          "detail": "present",
          "level": "ok",
          "name": "firms_key",
          "sha256": null,
          "status": "present"
        }
      ],
      "ok": true
    },
    "delivered_axis_recorded": true,
    "firms_window_recorded": true,
    "name": "provenance and build-report completeness",
    "package_versions": {
      "PIL": "12.1.1",
      "cdsapi": "unknown",
      "gdal": "3.12.1",
      "geopandas": "1.1.4",
      "netCDF4": "1.7.4",
      "numpy": "2.4.3",
      "platform": "Windows-11-10.0.26200-SP0",
      "pyproj": "3.7.2",
      "python": "3.13.14",
      "rasterio": "1.5.0",
      "scipy": "1.18.0",
      "shapely": "2.1.2",
      "xarray": "2026.7.0"
    },
    "variables_resolved": {
      "missing": [],
      "ok": true
    },
    "verdict": "PASS"
  },
  "5": {
    "checks": {
      "analysis_axis": {
        "extra": [],
        "missing": [],
        "n_delivered": 39,
        "n_requested": 39,
        "ok": true
      },
      "grid_shape": {
        "expected": [
          136,
          241
        ],
        "ok": true,
        "value": [
          136,
          241
        ]
      },
      "latitude_descends": {
        "ok": true
      },
      "lattice_residual": {
        "ok": true,
        "tolerance": 1e-09,
        "value": {
          "latitude": 4.831690603168681e-13,
          "longitude": 5.115907697472721e-13
        }
      },
      "variables_resolved": {
        "delivered": [
          "aod550",
          "bcaod550",
          "omaod550",
          "u10",
          "v10"
        ],
        "missing": [],
        "missing_optional": [],
        "missing_required": [],
        "ok": true
      }
    },
    "n_steps": 63,
    "name": "CAMS axis + lattice + grid",
    "verdict": "PASS"
  },
  "6": {
    "cadence_matched": true,
    "cadence_note": null,
    "caveat": "Measured power ~0.48 at 5% false alarm on the cached cubes; catches gross misjoins, not subtle ones. p99 of a small reference is an order statistic.",
    "n_reference": 23,
    "name": "analysis/forecast seam continuity",
    "p95": 0.003827401468455785,
    "p99": 0.004272114974227443,
    "priors": {
      "p95": 0.0313,
      "p99": 0.05
    },
    "r_seam": 0.0023301842398942606,
    "reference_source": "empirical",
    "seam_dt_h": 3.0,
    "verdict": "PASS"
  },
  "7": {
    "grid_sum": 1257393921.850506,
    "name": "mass conservation of the GHSL block-48 reduce",
    "raw_window_sum": 1257393921.850506,
    "rel_err": 0.0,
    "rtol": 1e-12,
    "verdict": "PASS"
  },
  "8": {
    "display_block_people": 1037658908,
    "domain_people": 1257393922,
    "largest_cell_people": 22693914,
    "margin_ring_people": 219735014,
    "max_abs_cell_diff_int_vs_float": 1.2993982512271032,
    "name": "population plausibility and exact closure",
    "note": "the before-fill figures are [print], not [assert]: they move by ~251 k people between GDAL/shapely builds (\u00a75.4)",
    "per_country_closes": true,
    "table_people": 1257393922,
    "table_rows": 15671,
    "unassigned_people_before_fill": 14271504.57978889,
    "unassigned_people_share_before_fill": 0.011350066460306665,
    "unassigned_pixels_after_fill": 0,
    "unassigned_pixels_before_fill": 40421542,
    "verdict": "PASS"
  },
  "9": {
    "applies_to": "the cos-lat replacement evaluated at the delivered raster's mercator bounds and pixel shape, AND the engine's shading pass ran through that same replacement on this render (basemap.assert_hillshade_patch_ran proved it), so the delivered basemap pixels were produced with the correction this gate scores.",
    "control_has_teeth": true,
    "corrected": {
      "north_std": 0.03377388045191765,
      "rel_diff": 0.003649306146508088,
      "south_std": 0.033650629222393036
    },
    "engine_patch_installed": true,
    "lat_north": 72.25223920810058,
    "lat_south": 29.53522956294847,
    "method": "synthetic constant-ground-wavelength ridge field on the delivered DEM geometry; not the Atlas-vs-Scandes terrain comparison of gate 9's prose",
    "name": "hillshade latitude uniformity after the cos-lat correction",
    "probe": {
      "amplitude_m": 155.08099027733033,
      "band_rows": 281,
      "ground_m_per_px": {
        "max": 3248.008665111274,
        "min": 1138.4366665349341
      },
      "period_rows_max": 57.06085829082184,
      "resolved": true,
      "samples_per_period_at_coarsest_row": 20.0,
      "wavelength_m": 64960.17330222548
    },
    "reason": null,
    "shape": [
      2250,
      4000
    ],
    "tolerance": 0.2,
    "uncorrected": {
      "north_std": 0.10747006821839553,
      "rel_diff": 0.26188690037715795,
      "south_std": 0.07932506516935821
    },
    "verdict": "PASS"
  }
}
```
