# Elevation change between two surveys

`forge3d.elevation_change` writes a registered GeoTIFF, a labelled PNG map and a CSV of signed volumes into a new directory. Install this checkout with the existing `raster` and `matplotlib` extras.

```python
from forge3d import elevation_change

product = elevation_change(
    "first.tif", "second.tif", "change-product",
    sigma1=0.12, sigma2="second_standard_error_m.tif", confidence=0.95,
    vertical_reference1="LN02", vertical_reference2="LN02",
    stable_ground="control.tif", correct_bias=False,
    error_source="Independent survey adjustment reports, standard errors in metres",
)
```

Use single-band elevation GeoTIFFs in metres, with matching dimensions, affine transform, pixel registration and horizontal CRS. Declare the vertical reference of each survey from its metadata, or use its embedded `vertical_reference` or `height_system` tag. The workflow checks declarations against both dataset and elevation-band tags. A missing affine raises a structured `PreparationError` before publication, even if the file carries a CRS. An untagged measurement band is understood to be in metres. Prepare any required reprojection, grid alignment, unit or datum conversion explicitly before comparison.

Standard errors can independently be scalars, registered GeoTIFFs in metres, or functions of one common registered slope raster. For example, pass `slope_degrees="slope.tif"` and `sigma2=my_error_model`, where the function returns one standard error in metres per slope value in degrees. The workflow decodes band scales and offsets. It intersects band masks, nodata and finite measurements across both surveys and their error inputs. A stable-ground mask identifies controls with 1 and other cells with 0. To estimate the standard error of the difference directly from the sample standard deviation on controls, supply `stable_ground` and omit both `sigma1` and `sigma2`. The normal test treats that estimated spread as a plug-in standard error.

## Reading the products

| File | Measurements |
| --- | --- |
| `change.tif` | Bands 1–4 contain Δz in metres, the uncorrected difference standard error in metres, the confidence threshold in metres and the significance class. Class −1 means significant loss, 0 means measured at or below threshold, and +1 means significant gain. NaN in all four bands marks missing coverage. |
| `map.png` | Signed significance classes, labelled grid coordinates, confidence and the threshold in metres. Spatially varying errors produce a range in the legend and an exact local threshold in raster band 3. |
| `volumes.csv` | Gain (positive), loss (negative) and net, each for all shared valid cells and for the significant subset. Each row includes cell count, signed volume and two confidence half-widths in m³. |

For independent surveys, σᵢ = hypot(σ₁ᵢ, σ₂ᵢ). The raw difference dᵢ is epoch 2 minus epoch 1. The stable-ground tags record control count n, mean difference b and sample standard deviation. With bias correction, band 1 stores dᵢ − b. Band 2 always stores raw σᵢ. Band 3 gives the effective two-sided normal threshold in metres. Cell significance uses independent-cell errors. For a corrected difference its variance is

    σᵢ² + Σcontrols σⱼ² / n² − 2 I(i is a control) σᵢ² / n.

The covariance term accounts for a control cell contributing to its own correction. Band 4 classifies a cell only where the absolute band-1 difference strictly exceeds band 3.

The area Aᵢ represents one cell footprint around the sample for both Area and Point registration. On projected grids the calculation uses the absolute affine determinant times the squared CRS conversion to metres. On geographic grids it integrates the ellipsoid’s surface area over the affine pixel in longitude and latitude, including rotated and sheared grids. On a north-up grid, the upper and lower edges follow parallels. Areas and error contributions are evaluated at each cell.

For a selected set S the volume is the sum of its area-weighted differences. Without correction the weight wᵢ is Aᵢ on S and zero elsewhere. With correction, subtract the total area of S divided by n from the weight of **every valid control**, including controls outside S. At the requested normal quantile q the volume half-widths are q √Σ(wᵢ σᵢ)² for independent cells and q |Σ wᵢ σᵢ| for perfectly positively correlated cells. A common error with equal σᵢ cancels after correction. Both calculations condition on the observed selection and the supplied or estimated standard errors.

### Reuse the classification in D02

The fourth GeoTIFF band carries the serialized D02 categorical style. Its legend includes the confidence and threshold description, signed class IDs, swatches and transparent nodata. The PNG draws missing coverage over a dark background.

```python
import json
import numpy as np
import rasterio
from forge3d.raster_style import raster_style_from_dict

with rasterio.open(product / "change.tif") as src:
    style = raster_style_from_dict(json.loads(src.tags(4)["forge3d_style"]))
    legend = style.legend()
    # Inspect the first tile with measured coverage.
    for _, window in src.block_windows(4):
        classes = src.read(4, window=window)
        if np.isfinite(classes).any():
            overlay = style.apply(classes)
            break
```

## Blatten, 19–30 May 2025

Use the two **DSM-RASTER** assets from swisstopo’s Rapid Mapping flights. The catalogue describes Leica ADS100 photogrammetric surfaces in LV95 / LN02 at 0.5 m spacing. The example verifies both published SHA-256 digests and copies the exact native cells from 30 May covering the 19 May grid. All calculations use the full-resolution bands. [19 May catalogue record](https://data.geo.admin.ch/api/stac/v1/collections/ch.swisstopo.spezialbefliegungen/items/ram-2025-05-19t12570000) · [30 May catalogue record](https://data.geo.admin.ch/api/stac/v1/collections/ch.swisstopo.spezialbefliegungen/items/ram-2025-05-30t08090000)

**Survey-error source.** Neither DSM asset record supplies a per-flight vertical standard error. For this demonstration the paired quick orthophotos identify apparently undisturbed grassland upslope of the deposit. The rectangle in LV95 extends from (2628180, 1140814) to (2628194, 1140825). Its 616 shared half-metre cells give a mean difference of −0.038817 m and a sample spread of 0.083032 m. The control-only workflow uses the measured 0.083032 m spread directly as the difference standard error across the footprint and leaves the bias uncorrected. The two surveys alone cannot identify individual flight variances. Independent checkpoints across steep rock, ice, buildings and vegetation would be needed to establish field-wide significance. [swisstopo flight imagery and product access](https://www.swisstopo.admin.ch/fr/rapidmapping-news-fr-20261021)

```shell
python examples/blatten_elevation_change.py ram-2025-05-19t12570000-dsm-raster.tif ram-2025-05-30t08090000-dsm-raster.tif blatten --confidence 0.95
```

The executed comparison covered 30,585,229 shared cells at 0.25 m² each. Under the meadow model the two-sided 95% threshold was 0.162739 m. These are surface-volume changes across the shared footprint, which includes vegetation, structures and snow.

| Cells | Gain (m³) | Loss (m³) | Net (m³) | Net ± independent (m³) | Net ± fully correlated (m³) |
| --- | ---: | ---: | ---: | ---: | ---: |
| All shared | 21,933,368 | −17,581,359 | 4,352,009 | 225 | 1,244,351 |
| Above threshold | 21,870,766 | −17,532,172 | 4,338,593 | 203 | 1,015,178 |

The CSV retains unrounded results and both half-widths for gain and loss as well as net.
