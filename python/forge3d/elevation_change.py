"""Registered elevation differencing and conditional normal-model volume bounds."""
from __future__ import annotations

import csv
import json
import warnings
from contextlib import ExitStack
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import cache
from math import erf, exp, hypot, pi, sqrt, ulp
from numbers import Real
from pathlib import Path
from statistics import NormalDist
from tempfile import TemporaryDirectory
from typing import Callable, Union

import numpy as np

from .alignment import _require_rasterio
from .preparation import PreparationError, PreparationGrid
from .raster_style import CategoricalRasterStyle
from .raster_registration import geotiff_registration, require_compatible_registration

__all__ = ["elevation_change"]
PathLike = Union[str, Path]
ErrorInput = Union[float, PathLike, Callable[[np.ndarray], np.ndarray]]


def _grid(ds):
    if ds.count != 1 or ds.crs is None or ds.dtypes[0].startswith("complex"):
        raise PreparationError("invalid_raster", f"{ds.name} requires one real band and a horizontal CRS",
                               "Supply a single real elevation or ancillary band with a declared horizontal CRS.")
    if ds.gcps[0] or ds.rpcs:
        raise PreparationError("unsupported_georeferencing", f"{ds.name} requires a registered affine grid",
                               "Prepare an affine grid from the source georeferencing.")
    try:
        registration = geotiff_registration(ds)
    except ValueError as exc:
        raise PreparationError("invalid_registration", str(exc), "Verify AREA_OR_POINT.", ds.name) from exc
    return PreparationGrid(str(ds.crs), ds.width, ds.height, ds.transform.to_gdal(),
                           registration)


def _height_reference(ds, declaration):
    declarations = [declaration]
    for tags in (ds.tags(), ds.tags(1)):
        declarations.extend(tags.get(key) for key in ("vertical_reference", "height_system"))
    references = [ref.strip().casefold() for ref in declarations if ref is not None]
    if not references or any(ref in {"", "unknown", "unspecified"} for ref in references):
        raise PreparationError("missing_vertical_reference", f"{ds.name} requires a known vertical reference",
                               "Declare the measured height datum for each survey.")
    if len(set(references)) != 1:
        raise PreparationError("conflicting_vertical_reference", f"{ds.name} has conflicting vertical references",
                               "Reconcile caller, dataset and band height datums.")
    return references[0]


def _check_units(ds, permitted):
    for unit in (ds.units[0], ds.tags().get("units"), ds.tags(1).get("units")):
        if unit and unit.lower().strip() not in permitted:
            raise PreparationError("conflicting_units", f"{ds.name} has incompatible measurement units {unit!r}",
                                   "Supply measurements in the documented units.")


def _read(ds, window):
    raw = ds.read(1, window=window)
    valid = ds.read_masks(1, window=window) != 0
    if ds.nodata is not None:
        valid &= raw != ds.nodata
    values = raw.astype("float64")
    values[~valid] = np.nan
    values *= ds.scales[0]
    values += ds.offsets[0]
    values[~np.isfinite(values)] = np.nan
    return values


def _errors(source, slope, window):
    if isinstance(source, Real):
        return source
    if callable(source):
        values = np.ma.asarray(source(slope), dtype="float64").filled(np.nan)
        if values.shape != slope.shape:
            raise PreparationError("invalid_error_model", "slope error function must return one standard error per cell",
                                   "Return an array matching the slope grid.")
        values = np.where(np.isfinite(slope), values, np.nan)
    else:
        values = _read(source, window)
    if np.any(values < 0) or np.isinf(values).any():
        raise PreparationError("invalid_standard_error", "standard errors must be finite and nonnegative on valid cells",
                               "Supply nonnegative one-sigma errors in metres.")
    return values


@dataclass
class _Stable:
    count: int = 0
    mean: float = 0.0
    squared_deviations: float = 0.0

    def add(self, values):
        if values.size:
            offset = values.mean() - self.mean
            total = self.count + values.size
            self.squared_deviations += (np.sum((values - values.mean()) ** 2)
                                        + offset ** 2 * self.count * values.size / total)
            self.mean += offset * values.size / total
            self.count = total


def _cell_areas(ds):
    """Return square-metre weights for a window on the original sample grid."""
    t = ds.transform
    if ds.crs.is_projected:
        area = abs(t.determinant) * ds.crs.linear_units_factor[1] ** 2
        if not np.isfinite(area) or area <= 0:
            raise PreparationError("invalid_cell_area", "Invalid projected cell area.",
                                   "Verify the affine transform and the horizontal CRS units.")
        return lambda w: np.broadcast_to(area, (w.height, w.width)), {"cell_area_m2": area}
    if not ds.crs.is_geographic:
        raise PreparationError("unsupported_grid_crs", "Unknown horizontal CRS geometry.",
                               "Supply a geographic or projected horizontal CRS.")
    from pyproj import CRS

    crs = CRS.from_wkt(ds.crs.to_wkt())
    geod = crs.get_geod()
    axes = {axis.direction: axis.unit_conversion_factor for axis in crs.axis_info}
    lat_unit = axes.get("north") or axes["south"]
    lon_unit = axes.get("east") or axes["west"]
    pole = (np.pi / 2) / lat_unit
    # Evaluate the four corners exactly from the stored binary affine. A pixel
    # step computed from bounds can itself have up to half an ulp of rounding.
    origin, col_step, row_step = map(Fraction, (t.f, t.d, t.e))
    col_error = Fraction(ulp(t.d)) / 2
    row_error = Fraction(ulp(t.e)) / 2
    for col in (0, ds.width):
        for row in (0, ds.height):
            latitude = origin + col * col_step + row * row_step
            excess = abs(latitude) - Fraction(pole)
            if excess > col * col_error + row * row_error:
                raise PreparationError("invalid_cell_area", "Affine footprint extends beyond a pole.",
                                       "Verify the angular coordinates and corner locations.")
    # M(phi) N(phi) cos(phi) dphi dlambda, pulled back to the unit pixel.
    scale = abs(t.determinant) * lon_unit * lat_unit * geod.a ** 2 * (1 - geod.es)

    @cache
    def samples(order, latitude_step):
        if latitude_step == 0:
            return [(0., 1.)]
        nodes, weights = np.polynomial.legendre.leggauss(order)
        return list(zip((nodes + 1) * (.5 * latitude_step * lat_unit), weights * .5))

    def geographic(window):
        cols = np.arange(window.col_off, window.col_off + window.width) if t.d else np.array([0])
        rows = np.arange(window.row_off, window.row_off + window.height) if t.e else np.array([0])
        offsets = t.e * rows[:, None] + t.d * cols
        # Work relative to the nearer pole to retain precision in small polar cells.
        nearest_pole = np.copysign(pole, t.f + offsets)
        origin = ((t.f - nearest_pole) + offsets) * lat_unit
        order = 1
        previous = None
        while True:
            col_samples, row_samples = samples(order, t.d), samples(order, t.e)
            density = np.zeros_like(origin)
            for dy_col, wx in col_samples:
                for dy_row, wy in row_samples:
                    distance = np.abs(origin + dy_col + dy_row)
                    density += wx * wy * np.sin(distance) / (1 - geod.es * np.cos(distance) ** 2) ** 2
            area = scale * density
            if not np.all(np.isfinite(area) & (area > 0)):
                raise PreparationError("invalid_cell_area", "Undefined geographic cell footprint.",
                                       "Verify the affine transform and CRS ellipsoid.")
            # Refine until the change is within the rounding error of the weighted sum.
            terms = len(col_samples) * len(row_samples)
            if previous is not None and np.all(np.abs(area - previous) <= terms * np.spacing(area)):
                return np.broadcast_to(area, (window.height, window.width))
            previous = area
            order *= 2

    return geographic, {"cell_area_method": "ellipsoid surface integral over affine pixel"}


@dataclass
class _Volume:
    scope: str
    quantity: str
    cells: int = 0
    volume: float = 0.0
    area: float = 0.0
    independent_sd: float = 0.0
    common_error: float = 0.0

    def selection(self, delta, valid, significant):
        selected = valid if self.scope == "all" else significant
        if self.quantity == "gain":
            return selected & (delta > 0)
        if self.quantity == "loss":
            return selected & (delta < 0)
        return selected

    def measure(self, delta, area):
        self.cells += delta.size
        self.area += area.sum()
        self.volume += np.sum(area * delta)

    def propagate(self, weights, sigma, common_reference=0.0):
        self.independent_sd = hypot(self.independent_sd, np.linalg.norm(weights * sigma))
        self.common_error += np.sum(weights * (sigma - common_reference))

    def record(self, confidence, q):
        return dict(scope=self.scope, quantity=self.quantity, confidence=confidence,
                    cells=self.cells, volume_m3=self.volume,
                    independent_half_width_m3=q * self.independent_sd,
                    correlated_half_width_m3=q * abs(self.common_error))


def _central_quantile(confidence):
    q = -NormalDist().inv_cdf((1 - confidence) / 2)
    if confidence < .5:
        # Newton refinement in central-probability coordinates avoids subtracting
        # a small confidence from one. The linear term suffices at machine epsilon.
        if confidence ** 2 < np.finfo(float).eps:
            return confidence * sqrt(pi / 2)
        q += (confidence - erf(q / sqrt(2))) * sqrt(pi / 2) * exp(q * q / 2)
    return q


def _write_map(path, ds, title, style, error_source):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib.transforms import Affine2D

    figure = Figure(figsize=(9, 8), dpi=150, constrained_layout=True)
    ax, key = figure.subplots(2, 1, gridspec_kw={"height_ratios": [6, 1.5]})
    legend = style.legend()
    entries = legend["entries"]
    colors = [np.asarray(entry["rgba"]) / 255 for entry in entries]
    swatches = [Patch(color=c, label=e["label"]) for c, e in zip(colors, entries)]
    missing_background = "#454953"
    swatches.append(Patch(color=missing_background, label="Missing coverage"))
    cmap = ListedColormap(colors)
    cmap.set_bad(np.asarray(legend["nodata_rgba"]) / 255)
    ax.set_facecolor(missing_background)
    # Limit display sampling to the printed figure resolution; totals use every native cell.
    scale = min(1, figure.bbox.width / ds.width, figure.bbox.height / ds.height)
    shape = (max(1, round(ds.height * scale)), max(1, round(ds.width * scale)))
    classes = ds.read(4, out_shape=shape)
    t = ds.transform
    transform = Affine2D.from_values(t.a, t.d, t.b, t.e, t.c, t.f)
    ax.imshow(classes, origin="upper", extent=(0, ds.width, ds.height, 0),
              interpolation="nearest", cmap=cmap,
              norm=BoundaryNorm([-1.5, -.5, .5, 1.5], 3),
              transform=transform + ax.transData)
    corners = [t * (c, r) for c in (0, ds.width) for r in (0, ds.height)]
    xs, ys = zip(*corners)
    if ds.crs.is_geographic:
        unit = ds.crs.units_factor[0]
        xlabel, ylabel = f"Longitude ({unit})", f"Latitude ({unit})"
    else:
        xlabel, ylabel = f"Easting ({ds.crs.linear_units})", f"Northing ({ds.crs.linear_units})"
    ax.set(xlim=(min(xs), max(xs)), ylim=(min(ys), max(ys)), title=title,
           xlabel=xlabel, ylabel=ylabel)
    ax.ticklabel_format(style="plain", useOffset=False)
    key.axis("off")
    key.legend(handles=swatches,
               loc="upper center", ncol=2, frameon=False)
    key.text(.5, .4, legend["caption"], ha="center", va="top", transform=key.transAxes)
    key.text(.5, .05, error_source, ha="center", va="top", wrap=True, fontsize=8,
             transform=key.transAxes)
    figure.savefig(path, bbox_inches="tight", metadata={"Description": legend["caption"] + "\n" + error_source})


def elevation_change(
    survey1: PathLike, survey2: PathLike, output_dir: PathLike, *,
    confidence: float, sigma1: ErrorInput | None = None, sigma2: ErrorInput | None = None,
    vertical_reference1: str | None = None, vertical_reference2: str | None = None,
    stable_ground: PathLike | None = None, correct_bias: bool = False,
    slope_degrees: PathLike | None = None, error_source: str | None = None,
    title: str = "Elevation change",
) -> Path:
    """Write change.tif, map.png and volumes.csv in a new directory and return its path.

    Supply single-band GeoTIFFs in metres on the same registered grid. Each error
    may be a scalar, a registered raster in metres, or a function of a supplied
    slope raster in degrees. Declare each height datum when absent from file tags.
    The optional stable-ground raster codes 1 for control and 0 for other cells.
    Omit both errors to estimate the difference spread from stable controls.
    Volume half-widths condition on observed membership and propagate the control mean.
    """
    if not 0 < confidence < 1:
        raise PreparationError("invalid_confidence", "confidence must be strictly between 0 and 1",
                               "Supply a probability strictly between zero and one.")
    if correct_bias and stable_ground is None:
        raise PreparationError("missing_stable_ground", "bias correction requires stable ground",
                               "Provide the stable-ground control mask.")
    if (callable(sigma1) or callable(sigma2)) != (slope_degrees is not None):
        raise PreparationError("missing_slope", "slope error functions require a matching slope_degrees raster",
                               "Supply a slope raster when using a slope error function.")
    empirical = sigma1 is None and sigma2 is None
    if (sigma1 is None) != (sigma2 is None) or (empirical and stable_ground is None):
        raise PreparationError("missing_survey_error", "Incomplete survey error model.",
                               "Supply both survey errors, or omit both and supply stable ground.")
    if error_source is None:
        error_source = "Stable-ground sample standard deviation" if empirical else "Caller-supplied standard errors"
    multiplier = _central_quantile(confidence)
    rio = _require_rasterio()
    dest = Path(output_dir)
    if dest.exists():
        raise FileExistsError(dest)
    metres = {"m", "metre", "metres", "meter", "meters"}
    with ExitStack() as stack:
        def opened(path):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", rio.errors.NotGeoreferencedWarning)
                    ds = stack.enter_context(rio.open(path))
                    ds.read_transform()  # Require GDAL's actual affine, including a stored identity.
                    return ds
            except rio.errors.NotGeoreferencedWarning as exc:
                raise PreparationError("missing_geotransform", "Missing affine geotransform.",
                                       "Supply the verified source transform.", str(path)) from exc
            except rio.errors.RasterioIOError as exc:
                raise PreparationError("raster_read_failed", str(exc), "Verify the source GeoTIFF.",
                                       str(path)) from exc

        first = opened(survey1)
        grid = _grid(first)

        def aligned(path):
            ds = opened(path)
            candidate = _grid(ds)
            try:
                require_compatible_registration(grid.registration, candidate.registration)
            except ValueError as exc:
                raise PreparationError("registration_mismatch", str(exc),
                                       "Prepare matching pixel registrations.", str(path)) from exc
            if ds.crs != first.crs:
                raise PreparationError("conflicting_crs", f"CRS mismatch for {path}",
                                       "Prepare both sources in the same horizontal CRS.")
            if not grid.same_as(candidate):
                raise PreparationError("conflicting_grid", f"grid mismatch for {path}",
                                       "Supply identical shapes and affine transforms.")
            return ds

        second = aligned(survey2)
        reference = _height_reference(first, vertical_reference1)
        if reference != _height_reference(second, vertical_reference2):
            raise PreparationError("conflicting_vertical_reference", "survey vertical references differ",
                                   "Convert the surveys to a common vertical datum before comparison.")
        cell_areas, area_tags = _cell_areas(first)
        for ds in (first, second):
            _check_units(ds, metres)

        def error_input(source):
            if isinstance(source, Real):
                if not np.isfinite(source) or source < 0:
                    raise PreparationError("invalid_standard_error", "scalar standard errors must be finite and nonnegative",
                                           "Supply nonnegative one-sigma errors in metres.")
                return float(source)
            if callable(source):
                return source
            ds = aligned(source)
            _check_units(ds, metres)
            return ds

        errors = () if empirical else (error_input(sigma1), error_input(sigma2))
        slope = aligned(slope_degrees) if slope_degrees is not None else None
        if slope is not None:
            _check_units(slope, {"degree", "degrees", "deg"})
        control = aligned(stable_ground) if stable_ground is not None else None
        dest.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(stack.enter_context(TemporaryDirectory(dir=dest.parent))) / "product"
        stage.mkdir()
        profile = dict(driver="GTiff", dtype="float64", count=4, width=grid.width,
                       height=grid.height, crs=first.crs, transform=first.transform,
                       tiled=True, compress="deflate", predictor=3, nodata=np.nan, BIGTIFF="IF_SAFER")
        stable = _Stable()
        control_variance_sum = 0.0
        control_reference = None
        rows = [_Volume(scope, quantity) for scope in ("all", "significant")
                for quantity in ("gain", "loss", "net")]
        minimum, maximum = np.inf, -np.inf
        with rio.open(stage / "change.tif", "w+", **profile) as ds:
            for _, window in ds.block_windows(1):
                angles = _read(slope, window) if slope is not None else None
                if angles is not None and np.any((angles < 0) | (angles > 90)):
                    raise PreparationError("invalid_slope", "slope must be between 0 and 90 degrees",
                                           "Verify the slope angles in degrees.")
                delta = _read(second, window) - _read(first, window)
                valid = np.isfinite(delta)
                if not empirical:
                    sigma = np.broadcast_to(np.hypot(*(_errors(s, angles, window) for s in errors)),
                                            delta.shape)
                    valid &= np.isfinite(sigma)
                    ds.write(np.where(valid, sigma, np.nan), 2, window=window)
                delta[~valid] = np.nan
                ds.write(delta, 1, window=window)
                if control is not None:
                    mask = _read(control, window)
                    if np.any(np.isfinite(mask) & (mask != 0) & (mask != 1)):
                        raise PreparationError("invalid_stable_ground", "stable-ground raster must contain 0 or 1",
                                               "Use class one for stable controls and zero elsewhere.")
                    controls = valid & (mask == 1)
                    stable.add(delta[controls])
                    if correct_bias and not empirical and controls.any():
                        raw_errors = sigma[controls]
                        control_variance_sum += np.sum(raw_errors ** 2)
                        if control_reference is None:
                            control_reference = raw_errors[0]
            offset = 0.0
            if control is not None:
                if stable.count < 2:
                    raise PreparationError("insufficient_controls", "sample spread requires two or more shared stable-ground cells",
                                           "Supply at least two controls in shared valid coverage.")
                spread = sqrt(stable.squared_deviations / (stable.count - 1))
                if empirical:
                    control_variance_sum = stable.count * spread ** 2
                    control_reference = spread
                ds.update_tags(stable_cells=stable.count, stable_bias_m=stable.mean,
                               stable_sample_sd_m=spread)
                if correct_bias:
                    offset = stable.mean
            for _, window in ds.block_windows(1):
                delta = ds.read(1, window=window)
                valid = np.isfinite(delta)
                if empirical:
                    sigma = np.where(valid, spread, np.nan)
                    ds.write(sigma, 2, window=window)
                else:
                    sigma = ds.read(2, window=window)
                test_sd = sigma
                if correct_bias:
                    delta -= offset
                    ds.write(delta, 1, window=window)
                    controls = _read(control, window) == 1
                    # Var(d_i - mean_C) includes covariance when i belongs to C.
                    test_sd = np.sqrt(sigma ** 2 * (1 - 2 * controls / stable.count)
                                      + control_variance_sum / stable.count ** 2)
                threshold = test_sd * multiplier
                significant = valid & (np.abs(delta) > threshold)
                classes = np.where(valid, np.where(significant, np.sign(delta), 0), np.nan)
                ds.write(threshold, 3, window=window)
                ds.write(classes, 4, window=window)
                if valid.any():
                    minimum = min(minimum, np.nanmin(threshold))
                    maximum = max(maximum, np.nanmax(threshold))
                    area = cell_areas(window)
                    for row in rows:
                        selected = row.selection(delta, valid, significant)
                        row.measure(delta[selected], area[selected])
                        if not correct_bias:
                            row.propagate(area[selected], sigma[selected])
            if correct_bias:
                # The total selected area is now known, including selections in other tiles.
                for _, window in ds.block_windows(1):
                    delta, sigma, classes = ds.read((1, 2, 4), window=window)
                    valid = np.isfinite(delta)
                    controls = (_read(control, window) == 1) & valid
                    area = cell_areas(window)
                    for row in rows:
                        selected = row.selection(delta, valid, np.abs(classes) == 1)
                        weights = area * selected - controls * (row.area / stable.count)
                        # Corrected weights sum to zero. Centering preserves exact
                        # common-mode cancellation even for nonbinary standard errors.
                        row.propagate(weights[valid], sigma[valid], control_reference)
            if minimum == np.inf:
                raise PreparationError("empty_coverage", "no shared valid survey and error coverage",
                                       "Check the coverage and validity masks of both surveys and their errors.")
            threshold_text = (f"{minimum:.6g} m" if minimum == maximum else
                              f"{minimum:.6g} to {maximum:.6g} m across cells (band 3)")
            percent = Decimal(str(confidence)).scaleb(2).normalize()
            percent_text = str(percent) if percent < 1 else format(percent, "f")
            uncertainty = "corrected σ (independent cells)" if correct_bias else "σΔ"
            legend = (f"{percent_text}% confidence, two-sided normal test\n"
                      f"Significant where |Δz| > {multiplier:.6g} × {uncertainty}  ·  threshold {threshold_text}")
            style = CategoricalRasterStyle(
                palette={-1: (187, 88, 51, 255), 0: (232, 230, 222, 255), 1: (36, 114, 154, 255)},
                labels={-1: "Significant loss", 0: "Below or at threshold", 1: "Significant gain"},
                units="m", caption=legend,
            )
            ds.update_tags(4, forge3d_style=json.dumps(style.to_dict(), ensure_ascii=False))
            ds.descriptions = ("delta_z_m", "sigma_delta_m", "threshold_m", "significance_class")
            ds.units = ("m", "m", "m", "1")
            ds.update_tags(AREA_OR_POINT=grid.registration.title(), vertical_reference=reference,
                           confidence=confidence, normal_multiplier=multiplier, applied_bias_m=offset,
                           bias_corrected=str(correct_bias), error_source=error_source, **area_tags,
                           error_model="stable_difference_sd" if empirical else "independent_surveys",
                           classes="-1 loss, 0 at or below threshold, +1 gain, nodata missing")
            _write_map(stage / "map.png", ds, title, style, error_source)
        with (stage / "volumes.csv").open("w", encoding="utf-8", newline="") as stream:
            records = [row.record(confidence, multiplier) for row in rows]
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        stage.rename(dest)
    return dest
