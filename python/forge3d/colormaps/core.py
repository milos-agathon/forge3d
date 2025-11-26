from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# --- Color space helpers (sRGB <-> linear) ---
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)

def _hex_to_rgba_linear(hex_s: str, alpha: float = 1.0) -> np.ndarray:
    hex_s = hex_s.lstrip("#")
    r = int(hex_s[0:2], 16) / 255.0
    g = int(hex_s[2:4], 16) / 255.0
    b = int(hex_s[4:6], 16) / 255.0
    rgb_lin = _srgb_to_linear(np.array([r, g, b], dtype=np.float32))
    return np.array([*rgb_lin, alpha], dtype=np.float32)

def _interp_stops(stops, n: int) -> np.ndarray:
    # stops: list[(pos0..1, "#RRGGBB" or (r,g,b[,a]))]
    xs = np.array([s[0] for s in stops], dtype=np.float32)
    cols = []
    for s in stops:
        c = s[1]
        if isinstance(c, str):
            cols.append(_hex_to_rgba_linear(c))
        else:
            c = np.array(c, dtype=np.float32)
            if c.size == 3:
                c = np.concatenate([c, [1.0]], axis=0)
            cols.append(_srgb_to_linear(c[:3]).tolist() + [float(c[3])])
    cols = np.array(cols, dtype=np.float32)  # (k,4)
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    rgba = np.column_stack([np.interp(x, xs, cols[:, i]) for i in range(4)]).astype(np.float32)
    return np.clip(rgba, 0.0, 1.0)

@dataclass(frozen=True)
class Colormap:
    name: str
    rgba: np.ndarray           # shape (N,4), linear sRGB, float32
    under: tuple | None = None # linear sRGB
    over: tuple | None = None
    bad: tuple | None = (0.0, 0.0, 0.0, 0.0)

    def reversed(self) -> "Colormap":
        return Colormap(self.name + "_r", self.rgba[::-1].copy(), self.under, self.over, self.bad)

    def with_endcaps(self, under=None, over=None, bad=None) -> "Colormap":
        return Colormap(self.name, self.rgba, under if under is not None else self.under,
                        over if over is not None else self.over, bad if bad is not None else self.bad)

def from_stops(name: str, stops, n: int = 256) -> Colormap:
    return Colormap(name=name, rgba=_interp_stops(stops, n=n))


def interpolate_hex_colors(colors, size: int):
    """Interpolate a list of hex colors to a target length using linear RGB."""

    if size <= 0 or len(colors) < 2:
        return list(colors)

    rgb_colors = []
    for hex_color in colors:
        h = str(hex_color).lstrip("#")
        if len(h) != 6:
            continue
        try:
            r = int(h[0:2], 16) / 255.0
            g = int(h[2:4], 16) / 255.0
            b = int(h[4:6], 16) / 255.0
        except ValueError:
            continue
        rgb_colors.append((float(r), float(g), float(b)))

    if len(rgb_colors) < 2:
        return list(colors)

    indices = np.linspace(0, len(rgb_colors) - 1, int(size))
    interpolated = []
    for idx in indices:
        i0 = int(np.floor(idx))
        i1 = min(i0 + 1, len(rgb_colors) - 1)
        t = float(idx - i0)

        r = rgb_colors[i0][0] * (1.0 - t) + rgb_colors[i1][0] * t
        g = rgb_colors[i0][1] * (1.0 - t) + rgb_colors[i1][1] * t
        b = rgb_colors[i0][2] * (1.0 - t) + rgb_colors[i1][2] * t

        interpolated.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")

    return interpolated


def elevation_stops_from_hex_colors(domain, colors, *, heightmap=None, q_lo: float = 0.0, q_hi: float = 1.0):
    """Map a sequence of hex colors across a DEM elevation domain."""

    colors = [str(c) for c in colors]
    if not colors:
        return []

    # Optional quantile placement when a heightmap is available.
    if heightmap is not None:
        data = np.asarray(heightmap, dtype=np.float32).reshape(-1)
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            qs = np.linspace(q_lo, q_hi, len(colors))
            elevs = np.quantile(finite, qs)
            return [(float(e), c) for e, c in zip(elevs, colors)]

    # Fallback: evenly spread colors across the numeric domain.
    domain_min, domain_max = float(domain[0]), float(domain[1])
    domain_range = domain_max - domain_min
    n = len(colors)
    if n == 1 or domain_range <= 0.0:
        return [(domain_min, colors[0])]

    stops = []
    for i, c in enumerate(colors):
        t = i / (n - 1)
        elevation = domain_min + t * domain_range
        stops.append((elevation, c))

    return stops
