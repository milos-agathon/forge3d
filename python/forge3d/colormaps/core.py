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
