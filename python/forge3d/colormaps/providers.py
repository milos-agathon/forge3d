from __future__ import annotations
from typing import Callable
import importlib
import numpy as np
from .core import Colormap

def _mpl_to_colormap(mpl_cmap, name: str, n: int = 256) -> Colormap:
    xs = np.linspace(0, 1, n, dtype=np.float32)
    rgba_srgb = np.array(mpl_cmap(xs), dtype=np.float32)  # (n,4) in sRGB
    # convert to linear sRGB:
    a = 0.055
    rgb = rgba_srgb[:, :3]
    rgb_lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)
    rgba_lin = np.concatenate([rgb_lin, rgba_srgb[:, 3:4]], axis=1).astype(np.float32)
    return Colormap(name, rgba_lin)

def load_provider(provider: str) -> Callable[[str], Colormap]:
    p = provider.lower()
    if p == "cmcrameri":
        def f(name: str) -> Colormap:
            cmc = importlib.import_module("cmcrameri.cm")
            mpl_cmap = getattr(cmc, name)
            return _mpl_to_colormap(mpl_cmap, f"cmcrameri:{name}")
        return f
    if p == "cmocean":
        def f(name: str) -> Colormap:
            cmo = importlib.import_module("cmocean.cm")
            mpl_cmap = getattr(cmo, name)
            return _mpl_to_colormap(mpl_cmap, f"cmocean:{name}")
        return f
    if p == "colorcet":
        def f(name: str) -> Colormap:
            cc = importlib.import_module("colorcet")
            mpl_cmap = cc.cm[name]
            return _mpl_to_colormap(mpl_cmap, f"colorcet:{name}")
        return f
    if p == "palettable":
        def f(name: str) -> Colormap:
            # name like "colorbrewer.sequential.YlGn_9"
            mod = importlib.import_module("palettable." + ".".join(name.split(".")[:-1]))
            obj = getattr(mod, name.split(".")[-1])
            return _mpl_to_colormap(obj.mpl_colormap, f"palettable:{name}")
        return f
    if p == "pycpt":
        def f(path_or_name: str) -> Colormap:
            # path or name resolved by pycpt loaders
            m = importlib.import_module("pycpt")
            cmap = m.load(path_or_name)
            # pycpt returns a Matplotlib cmap
            return _mpl_to_colormap(cmap, f"pycpt:{path_or_name}")
        return f
    raise KeyError(f"Unknown provider: {provider}")
