from .core import Colormap
from .registry import register, get, available, to_linear_rgba_u8
from .core_palettes import OKABE_ITO  # triggers core registration on import
from .io import load_json, load_cpt
__all__ = ["Colormap", "register", "get", "available", "to_linear_rgba_u8", "OKABE_ITO", "load_json", "load_cpt"]
