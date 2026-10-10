"""Ordered, deterministic composition of existing RGBA render outputs.

This is a Python image compositor, not a native scene renderer. Inputs are
snapshots of caller-supplied images; no source image is inferred or rendered.
"""

from __future__ import annotations

import json
import math
import hashlib
import io
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

def _name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("pass and input names must be nonempty strings without surrounding whitespace")
    return value


def _space(value: str) -> None:
    if value not in ("srgb", "linear"):
        raise ValueError("color_space must be 'srgb' or 'linear'")


@dataclass(frozen=True)
class RenderPassInput:
    """An immutable RGBA snapshot: uint8 [0,255] or float [0,1].

    Premultiplied RGB is multiplied by alpha in the declared color space.
    Fully transparent RGB is discarded when the input is used.
    """

    data: Any = field(repr=False, compare=False)
    color_space: str = "srgb"
    alpha_mode: str = "straight"
    _binary: bytes = field(init=False, repr=False)
    _sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        import numpy as np

        _space(self.color_space)
        if self.alpha_mode not in ("straight", "premultiplied"):
            raise ValueError("alpha_mode must be 'straight' or 'premultiplied'")
        raw = np.asarray(self.data)
        if raw.ndim != 3 or raw.shape[2] != 4 or min(raw.shape[:2]) <= 0:
            raise ValueError("pass input must have nonempty dimensions (height, width, 4)")
        if raw.dtype == np.uint8:
            rgba = raw.astype(np.float64) / 255.0
        elif raw.dtype.kind == "f":
            rgba = raw.astype(np.float64)
        else:
            raise ValueError("pass input must be uint8 or floating-point RGBA")
        if not np.isfinite(rgba).all() or (rgba < 0).any() or (rgba > 1).any():
            raise ValueError("pass input channels must be finite and in [0,1]")
        if self.alpha_mode == "premultiplied" and (rgba[..., :3] > rgba[..., 3:4]).any():
            raise ValueError("premultiplied RGB must not exceed alpha")
        rgba[rgba == 0] = 0.0  # Canonical positive zero, including decode.
        # Pin the binary encoding: NumPy v1.0, C order, little-endian float64.
        stream = io.BytesIO()
        np.lib.format.write_array(stream, np.ascontiguousarray(rgba, dtype='<f8'),
                                  version=(1, 0), allow_pickle=False)
        binary = stream.getvalue()
        # Array and serialized snapshot share immutable bytes, without a second copy.
        snapshot = np.frombuffer(binary, dtype='<f8', offset=len(binary) - rgba.nbytes).reshape(rgba.shape)
        object.__setattr__(self, "data", snapshot)
        object.__setattr__(self, "_binary", binary)
        object.__setattr__(self, "_sha256", hashlib.sha256(binary).hexdigest())

    def _asset_descriptor(self) -> dict[str, Any]:
        return {"color_space": self.color_space, "alpha_mode": self.alpha_mode,
                "asset": f"scene/render_pass_inputs/{self._sha256}.npy",
                "sha256": self._sha256, "dtype": "<f8", "shape": list(self.data.shape)}

    def to_dict(self) -> dict[str, Any]:
        return {"color_space": self.color_space, "alpha_mode": self.alpha_mode,
                "rgba": self.data.tolist()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RenderPassInput:
        import numpy as np

        return cls(np.asarray(data["rgba"], dtype=np.float64),
                   color_space=data["color_space"], alpha_mode=data["alpha_mode"])


@dataclass(frozen=True)
class RenderPassSpec:
    """Named color/relief/overlay output in explicit execution order.

    ``inputs`` is (source,) for alpha-over on transparent black, or
    (backdrop, source) for blending. References name an image input or an
    earlier pass. ``parameters`` accepts only source ``opacity`` in [0,1].
    Blending uses ``color_space``; all outputs use straight alpha.
    """

    name: str
    kind: str
    inputs: Sequence[str]
    operation: str = "alpha_over"
    parameters: Mapping[str, float] = field(default_factory=dict)
    color_space: str = "linear"

    def __post_init__(self) -> None:
        _name(self.name)
        if self.kind not in ("color", "relief", "overlay"):
            raise ValueError("pass kind must be color, relief or overlay")
        if self.operation not in ("multiply", "screen", "alpha_over"):
            raise ValueError("unsupported pass operation")
        _space(self.color_space)
        if isinstance(self.inputs, (str, bytes)):
            raise ValueError("pass inputs must be an ordered sequence of names")
        inputs = tuple(_name(value) for value in self.inputs)
        if len(inputs) != 2 and not (len(inputs) == 1 and self.operation == "alpha_over"):
            raise ValueError("pass requires (backdrop, source), or (source,) for alpha_over")
        parameters = dict(self.parameters)
        if set(parameters) - {"opacity"}:
            raise ValueError("unsupported pass parameter; only opacity is supported")
        opacity = parameters.get("opacity", 1.0)
        if isinstance(opacity, bool) or not isinstance(opacity, (float, int)):
            raise ValueError("opacity must be a finite number in [0,1]")
        if not math.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError("opacity must be a finite number in [0,1]")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "parameters", MappingProxyType({"opacity": float(opacity) or 0.0}))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "inputs": list(self.inputs),
                "operation": self.operation, "parameters": dict(self.parameters),
                "color_space": self.color_space}

    def __hash__(self) -> int:
        return hash((self.name, self.kind, self.inputs, self.operation,
                     tuple(self.parameters.items()), self.color_space))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RenderPassSpec:
        return cls(name=data["name"], kind=data["kind"], inputs=data["inputs"],
                   operation=data["operation"], parameters=data.get("parameters") or {},
                   color_space=data["color_space"])


def compile_passes(passes: Sequence[RenderPassSpec], inputs: Mapping[str, RenderPassInput],
                   shape: tuple[int, int], *, include_data: bool = True) -> dict[str, Any]:
    """Validate the entire graph and dimensions before any composition."""
    available = set()
    for name, image in inputs.items():
        _name(name)
        if not isinstance(image, RenderPassInput):
            raise TypeError("named pass inputs must be RenderPassInput snapshots")
        if image.data.shape != (*shape, 4):
            raise ValueError(f"pass input {name!r} dimensions must match output {shape}")
        available.add(name)
    names = set(available)
    for spec in passes:
        if not isinstance(spec, RenderPassSpec):
            raise TypeError("passes must contain RenderPassSpec values")
        if spec.name in names:
            raise ValueError(f"duplicate pass or input name: {spec.name!r}")
        names.add(spec.name)
        for reference in spec.inputs:
            if reference not in available:
                raise ValueError(f"pass {spec.name!r} has missing or out-of-order reference {reference!r}")
        available.add(spec.name)
    if inputs and not passes:
        raise ValueError("pass inputs require at least one pass")
    if not passes:
        return {}
    # Only the final pass is exported. Reject branches and snapshots that cannot
    # contribute to it, rather than silently accepting a disconnected graph.
    reachable = {passes[-1].name}
    for spec in reversed(passes):
        if spec.name in reachable:
            reachable.update(spec.inputs)
    unused = names - reachable
    if unused:
        raise ValueError(f"unused pass or input names: {sorted(unused)!r}")
    # Inputs and parameters have already been normalized at their immutable
    # boundaries; avoid traversing every scalar again during compilation.
    return {"passes": [spec.to_dict() for spec in passes],
            "inputs": {name: image.to_dict() if include_data else image._asset_descriptor()
                       for name, image in inputs.items()}}


def decode_passes(payload: Mapping[str, Any], snapshots: Mapping[str, RenderPassInput] | None = None
                  ) -> tuple[tuple[RenderPassSpec, ...], dict[str, RenderPassInput]]:
    inputs = {}
    for name, image in payload.get("inputs", {}).items():
        if "rgba" in image:
            inputs[name] = RenderPassInput.from_dict(image)
        else:
            snapshot = (snapshots or {}).get(name)
            if snapshot is None or snapshot._asset_descriptor() != image:
                raise ValueError(f"missing or mismatched render pass snapshot: {name!r}")
            inputs[name] = snapshot
    return tuple(RenderPassSpec.from_dict(spec) for spec in payload.get("passes", ())), inputs


def _convert(rgb: Any, source: str, target: str) -> Any:
    import numpy as np

    if source == target:
        return rgb
    if target == "linear":
        return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055)


def execute_passes(compiled_json: str, shape: tuple[int, int],
                   snapshots: Mapping[str, RenderPassInput] | None = None) -> Any:
    """Replay a compiled specification; return straight-alpha sRGB floats."""
    import numpy as np

    passes, inputs = decode_passes(json.loads(compiled_json), snapshots)
    compile_passes(passes, inputs, shape, include_data=False)
    images = {}
    for name, image in inputs.items():
        rgba = image.data.copy()
        alpha = rgba[..., 3:4]
        if image.alpha_mode == "premultiplied":
            rgba[..., :3] = np.divide(rgba[..., :3], alpha, out=np.zeros_like(rgba[..., :3]), where=alpha > 0)
        rgba[..., :3] = _convert(np.where(alpha > 0, rgba[..., :3], 0), image.color_space, "linear")
        images[name] = rgba
    for spec in passes:
        bottom = images[spec.inputs[0]] if len(spec.inputs) == 2 else np.zeros((*shape, 4))
        top = images[spec.inputs[-1]]
        cb = _convert(bottom[..., :3], "linear", spec.color_space)
        cs = _convert(top[..., :3], "linear", spec.color_space)
        ab = bottom[..., 3:4]
        a = top[..., 3:4] * spec.parameters["opacity"]
        if spec.operation == "multiply":
            blended = cb * cs
        elif spec.operation == "screen":
            blended = cb + cs - cb * cs
        else:
            blended = cs
        alpha = a + ab * (1 - a)
        # Porter-Duff source-over with blending only in the overlap region.
        premultiplied = a * (1 - ab) * cs + a * ab * blended + (1 - a) * ab * cb
        rgb = np.divide(premultiplied, alpha, out=np.zeros_like(premultiplied), where=alpha > 0)
        images[spec.name] = np.concatenate((_convert(rgb, spec.color_space, "linear"), alpha), axis=2)
    result = images[passes[-1].name].copy()
    result[..., :3] = _convert(result[..., :3], "linear", "srgb")
    return np.clip(result, 0, 1)


__all__ = ["RenderPassInput", "RenderPassSpec"]
