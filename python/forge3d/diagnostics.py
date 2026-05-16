"""Structured diagnostics for offline map-rendering validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping, Sequence


SEVERITIES = ("info", "warning", "error", "fatal")
SUPPORT_LEVELS = (
    "supported",
    "underdeveloped",
    "missing",
    "Pro-gated",
    "placeholder/fallback",
    "experimental",
    "unsupported",
    "non-goal",
)
REQUIRED_DIAGNOSTIC_CODES = frozenset(
    {
        "crs_mismatch",
        "missing_glyphs",
        "unsupported_style_field",
        "unsupported_style_layer_type",
        "pro_gated_path",
        "placeholder_fallback",
        "experimental_feature",
        "vt_unsupported_family",
        "python_public_3dtiles_incomplete",
        "estimated_gpu_memory",
        "label_rejection_summary",
    }
)

_STATUS_RANK = {"ok": 0, "info": 0, "warning": 1, "error": 2, "fatal": 3}
_DIAGNOSTIC_SORT_RANK = {"fatal": 0, "error": 1, "warning": 2, "info": 3}


class RenderFailurePolicy:
    """Render blocking policy for warning-level diagnostics."""

    CONTINUE_ON_WARNING = "continue_on_warning"
    FAIL_ON_WARNING = "fail_on_warning"

    _VALUES = (CONTINUE_ON_WARNING, FAIL_ON_WARNING)

    @classmethod
    def validate(cls, policy: str) -> str:
        if policy not in cls._VALUES:
            raise ValueError(f"Unknown render failure policy: {policy!r}")
        return policy


class SeverityPolicy:
    """Severity validation and report status behavior."""

    @staticmethod
    def validate(severity: str) -> str:
        if severity not in SEVERITIES:
            raise ValueError(f"Unknown diagnostic severity: {severity!r}")
        return severity

    @staticmethod
    def status_for(severities: Sequence[str]) -> str:
        status = "ok"
        for severity in severities:
            SeverityPolicy.validate(severity)
            if _STATUS_RANK[severity] > _STATUS_RANK[status]:
                status = severity
        return status

    @staticmethod
    def render_blocked(status: str, policy: str = RenderFailurePolicy.CONTINUE_ON_WARNING) -> bool:
        RenderFailurePolicy.validate(policy)
        if status not in ("ok", "warning", "error", "fatal"):
            raise ValueError(f"Unknown validation status: {status!r}")
        if status in ("error", "fatal"):
            return True
        return status == "warning" and policy == RenderFailurePolicy.FAIL_ON_WARNING


def _validate_support_level(support_level: str | None) -> str | None:
    if support_level is not None and support_level not in SUPPORT_LEVELS:
        raise ValueError(f"Unknown support level: {support_level!r}")
    return support_level


def _normalize_support_summary(summary: Mapping[str, str] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in sorted(dict(summary or {}).items()):
        normalized[str(key)] = _validate_support_level(str(value)) or str(value)
    return normalized


def _json_safe(value: Any, *, context: str) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key in sorted(value.keys(), key=str):
            if not isinstance(key, str):
                raise TypeError(f"{context} must use string mapping keys")
            normalized[key] = _json_safe(value[key], context=context)
        return normalized
    if isinstance(value, tuple):
        return [_json_safe(item, context=context) for item in value]
    if isinstance(value, list):
        return [_json_safe(item, context=context) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{context} must be JSON-serializable")


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass
class Diagnostic:
    code: str
    severity: str
    message: str
    remediation: str
    support_level: str | None = None
    layer_id: str | None = None
    object_id: str | None = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.severity = SeverityPolicy.validate(str(self.severity))
        self.support_level = _validate_support_level(self.support_level)
        self.details = _json_safe(dict(self.details or {}), context="details")
        _stable_json(self.details)

    def sort_key(self) -> tuple[int, str, str, str, str, str]:
        return (
            _DIAGNOSTIC_SORT_RANK[self.severity],
            self.code,
            self.layer_id or "",
            self.object_id or "",
            self.message,
            _stable_json(self.details),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "remediation": self.remediation,
            "support_level": self.support_level,
            "layer_id": self.layer_id,
            "object_id": self.object_id,
            "details": _json_safe(dict(self.details or {}), context="details"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Diagnostic":
        return cls(
            code=str(data["code"]),
            severity=str(data["severity"]),
            message=str(data["message"]),
            remediation=str(data["remediation"]),
            support_level=data.get("support_level"),
            layer_id=data.get("layer_id"),
            object_id=data.get("object_id"),
            details=data.get("details") or {},
        )


def crs_mismatch_diagnostic(
    scene_crs: str,
    layer_crs: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="crs_mismatch",
        severity="error",
        message="Layer CRS differs from the scene or terrain CRS and no transform was provided.",
        remediation="Use matching CRS metadata or provide an explicit transform before rendering.",
        support_level="unsupported",
        layer_id=layer_id,
        object_id=object_id,
        details={"layer_crs": layer_crs, "scene_crs": scene_crs},
    )


def missing_glyphs_diagnostic(
    missing_glyphs: Sequence[str],
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    glyphs = sorted(str(glyph) for glyph in missing_glyphs)
    return Diagnostic(
        code="missing_glyphs",
        severity="warning",
        message=f"{len(glyphs)} glyphs are missing from the active atlas.",
        remediation="Load an atlas with the missing glyphs or change label text before rendering.",
        support_level="underdeveloped",
        layer_id=layer_id,
        object_id=object_id,
        details={"count": len(glyphs), "missing_glyphs": glyphs},
    )


def unsupported_style_field_diagnostic(
    layer_id: str,
    fields: Sequence[str],
    *,
    section: str | None = None,
) -> Diagnostic:
    unsupported_fields = sorted(str(field) for field in fields)
    details: dict[str, Any] = {"fields": unsupported_fields}
    if section:
        details["section"] = section
    return Diagnostic(
        code="unsupported_style_field",
        severity="warning",
        message="Style layer uses paint or layout fields that forge3d does not support.",
        remediation="Remove unsupported fields or use the documented local feature styling subset.",
        support_level="unsupported",
        layer_id=layer_id,
        details=details,
    )


def unsupported_style_layer_type_diagnostic(layer_id: str, layer_type: str) -> Diagnostic:
    return Diagnostic(
        code="unsupported_style_layer_type",
        severity="error",
        message="Style layer type is not supported by forge3d offline feature styling.",
        remediation="Use a supported local feature layer type such as fill, line, or circle.",
        support_level="unsupported",
        layer_id=layer_id,
        details={"layer_type": layer_type},
    )


def pro_gated_path_diagnostic(
    feature: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="pro_gated_path",
        severity="error",
        message="Requested workflow requires a Pro-gated native path.",
        remediation="Enable the required Pro/native capability or choose a supported public path.",
        support_level="Pro-gated",
        layer_id=layer_id,
        object_id=object_id,
        details={"feature": feature},
    )


def placeholder_fallback_diagnostic(
    feature: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="placeholder_fallback",
        severity="error",
        message="Requested workflow would use placeholder or non-renderable fallback output.",
        remediation="Use a renderable supported path or keep the workflow blocked before render.",
        support_level="placeholder/fallback",
        layer_id=layer_id,
        object_id=object_id,
        details={"feature": feature},
    )


def experimental_feature_diagnostic(
    feature: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="experimental_feature",
        severity="warning",
        message="Requested workflow uses a feature that is not production-stable.",
        remediation="Treat this path as experimental or use a documented supported alternative.",
        support_level="experimental",
        layer_id=layer_id,
        object_id=object_id,
        details={"feature": feature},
    )


def vt_unsupported_family_diagnostic(
    family: str,
    *,
    supported_family: str = "albedo",
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="vt_unsupported_family",
        severity="error",
        message="Requested terrain virtual-texturing family is not paged by the native runtime.",
        remediation="Use the albedo VT family or wait for native normal/mask runtime support.",
        support_level="unsupported",
        layer_id=layer_id,
        object_id=object_id,
        details={"family": family, "supported_family": supported_family},
    )


def python_public_3dtiles_incomplete_diagnostic(
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="python_public_3dtiles_incomplete",
        severity="error",
        message="Public Python 3D Tiles workflow cannot complete the requested render path.",
        remediation="Use supported local fixtures only for validation, or wait for public MapScene integration.",
        support_level="underdeveloped",
        layer_id=layer_id,
        object_id=object_id,
    )


def estimated_gpu_memory_diagnostic(
    estimated_bytes: int,
    budget_bytes: int | None,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    exceeds_budget = budget_bytes is not None and estimated_bytes > budget_bytes
    return Diagnostic(
        code="estimated_gpu_memory",
        severity="warning" if exceeds_budget else "info",
        message=(
            "Estimated GPU memory use exceeds the configured budget."
            if exceeds_budget
            else "Estimated GPU memory use is available for review."
        ),
        remediation=(
            "Reduce layer resolution, simplify inputs, or increase the configured GPU memory budget."
            if exceeds_budget
            else "No action is required unless other diagnostics indicate risk."
        ),
        support_level="supported",
        layer_id=layer_id,
        object_id=object_id,
        details={
            "budget_bytes": budget_bytes,
            "estimated_bytes": int(estimated_bytes),
        },
    )


def label_rejection_summary_diagnostic(
    rejection_counts: Mapping[str, int],
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    counts = {str(key): int(value) for key, value in sorted(dict(rejection_counts).items())}
    total = sum(counts.values())
    return Diagnostic(
        code="label_rejection_summary",
        severity="warning",
        message=f"{total} label candidates were rejected during placement.",
        remediation="Inspect rejection reasons and adjust priorities, keepouts, glyph coverage, or geometry.",
        support_level="underdeveloped",
        layer_id=layer_id,
        object_id=object_id,
        details={"rejection_counts": counts, "total": total},
    )


def validate_label_support(
    labels: Sequence[Mapping[str, Any]],
    *,
    atlas_glyphs: set[str] | frozenset[str] | None = None,
    layer_id: str | None = None,
) -> "ValidationReport":
    """Validate label support boundaries without requiring raw viewer IPC.

    This helper does not compile or render labels. It reports PRD-scoped
    diagnostics for experimental line/curved label paths and known missing
    glyphs so callers can include those findings in validation reports before
    render.
    """
    diagnostics: list[Diagnostic] = []
    glyphs = set(atlas_glyphs) if atlas_glyphs is not None else None

    for index, label in enumerate(labels):
        object_id = str(label.get("id", f"label_{index}"))
        kind = str(label.get("kind", "point"))
        text = str(label.get("text", ""))

        if kind in {"line", "curved"}:
            diagnostics.append(
                experimental_feature_diagnostic(
                    f"{kind} labels",
                    layer_id=layer_id,
                    object_id=object_id,
                )
            )

        if glyphs is not None:
            missing = sorted({char for char in text if char not in glyphs})
            if missing:
                diagnostics.append(
                    missing_glyphs_diagnostic(
                        missing,
                        layer_id=layer_id,
                        object_id=object_id,
                    )
                )

    return ValidationReport(
        diagnostics=diagnostics,
        supported_features={"labels.point": "underdeveloped"},
        unsupported_features={
            "labels.curved.production": "experimental",
            "labels.line.production": "experimental",
        },
    )


@dataclass
class LayerSummary:
    layer_id: str
    layer_type: str
    support_level: str
    diagnostic_codes: Sequence[str] = field(default_factory=tuple)
    object_count: int | None = None
    bounds: Sequence[float] | None = None
    memory_estimate_bytes: int | None = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.support_level = _validate_support_level(self.support_level) or self.support_level
        self.diagnostic_codes = tuple(sorted(str(code) for code in self.diagnostic_codes))
        self.bounds = tuple(float(value) for value in self.bounds) if self.bounds is not None else None
        self.details = _json_safe(dict(self.details or {}), context="details")

    def sort_key(self) -> tuple[str, str, str]:
        return (self.layer_id, self.layer_type, self.support_level)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "layer_type": self.layer_type,
            "support_level": self.support_level,
            "diagnostic_codes": list(self.diagnostic_codes),
            "object_count": self.object_count,
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "memory_estimate_bytes": self.memory_estimate_bytes,
            "details": _json_safe(dict(self.details or {}), context="details"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LayerSummary":
        return cls(
            layer_id=str(data["layer_id"]),
            layer_type=str(data["layer_type"]),
            support_level=str(data["support_level"]),
            diagnostic_codes=data.get("diagnostic_codes") or (),
            object_count=data.get("object_count"),
            bounds=data.get("bounds"),
            memory_estimate_bytes=data.get("memory_estimate_bytes"),
            details=data.get("details") or {},
        )


@dataclass
class SupportMatrixEntry:
    area: str
    capability: str
    support_level: str
    scope: str
    limitations: Sequence[str] = field(default_factory=tuple)
    diagnostic_codes: Sequence[str] = field(default_factory=tuple)
    remediation: str = ""
    evidence: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.support_level = _validate_support_level(self.support_level) or self.support_level
        self.limitations = tuple(sorted(str(item) for item in self.limitations))
        self.diagnostic_codes = tuple(sorted(str(code) for code in self.diagnostic_codes))
        self.evidence = tuple(sorted(str(item) for item in self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "area": self.area,
            "capability": self.capability,
            "support_level": self.support_level,
            "scope": self.scope,
            "limitations": list(self.limitations),
            "diagnostic_codes": list(self.diagnostic_codes),
            "remediation": self.remediation,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SupportMatrixEntry":
        return cls(
            area=str(data["area"]),
            capability=str(data["capability"]),
            support_level=str(data["support_level"]),
            scope=str(data["scope"]),
            limitations=data.get("limitations") or (),
            diagnostic_codes=data.get("diagnostic_codes") or (),
            remediation=str(data.get("remediation") or ""),
            evidence=data.get("evidence") or (),
        )


@dataclass
class ValidationReport:
    diagnostics: Sequence[Diagnostic | Mapping[str, Any]] = field(default_factory=tuple)
    layer_summaries: Sequence[LayerSummary | Mapping[str, Any]] = field(default_factory=tuple)
    estimated_gpu_memory_bytes: int | None = None
    supported_features: Mapping[str, str] | None = None
    unsupported_features: Mapping[str, str] | None = None
    status: str | None = None

    def __post_init__(self) -> None:
        self.diagnostics = tuple(
            sorted(
                (
                    diag if isinstance(diag, Diagnostic) else Diagnostic.from_dict(diag)
                    for diag in self.diagnostics
                ),
                key=lambda diag: diag.sort_key(),
            )
        )
        self.layer_summaries = tuple(
            sorted(
                (
                    summary if isinstance(summary, LayerSummary) else LayerSummary.from_dict(summary)
                    for summary in self.layer_summaries
                ),
                key=lambda summary: summary.sort_key(),
            )
        )
        derived_status = SeverityPolicy.status_for([diag.severity for diag in self.diagnostics])
        if self.status is not None:
            if self.status not in ("ok", "warning", "error", "fatal"):
                raise ValueError(f"Unknown validation status: {self.status!r}")
            if _STATUS_RANK[self.status] > _STATUS_RANK[derived_status]:
                derived_status = self.status
        self.status = derived_status
        self.supported_features = _normalize_support_summary(self.supported_features)
        self.unsupported_features = _normalize_support_summary(self.unsupported_features)

    @property
    def has_errors(self) -> bool:
        return self.status in ("error", "fatal")

    def render_blocked(self, policy: str = RenderFailurePolicy.CONTINUE_ON_WARNING) -> bool:
        return SeverityPolicy.render_blocked(self.status or "ok", policy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "diagnostics": [diag.to_dict() for diag in self.diagnostics],
            "layer_summaries": [summary.to_dict() for summary in self.layer_summaries],
            "estimated_gpu_memory_bytes": self.estimated_gpu_memory_bytes,
            "supported_features": dict(self.supported_features or {}),
            "unsupported_features": dict(self.unsupported_features or {}),
            "render_blocked": self.render_blocked(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationReport":
        return cls(
            diagnostics=data.get("diagnostics") or (),
            layer_summaries=data.get("layer_summaries") or (),
            estimated_gpu_memory_bytes=data.get("estimated_gpu_memory_bytes"),
            supported_features=data.get("supported_features") or {},
            unsupported_features=data.get("unsupported_features") or {},
            status=data.get("status"),
        )


__all__ = [
    "Diagnostic",
    "LayerSummary",
    "REQUIRED_DIAGNOSTIC_CODES",
    "RenderFailurePolicy",
    "SeverityPolicy",
    "SupportMatrixEntry",
    "ValidationReport",
    "SEVERITIES",
    "SUPPORT_LEVELS",
    "crs_mismatch_diagnostic",
    "estimated_gpu_memory_diagnostic",
    "experimental_feature_diagnostic",
    "label_rejection_summary_diagnostic",
    "missing_glyphs_diagnostic",
    "placeholder_fallback_diagnostic",
    "pro_gated_path_diagnostic",
    "python_public_3dtiles_incomplete_diagnostic",
    "unsupported_style_field_diagnostic",
    "unsupported_style_layer_type_diagnostic",
    "validate_label_support",
    "vt_unsupported_family_diagnostic",
]
