"""Report unsupported cache requests without changing render return types."""
import warnings


def report_cache_bypass(cache, entrypoint, reason="render_cache_unsupported"):
    if cache is not None:
        warnings.warn(
            f"{entrypoint}: cache bypassed ({reason}); rendering fresh",
            RuntimeWarning,
            stacklevel=3,
        )
