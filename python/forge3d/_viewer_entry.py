from __future__ import annotations

import sys

from ._native import get_native_module as _get_native_module


def main() -> int:
    """Console-script entrypoint for the interactive viewer."""
    native = _get_native_module()
    if native is None or not hasattr(native, "run_interactive_viewer_cli"):
        print(
            "forge3d native viewer entrypoint is unavailable. "
            "Reinstall forge3d so the native extension is built correctly.",
            file=sys.stderr,
        )
        return 1

    try:
        native.run_interactive_viewer_cli(sys.argv[1:])
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
