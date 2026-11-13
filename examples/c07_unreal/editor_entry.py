import importlib.util
import os
from pathlib import Path
import sys
import traceback
import unreal

root = Path(os.environ["FORGE3D_C07_MRQ_ATTEMPT"]).resolve(strict=True)
script = Path(__file__).with_name("execute_unreal_mrq.py")
attempt = None
seen = set()
terminal = ("queue-finished.json", "queue-failure.json", "scene-failure.json", "bootstrap-failure.log")
unreal.EditorPythonScripting.set_keep_python_script_alive(True)

def dispatch(_):
    global attempt
    if attempt is not None and not any((attempt / name).is_file() for name in terminal):
        return
    if (root / "quit-editor").is_file():
        unreal.unregister_slate_post_tick_callback(handle)
        unreal.SystemLibrary.quit_editor()
        return
    for request in [root / "request.json", *sorted((root / "retries").glob("*/request.json"))]:
        if (request.is_file() and request.parent not in seen
                and not any((request.parent / name).is_file() for name in terminal)):
            attempt = request.parent
            seen.add(attempt)
            name = f"c07_mrq_{len(seen)}"
            spec = importlib.util.spec_from_file_location(name, script)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                (attempt / "bootstrap-failure.log").write_text(traceback.format_exc(), encoding="utf-8")
                raise
            module.main(attempt)
            break

handle = unreal.register_slate_post_tick_callback(dispatch)
(root / "ready.txt").write_text(unreal.SystemLibrary.get_engine_version(), encoding="utf-8")
