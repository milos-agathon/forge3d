"""The release audit, as a test.

The goal prompt asks for four checks before every release. Two of them are
mechanical and belong in CI rather than in a reviewer's memory:

* **every `RenderSettings` field reaches the payload** — a field nobody reads
  is dead configuration, and the rule is to delete it, not to document it;
* **the payload and the service agree in both directions** — a key the shell
  sends that the service ignores is a control that does nothing, and a key the
  service reads that the shell never sends is a control that cannot work;
* **the project model carries no parameter the engine ignores** — the
  registry below is what was *measured* at 0 changed samples on the path
  Studio renders through (the roadmap has the numbers).

The ranges audit (every UI control inside the engine's range) and the
"no second implementation" audit stay manual: they compare against the
engine's own contracts, which a text scan cannot see.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "studio/crates/studio-core/src/project.rs"
RENDER = ROOT / "studio/src-tauri/src/render.rs"
COMMANDS = ROOT / "studio/src-tauri/src/commands.rs"
SERVICE = ROOT / "studio/src-tauri/src/studio_render.py"

#: Engine parameters measured to change **0** channel samples through Studio's
#: render path. They are engine features, not shell bugs — the shell must not
#: offer them until the engine's path can draw them.
# The clouds left this registry when the engine's quality surface reached the
# studio's still: they measured 0 changed samples while the clouds were
# multiplied into the stylized lighting model only, and 368,996 (mean 0.498)
# after the physical model consumed the cloud visibility too.
INERT_ENGINE_PARAMETERS = {
    "ambient_occlusion",
    "shadow_softness",
    "shadow_intensity",
    "cascades",
    "tonemap",
    "tonemap_operator",
    "white_balance",
    "ssao",
    "ssgi",
    "ssr",
    "reflection",
    "planar_reflection",
    "material_layers",
    "snow_layer",
    "rock_layer",
    "wetness_layer",
}


def _render_settings_fields() -> list[str]:
    text = CORE.read_text()
    start = text.index("pub struct RenderSettings {")
    end = text.index("\n}", start)
    return re.findall(r"\n    pub ([a-z0-9_]+):", text[start:end])


def _quality_payload_keys() -> tuple[set[str], set[str]]:
    """The render request's `quality` keys: the scalars and the nested blocks.

    The nested blocks (water, fog, bloom, sunVisibility, aov) hold their own
    keys, so they are returned separately: a block is a key of `quality`, and
    its leaves are not.
    """
    text = RENDER.read_text()
    start = text.index('"quality": {')
    depth = 0
    end = start
    for index, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    block = text[start:end]
    # The top level of `quality` is indented by 16 spaces; one level deeper
    # (inside a block) by 20.
    scalars = set(re.findall(r'\n\s{16}"([A-Za-z0-9_]+)":', block))
    blocks = set(re.findall(r'\n\s{16}"([A-Za-z0-9_]+)": \{', block))
    return scalars - blocks, blocks


def _service_quality_reads() -> set[str]:
    """The keys the service reads straight out of the request's `quality`."""
    text = SERVICE.read_text()
    reads = set(re.findall(r'quality\.get\(\s*["\']([A-Za-z0-9_]+)["\']', text))
    reads |= set(
        re.findall(r'\("quality"\) or \{\}\)\.get\(\s*["\']([A-Za-z0-9_]+)["\']', text)
    )
    return reads


def test_every_render_setting_reaches_the_payload() -> None:
    fields = _render_settings_fields()
    assert len(fields) > 20, f"the model looks empty: {fields}"
    sources = RENDER.read_text() + COMMANDS.read_text()
    missing = [
        field
        for field in fields
        if not re.search(rf"\.{re.escape(field)}\b", sources)
    ]
    assert not missing, (
        "these RenderSettings fields are never read by the shell, so they are "
        f"dead configuration: {missing}"
    )


def test_the_payload_and_the_service_agree_in_both_directions() -> None:
    payload, blocks = _quality_payload_keys()
    reads = _service_quality_reads()
    missing = sorted((payload | blocks) - reads)
    assert not missing, (
        "the payload sends these keys and the service never reads them, so they "
        f"are controls that do nothing: {missing}"
    )
    unsent = sorted(reads - payload - blocks)
    assert not unsent, (
        "the service reads these keys and the payload never sends them, so the "
        f"controls cannot work: {unsent}"
    )


def test_the_model_carries_no_parameter_the_engine_ignores() -> None:
    fields = set(_render_settings_fields())
    offenders = sorted(field for field in fields if field in INERT_ENGINE_PARAMETERS)
    assert not offenders, (
        "these settings were measured at 0 changed samples on Studio's render "
        f"path, so they must not be exposed: {offenders}"
    )


# --------------------------------------------------------------- dead code
#
# A helper nobody calls is not a helper: it is a promise the code does not
# keep, and it is how an increment grows a second implementation of something
# that already exists. The scan is text-level (the shell is TypeScript,
# Python and Rust, not a compiled library), so it looks for a *reference*
# rather than proving a call.

SHELL = ROOT / "studio"


def _shell_sources() -> list[Path]:
    """The studio's sources plus the tests that drive them."""
    sources = [
        path
        for folder in ("src", "src-tauri/src", "crates")
        for path in (SHELL / folder).rglob("*")
        if path.suffix in {".ts", ".tsx", ".py", ".rs"}
    ]
    sources += [
        path
        for path in (ROOT / "tests").glob("test_studio_*.py")
        if path.is_file()
    ]
    return sorted(sources)


def _definitions(path: Path, text: str) -> list[tuple[str, int]]:
    """The helper names a file declares, with the line each is declared on."""
    if path.suffix in {".ts", ".tsx"}:
        pattern = r"^export (?:async )?(?:function|const) ([A-Za-z_][A-Za-z0-9_]*)"
    elif path.suffix == ".py":
        # The service's public helpers (a leading underscore means it is
        # internal to the module).
        pattern = r"^def ([a-z][A-Za-z0-9_]*)\("
    else:
        pattern = r"^\s*pub fn ([a-z_][A-Za-z0-9_]*)"
    found = []
    for index, line in enumerate(text.splitlines()):
        match = re.match(pattern, line)
        if match:
            name = match.group(1)
            if path.suffix == ".py" and name.startswith("_"):
                continue
            found.append((name, index))
    return found


def test_no_dead_helpers_in_the_shell() -> None:
    """Every helper the studio declares has a caller.

    A declaration that nothing else mentions is dead code: the rule is to
    delete it, not to document it. The scan is text-level, so it errs on the
    side of listing — a name that appears anywhere else (a call, a test, a
    doc) counts as alive.
    """
    sources = _shell_sources()
    texts = {path: path.read_text(encoding="utf-8") for path in sources}
    dead: list[str] = []

    for path, text in texts.items():
        if path.parent == ROOT / "tests":
            # The tests are the callers, not the scanned code (pytest calls
            # them by name, so they have no textual reference).
            continue
        for name, index in _definitions(path, text):
            pattern = re.compile(rf"\b{re.escape(name)}\b")
            mentions = 0
            for other, other_text in texts.items():
                for number, line in enumerate(other_text.splitlines()):
                    if other == path and number == index:
                        # The declaration itself.
                        continue
                    if pattern.search(line):
                        mentions += 1
                        break
            if mentions == 0:
                dead.append(f"{path.relative_to(ROOT)}:{index + 1}: {name}")

    assert not dead, (
        "these helpers have no caller anywhere, so they are dead code: "
        f"{sorted(dead)}"
    )
