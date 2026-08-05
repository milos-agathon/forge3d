"""Artifact and credential verification (gate 16), the build report, and the
safe archive extractor every network zip in this package goes through.

A clean checkout cannot build this project: the compiled engine and the bulk
input data live outside the repo. This module is the contract that makes that
honest.

Three ideas carry the verification half.

**Class.** Every manifest entry declares what it *is*, and that decides what a
problem with it means. ``required`` artifacts (engine, GHSL, OSM land) must be
on disk before anything runs. ``acquired`` artifacts (Natural Earth) are
created by the build itself, so their absence before the acquiring step is
expected, not a failure. ``remote`` entries (CAMS, FIRMS) have no local file at
all and are gated by their credential instead.

**Status, not a boolean.** ``unpinned`` is a first-class third state, distinct
from both pass and fail. An artifact with no hash in the manifest is reported
with its observed hash, counted as a warning, written into the build report,
and it fails ``--strict``. It never quietly passes.

**Level.** Exactly one function, :func:`level_of`, maps status to
``ok``/``warn``/``fail``, and every consumer -- the CLI exit code, the fetch
refusal, the report -- reads that one map. There are not three opinions about
what counts as a failure; there is one table:

===============  ===============================================  =======
status           meaning                                          level
===============  ===============================================  =======
``verified``     pinned hash matched                              ok
``present``      credential file exists and is non-empty          ok
``remote``       live API, no local artifact, credential-gated    ok
``unpinned``     on disk, manifest carries no hash                warn
``pending``      class=acquired, not created yet                  warn
``missing``      class=required, not on disk                      fail
``size``         size differs from the pin                        fail
``mismatch``     sha256 differs from the pin                      fail
``absent``       credential file missing or empty                 fail
===============  ===============================================  =======

``--strict`` promotes every ``warn`` to ``fail``; that is the only thing strict
mode does.

**Nothing here writes a hash back into manifest.toml.** :func:`propose_pins`
computes them and :func:`apply_pins` can install them, but only the explicit
``probe --write-pins`` operator command calls the latter, it refuses to
overwrite an existing pin, and the result is a diff in a tracked file that a
human reviews. A hash that pins itself is not a control.

:func:`safe_extract` lives here rather than in a new module because it is the
same concern -- input integrity at a file-format/network boundary -- and both
of its call sites (``boundary.load_countries`` and ``fetch._open``) already
import this module.
"""
from __future__ import annotations

import hashlib
import json
import ntpath
import stat
import tomllib
import zipfile
from pathlib import Path
from typing import Any, Iterable, NamedTuple

from . import config

MANIFEST_PATH = Path(__file__).with_name("manifest.toml")
_CHUNK = 1 << 20

# Nothing this build legitimately downloads is anywhere near this large; the
# cap exists so a corrupt or hostile archive cannot fill D: before we notice.
# It is honest against a lying header: CPython stops reading a member at its
# declared file_size and then fails the CRC, so under-declaring cannot smuggle
# extra bytes past this (measured -- BadZipFile, no silent oversize write).
ZIP_MAX_TOTAL_BYTES = 4 << 30

# Windows rewrites or redirects these, so two distinct members can land on one
# file without the collision check ever seeing it (measured: a member named
# ``data.nc.`` silently OVERWROTE the genuine ``data.nc``). Reject rather than
# rewrite -- the same rule as everything else here.
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

# ------------------------------------------------------------------ statuses
VERIFIED = "verified"
PRESENT = "present"
REMOTE = "remote"
UNPINNED = "unpinned"
PENDING = "pending"
MISSING = "missing"
SIZE = "size"
MISMATCH = "mismatch"
ABSENT = "absent"

OK = "ok"
WARN = "warn"
FAIL = "fail"

_LEVELS = {
    VERIFIED: OK,
    PRESENT: OK,
    REMOTE: OK,
    UNPINNED: WARN,
    PENDING: WARN,
    MISSING: FAIL,
    SIZE: FAIL,
    MISMATCH: FAIL,
    ABSENT: FAIL,
}

# ------------------------------------------------------------------ classes
REQUIRED = "required"
ACQUIRED = "acquired"
REMOTE_CLASS = "remote"
_CLASSES = {REQUIRED, ACQUIRED, REMOTE_CLASS}


class UnsafeArchiveError(RuntimeError):
    """A zip member was rejected. Nothing was written to disk."""


def level_of(status: str) -> str:
    """The single source of truth for how bad a status is."""
    try:
        return _LEVELS[status]
    except KeyError:  # pragma: no cover - only reachable via a typo in this file
        raise RuntimeError(f"unknown provenance status {status!r}") from None


class Finding(NamedTuple):
    """One artifact or credential, and what we know about it.

    ``status`` is the real answer; ``level`` is its severity; ``ok`` is the
    convenience boolean and means only "does not block the default gate".
    Reading ``.ok`` alone loses the unpinned/pending distinction, which is
    exactly the bug this shape exists to prevent -- so the printer, the report
    and ``--strict`` all read ``status``.
    """

    name: str
    status: str
    level: str
    detail: str
    sha256: str | None = None

    @property
    def ok(self) -> bool:
        return self.level != FAIL

    @property
    def blocking(self) -> bool:
        return self.level == FAIL


def _finding(name: str, status: str, detail: str, sha256: str | None = None) -> Finding:
    return Finding(name, status, level_of(status), detail, sha256)


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    # Late-bound default: tests and the sandbox rebind MANIFEST_PATH, and a
    # default evaluated at def time would silently read the real manifest.
    path = Path(path or MANIFEST_PATH)
    with open(path, "rb") as fh:
        manifest = tomllib.load(fh)
    for name, entry in manifest.items():
        if not isinstance(entry, dict):
            raise RuntimeError(f"manifest entry {name!r} is not a table")
        klass = entry.get("class")
        if klass not in _CLASSES:
            raise RuntimeError(
                f"manifest entry {name!r} declares class={klass!r}; "
                f"expected one of {sorted(_CLASSES)}"
            )
        if klass == REMOTE_CLASS and entry.get("path"):
            raise RuntimeError(f"remote entry {name!r} must not carry a path")
        if klass != REMOTE_CLASS and not entry.get("path"):
            raise RuntimeError(f"{klass} entry {name!r} must carry a path")
    return manifest


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _resolve(raw: str) -> Path:
    """Manifest path -> real path.

    Expands ``{repo}``/``{build}``/``{cache}`` against the CURRENT module
    attributes (not values bound at import), then makes a relative path
    repo-relative. Without the tokens the manifest has to repeat the build
    directory as a literal, and every artifact the build itself creates then
    reports missing whenever BUILD_DIR is overridden -- which silently turns
    gate 16 into a test of one hardcoded string.
    """
    expanded = raw.format(
        repo=Path(config.REPO_ROOT).as_posix(),
        build=Path(config.BUILD_DIR).as_posix(),
        cache=Path(config.CACHE_DIR).as_posix(),
    )
    p = Path(expanded)
    return p if p.is_absolute() else config.REPO_ROOT / p


def check_artifact(name: str, entry: dict[str, Any], *,
                   hash_unpinned: bool = True) -> Finding:
    """Verify one manifest entry. Never raises on a bad artifact."""
    klass = entry["class"]
    if klass == REMOTE_CLASS:
        cred = entry.get("credential", "?")
        return _finding(name, REMOTE, f"live API, no local artifact; gated by {cred}")

    path = _resolve(entry["path"])
    if not path.exists():
        if klass == ACQUIRED:
            by = entry.get("acquired_by", "the build")
            return _finding(name, PENDING, f"not created yet; {by} writes {path}")
        return _finding(name, MISSING, f"missing: {path}")

    size = path.stat().st_size
    expected_size = entry.get("bytes")
    if expected_size and size != expected_size:
        return _finding(name, SIZE, f"size {size} != pinned {expected_size}: {path}")

    expected_hash = (entry.get("sha256") or "").strip().lower()
    if not expected_hash:
        observed = sha256_file(path) if hash_unpinned else None
        shown = observed if observed else "not computed"
        return _finding(
            name,
            UNPINNED,
            f"on disk but manifest carries no hash; sha256={shown} "
            f"(review provenance, then `probe --write-pins`)",
            observed,
        )

    actual = sha256_file(path)
    if actual != expected_hash:
        return _finding(
            name, MISMATCH, f"sha256 {actual} != pinned {expected_hash}: {path}", actual
        )
    return _finding(name, VERIFIED, f"verified {size} B", actual)


def check_artifacts(manifest: dict[str, Any], *,
                    hash_unpinned: bool = True) -> list[Finding]:
    """Verify every manifest entry, in manifest order."""
    return [
        check_artifact(name, entry, hash_unpinned=hash_unpinned)
        for name, entry in manifest.items()
    ]


def check_credentials() -> list[Finding]:
    """Presence and non-emptiness only. Never reads a value into the report."""
    findings = []
    for name, path in (("cdsapirc", config.CDSAPIRC), ("firms_key", config.FIRMS_KEY)):
        if not path.exists():
            findings.append(_finding(name, ABSENT, f"missing: {path}"))
        elif path.stat().st_size == 0:
            findings.append(_finding(name, ABSENT, f"empty: {path}"))
        else:
            findings.append(_finding(name, PRESENT, "present"))
    return findings


def check_all(*, strict: bool = False,
              hash_unpinned: bool = True) -> tuple[bool, list[Finding]]:
    """Run every check.

    Returns ``(ok, findings)`` where ``ok`` is False if anything is at level
    ``fail``. ``strict=True`` promotes warnings (unpinned, pending) to
    failures -- the release gate, not the day-to-day one.
    """
    findings = check_artifacts(
        load_manifest(), hash_unpinned=hash_unpinned
    ) + check_credentials()
    if strict:
        findings = [f._replace(level=FAIL) if f.level == WARN else f for f in findings]
    return not any(f.level == FAIL for f in findings), findings


def summarise(findings: Iterable[Finding]) -> dict[str, Any]:
    """Counts plus the names behind each level, for the build report."""
    findings = list(findings)
    return {
        "counts": {
            lvl: sum(1 for f in findings if f.level == lvl) for lvl in (OK, WARN, FAIL)
        },
        "failing": [f.name for f in findings if f.level == FAIL],
        "warning": [f.name for f in findings if f.level == WARN],
        "unpinned": [f.name for f in findings if f.status == UNPINNED],
        "pending": [f.name for f in findings if f.status == PENDING],
        "findings": [f._asdict() for f in findings],
    }


def format_findings(findings: Iterable[Finding]) -> list[str]:
    tag = {OK: "ok  ", WARN: "warn", FAIL: "FAIL"}
    return [f"[{tag[f.level]}] {f.name}: {f.status}: {f.detail}" for f in findings]


# ------------------------------------------------------------------ pinning
def propose_pins(findings: Iterable[Finding]) -> dict[str, str]:
    """Observed hashes for everything currently unpinned. Computes nothing.

    Only ``unpinned`` entries are offered. That single restriction is what
    stops the CLI rotating a pin or repinning a ``mismatch`` to make it green;
    apply_pins() refuses too, but the CLI never gets that far.
    """
    return {f.name: f.sha256 for f in findings if f.status == UNPINNED and f.sha256}


def write_proposed_pins(pins: dict[str, str], path: Path | None = None) -> Path:
    """Drop the proposal OUTSIDE the tracked tree so it cannot be mistaken
    for the manifest, and so the operator has the value after the run ends."""
    path = Path(path or (config.BUILD_DIR / "proposed_pins.toml"))
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Observed hashes for currently-unpinned artifacts.",
             "# NOT the manifest. Verify each artifact's provenance against its",
             "# upstream before running `probe --write-pins`.", ""]
    for name, digest in sorted(pins.items()):
        lines += [f"[{name}]", f'sha256 = "{digest}"', ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def apply_pins(pins: dict[str, str], path: Path | None = None) -> list[str]:
    """Install proposed hashes into manifest.toml. Operator-invoked only.

    Refuses to overwrite an entry whose ``sha256`` is already non-empty, so a
    pin can never be silently rotated: changing one is a deliberate hand edit
    that shows up in review. Returns the names actually written.
    """
    path = Path(path or MANIFEST_PATH)
    manifest = load_manifest(path)
    text = path.read_text(encoding="utf-8")
    written: list[str] = []
    for name, digest in sorted(pins.items()):
        entry = manifest.get(name)
        if entry is None:
            raise RuntimeError(f"cannot pin unknown manifest entry {name!r}")
        if (entry.get("sha256") or "").strip():
            raise RuntimeError(
                f"{name} is already pinned; rotating a pin is a hand edit, not an "
                f"automated step"
            )
        if entry.get("class") == REMOTE_CLASS:
            raise RuntimeError(f"{name} is a remote entry and has no hash to pin")
        needle = '\nsha256 = ""'
        head, sep, tail = text.partition(f"[{name}]")
        if not sep:
            raise RuntimeError(f"could not locate [{name}] in {path}")
        nxt = tail.find("\n[")
        body, rest = (tail[:nxt], tail[nxt:]) if nxt != -1 else (tail, "")
        if needle not in body:
            raise RuntimeError(f"could not locate empty sha256 in [{name}] in {path}")
        body = body.replace(needle, f'\nsha256 = "{digest}"', 1)
        text = head + sep + body + rest
        written.append(name)
    if written:
        path.write_text(text, encoding="utf-8")
    return written


# --------------------------------------------------------------- safe unzip
#
# Both zips this package opens come off the network: Natural Earth over HTTPS
# from an S3 bucket, and the ADS ``netcdf_zip`` payload. AGENTS.md requires
# validating inputs at file-format and network boundaries, so neither call site
# may use ``ZipFile.extractall`` directly.
#
# Measured on CPython 3.13: ``extractall`` does NOT let a member escape the
# destination -- ``_extract_member`` strips the drive, drops every ``..`` and
# ``.`` component and discards a leading separator. But it does that SILENTLY
# and LOSSILY, which is its own defect for us:
#
#   archive member               lands as
#   ../../pwned.txt           -> pwned.txt
#   C:/Windows/Temp/p.txt     -> Windows/Temp/p.txt
#   /abs.txt                  -> abs.txt
#   ../data.nc + good/data.nc -> data.nc AND good/data.nc  (two members, one of
#                                which is now a top-level file the archive never
#                                named that way)
#
# ``fetch._open`` then picks NetCDFs out of that directory, so a mangled member
# is silently merged into the CAMS cube. Rejecting is the only honest
# behaviour: an archive whose member names need rewriting is not the archive we
# asked for. That principle has to extend past the member string to the names
# the FILESYSTEM rewrites -- a trailing dot or space, and the reserved device
# names -- because those collide after validation, where nothing is watching.
#
# ``ZipFile.extractall`` has no ``filter=`` parameter -- that is ``tarfile``
# only (3.12+, ``filter='data'``) -- so there is nothing to reuse and this
# helper must exist.


def _reject(zip_path: Path, name: str, why: str) -> None:
    raise UnsafeArchiveError(f"{zip_path}: refusing member {name!r}: {why}")


def _member_mode(info: zipfile.ZipInfo) -> int:
    """Unix mode bits, or 0 when the archive was not written by a unix host."""
    if info.create_system != 3:  # 3 == unix; DOS/NT archives carry no mode
        return 0
    return (info.external_attr >> 16) & 0xFFFF


def _validate_member(zip_path: Path, dest: Path, info: zipfile.ZipInfo) -> Path:
    zip_path = Path(zip_path)
    dest = Path(dest)
    name = info.filename
    if not name or not name.strip():
        _reject(zip_path, name, "empty name")
    if "\x00" in name:
        # POSIX-only in practice: _sanitize_filename truncates at the first NUL
        # on read. Kept so the rule is stated where it belongs.
        _reject(zip_path, name, "NUL in name")
    if "\\" in name:
        # PKZIP APPNOTE 4.4.17.1: the path separator is '/' only. A name
        # carrying a backslash is either malformed or a traversal aimed at
        # Windows extractors. Reachable on POSIX only: ZipInfo.__init__ runs
        # _sanitize_filename, which rewrites os.sep -> '/' on read, so on
        # Windows a backslash member arrives here already forward-slashed and
        # is caught by the '..' check below instead (measured).
        _reject(zip_path, name, "backslash in name; zip paths use '/' only")
    if ntpath.splitdrive(name)[0]:
        _reject(zip_path, name, "drive letter or UNC prefix")
    if name.startswith("/"):
        _reject(zip_path, name, "absolute path")
    if ":" in name:
        # zipfile rewrites ':' to '_' on Windows -- silent mangling again.
        _reject(zip_path, name, "':' is not portable and would be rewritten")

    # Split by hand. PurePosixPath(name).parts silently DROPS '.' components
    # ('./x' -> ('x',)) -- that is the same normalisation this helper exists to
    # refuse, and it made an earlier version of this check pass './sneaky.txt'.
    parts = name.split("/")
    if parts and parts[-1] == "":
        parts = parts[:-1]              # a trailing '/' marks a directory member
    if not parts:
        _reject(zip_path, name, "no usable path components")
    for part in parts:
        if part in ("", ".", ".."):
            _reject(zip_path, name, f"empty, '.' or '..' path component: {part!r}")
        if part != part.rstrip(" ."):
            # Windows strips trailing dots and spaces from a path component
            # before it ever reaches the filesystem, so 'data.nc.' and
            # 'data.nc' are the SAME file there while comparing unequal here.
            # Measured: with only the checks above, the trailing-dot member was
            # ACCEPTED and its bytes overwrote the genuine 'data.nc', and
            # fetch._open then read the attacker's payload as the CAMS cube.
            _reject(zip_path, name,
                    f"component {part!r} ends in a space or dot; Windows strips "
                    "those, so two members would silently land on one file")
        if part.split(".")[0].lower() in _WINDOWS_RESERVED:
            # 'NUL' resolves to the null device in ANY directory: the write is
            # silently discarded and safe_extract would return a path that does
            # not exist (measured).
            _reject(zip_path, name,
                    f"component {part!r} is a Windows reserved device name")

    mode = _member_mode(info)
    if stat.S_ISLNK(mode):
        # CPython does not create a symlink here -- it writes the link target as
        # the contents of a regular file, so the member silently becomes bogus
        # data. And an extractor that DID honour it could redirect a later
        # member's write outside dest. Neither payload ships symlinks.
        _reject(zip_path, name, "symlink member")
    if mode and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        _reject(zip_path, name, f"not a regular file or directory (mode {mode:#o})")

    target = dest.joinpath(*parts)
    # Belt and braces: whatever the component checks missed, the resolved path
    # still has to sit strictly inside dest.
    root = dest.resolve()
    resolved = target.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        _reject(zip_path, name, f"resolves outside the destination: {resolved}")
    if resolved == zip_path.resolve():
        # boundary.load_countries extracts into zip_path.parent, so a member
        # named after the archive lands ON the archive. Measured: the download
        # was silently replaced by the member's contents, and load_countries
        # caches on "does the zip exist", so the substitution would then be
        # reused on every later run.
        _reject(zip_path, name, "would overwrite the archive being read")
    return target


def safe_extract(
    zip_path: Path | str,
    dest: Path | str,
    *,
    max_total_bytes: int | None = ZIP_MAX_TOTAL_BYTES,
) -> list[Path]:
    """Extract ``zip_path`` into ``dest``, rejecting anything unsafe.

    All-or-nothing: every member is validated before a single byte is written,
    and anything the validator could not foresee (a bad CRC, a filesystem
    refusal) rolls back what this call wrote. Returns the files written, in
    archive order.

    Rejects absolute members, drive-letter and UNC members, ``..``/``.``
    components, backslashes, symlink and device members, names Windows would
    silently rewrite (a trailing dot or space) or redirect (``NUL``, ``COM1``),
    two members that would land on one path, a member that another member needs
    to be a directory, a member that would overwrite the archive itself, and an
    archive whose uncompressed total exceeds ``max_total_bytes``.
    """
    zip_path = Path(zip_path)
    dest = Path(dest)
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        targets: list[Path] = []
        seen: dict[str, str] = {}
        total = 0
        for info in infos:
            target = _validate_member(zip_path, dest, info)
            # NTFS is case-insensitive, so a case-only difference is still a
            # collision here. Rejecting it on a case-sensitive filesystem too
            # costs us nothing -- neither payload relies on it.
            key = str(target).casefold()
            if key in seen and not info.is_dir():
                _reject(zip_path, info.filename,
                        f"collides with earlier member {seen[key]!r} at {target}")
            seen[key] = info.filename
            total += int(info.file_size)
            if max_total_bytes is not None and total > max_total_bytes:
                _reject(zip_path, info.filename,
                        f"uncompressed total exceeds {max_total_bytes} B")
            targets.append(target)

        # A member that is a FILE cannot also be another member's parent
        # directory. Nothing above catches that -- the two paths differ -- so
        # the clash surfaced only at write time, as a bare FileExistsError in
        # one member order and a PermissionError in the other, half way
        # through, leaving a partial tree for fetch._open to read (measured).
        # Decide it here, before anything is written, and as an
        # UnsafeArchiveError like every other refusal.
        file_owner: dict[str, str] = {}
        dir_owner: dict[str, str] = {}
        for info, target in zip(infos, targets):
            if info.is_dir():
                dir_owner.setdefault(str(target).casefold(), info.filename)
            else:
                file_owner[str(target).casefold()] = info.filename
            for ancestor in list(target.relative_to(dest).parents)[:-1]:
                dir_owner.setdefault(str(dest / ancestor).casefold(), info.filename)
        clash = sorted(set(file_owner) & set(dir_owner))
        if clash:
            _reject(zip_path, file_owner[clash[0]],
                    f"is a file, but member {dir_owner[clash[0]]!r} needs that "
                    "same path to be a directory")

        dest.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        try:
            for info, target in zip(infos, targets):
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # Recorded BEFORE the write, so a half-written file is rolled
                # back too.
                written.append(target)
                with zf.open(info) as src, open(target, "wb") as out:
                    while chunk := src.read(_CHUNK):
                        out.write(chunk)
        except BaseException:
            # All-or-nothing has to survive failures the validator cannot
            # predict -- a CRC error on a corrupt payload, a filesystem
            # refusal. The caller's next move is to read whatever NetCDFs it
            # finds in this directory, so half an archive is worse than none.
            for path in reversed(written):
                try:
                    path.unlink()
                except OSError:
                    pass
            raise
    return written


class BuildReport:
    """Accumulates what the build actually did, for §10.2 provenance."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def update(self, **kwargs: Any) -> None:
        self._data.update(kwargs)

    @property
    def data(self) -> dict[str, Any]:
        return dict(self._data)

    def write(self, path: Path | None = None) -> Path:
        path = Path(path or config.REPORT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")
        return path
