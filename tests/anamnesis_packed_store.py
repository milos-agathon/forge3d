"""Independent raw-file operations for packed-store corruption/residency tests."""
import json
import os
from pathlib import Path

DATA = "data.v2.bin"
INDEX = "index.v2.json"


def _mark_external_edit(path, previous_mtime):
    # Fast raw-file mutations can share an mtime on some filesystems. Make
    # the external edit visible to the store's file-identity check so the
    # corruption tests exercise fresh on-disk bytes, not a resident snapshot.
    current = path.stat()
    if current.st_mtime_ns == previous_mtime:
        os.utime(path, ns=(current.st_atime_ns, previous_mtime + 1_000_000_000))


def index(root):
    return json.loads((Path(root) / INDEX).read_text(encoding="utf-8"))


def metadata(root, key):
    return index(root)["entries"][key]["metadata"]


def write_metadata(root, key, value):
    value_index = index(root)
    value_index["entries"][key]["metadata"] = value
    path = Path(root) / INDEX
    previous_mtime = path.stat().st_mtime_ns
    path.write_text(json.dumps(value_index, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    _mark_external_edit(path, previous_mtime)


def read_blob(root, key):
    record = index(root)["entries"][key]
    with (Path(root) / DATA).open("rb") as stream:
        stream.seek(record["offset"])
        return stream.read(record["length"])


def write_blob(root, key, value):
    record = index(root)["entries"][key]
    path = Path(root) / DATA
    previous_mtime = path.stat().st_mtime_ns
    with path.open("r+b") as stream:
        stream.seek(record["offset"])
        stream.write(value)
    _mark_external_edit(path, previous_mtime)


def remove_entry(root, key):
    value = index(root)
    value["entries"].pop(key, None)
    (Path(root) / INDEX).write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def resident(root, key):
    try:
        record = index(root)["entries"][key]
        return (type(record["offset"]) is int and type(record["length"]) is int
                and 0 <= record["offset"] <= record["offset"] + record["length"]
                <= (Path(root) / DATA).stat().st_size)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def metadata_entries(root):
    return {key: record["metadata"] for key, record in index(root)["entries"].items()}
