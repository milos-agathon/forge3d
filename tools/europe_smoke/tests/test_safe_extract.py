"""Every zip this package opens arrives over the network. None of them may go
through ZipFile.extractall (AGENTS.md: validate inputs at file-format and
network boundaries)."""
import stat
import zipfile

import pytest

from tools.europe_smoke import provenance


def _zip(path, members, *, symlinks=()):
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in members:
            info = zipfile.ZipInfo(name)
            info.filename = name          # bypass ZipInfo's separator rewrite
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            zf.writestr(info, payload)
        for name, target in symlinks:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, target)
    return path


def _dest(tmp_path):
    d = tmp_path / "dest"
    d.mkdir()
    return d


def test_extracts_a_benign_archive(tmp_path):
    z = _zip(tmp_path / "ok.zip", [("data.nc", "netcdf"), ("sub/more.nc", "more")])
    written = provenance.safe_extract(z, _dest(tmp_path))
    assert [p.relative_to(tmp_path / "dest").as_posix() for p in written] == [
        "data.nc", "sub/more.nc"]
    assert (tmp_path / "dest" / "sub" / "more.nc").read_text() == "more"


@pytest.mark.parametrize("name, why", [
    ("../../pwned.txt", "'..' path component"),
    ("good/../../pwned.txt", "'..' path component"),
    ("./sneaky.txt", "'.' or '..' path component"),
    ("a//b.txt", "empty, '.' or '..'"),
    ("/abs_posix.txt", "absolute"),
    ("C:/Windows/Temp/pwned.txt", "drive letter"),
    ("//server/share/pwned.txt", "drive letter"),
    ("stream:name.txt", "':'"),
])
def test_rejects_a_traversal_member_and_writes_nothing(tmp_path, name, why):
    z = _zip(tmp_path / "evil.zip", [(name, "payload")])
    dest = _dest(tmp_path)
    with pytest.raises(provenance.UnsafeArchiveError) as e:
        provenance.safe_extract(z, dest)
    assert why in str(e.value)
    assert list(dest.rglob("*")) == []
    # and nothing landed beside the destination either
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dest", "evil.zip"]


@pytest.mark.parametrize("name, why", [
    ("..\\..\\pwned.txt", "backslash"),
    ("nul\x00.txt", "NUL"),
])
def test_rejects_names_the_zip_reader_would_have_rewritten(name, why, tmp_path):
    """Measured: ZipInfo.__init__ runs _sanitize_filename on read, so on Windows
    a backslash becomes '/' and a name is truncated at the first NUL before
    safe_extract ever sees it. These names therefore cannot be produced by a
    round trip through zipfile -- validate the member directly so the rule is
    still covered, and so it holds on POSIX where they DO survive."""
    info = zipfile.ZipInfo("placeholder")
    info.filename = name
    with pytest.raises(provenance.UnsafeArchiveError, match=why):
        provenance._validate_member(tmp_path / "x.zip", tmp_path / "dest", info)


def test_rejects_a_symlink_member(tmp_path):
    z = _zip(tmp_path / "link.zip", [("data.nc", "x")],
             symlinks=[("escape", "C:/Windows")])
    with pytest.raises(provenance.UnsafeArchiveError, match="symlink"):
        provenance.safe_extract(z, _dest(tmp_path))


def test_rejects_two_members_that_land_on_one_path(tmp_path):
    # stdlib mangles '../data.nc' to 'data.nc'; here they must not collide
    # silently, because fetch._open picks NetCDFs out of that directory.
    z = _zip(tmp_path / "dup.zip", [("data.nc", "first"), ("DATA.NC", "second")])
    with pytest.raises(provenance.UnsafeArchiveError, match="collides"):
        provenance.safe_extract(z, _dest(tmp_path))


def test_rejects_an_archive_over_the_uncompressed_cap(tmp_path):
    z = _zip(tmp_path / "bomb.zip", [("a.nc", "x" * 4096), ("b.nc", "y" * 4096)])
    with pytest.raises(provenance.UnsafeArchiveError, match="uncompressed total"):
        provenance.safe_extract(z, _dest(tmp_path), max_total_bytes=5000)


def test_extraction_is_all_or_nothing(tmp_path):
    # the hostile member is LAST; the benign one before it must not survive
    z = _zip(tmp_path / "mixed.zip",
             [("good.nc", "keep me"), ("../../pwned.txt", "payload")])
    dest = _dest(tmp_path)
    with pytest.raises(provenance.UnsafeArchiveError):
        provenance.safe_extract(z, dest)
    assert list(dest.rglob("*")) == []


def test_stdlib_extractall_would_have_mangled_rather_than_refused(tmp_path):
    """Why this helper exists, as an executable claim.

    CPython does not let a member escape -- it silently rewrites the name. Two
    members then land where the archive named one, which is exactly the input
    fetch._open reads. If a future CPython starts refusing outright, this test
    fails and the comment in provenance.py should be re-read, not deleted.
    """
    z = _zip(tmp_path / "mangle.zip",
             [("../../pwned.txt", "a"), ("good/data.nc", "b"), ("../data.nc", "c")])
    dest = _dest(tmp_path)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dest)          # stdlib: no exception
    landed = sorted(p.relative_to(dest).as_posix()
                    for p in dest.rglob("*") if p.is_file())
    assert landed == ["data.nc", "good/data.nc", "pwned.txt"]
    assert (dest / "data.nc").read_text() == "c"   # the '../' member, renamed
    assert not (tmp_path / "pwned.txt").exists()   # no escape, just mangling


# -------------------------------------------------- names the FILESYSTEM
# rewrites, which the checks above cannot see because they compare the member
# string, not what Windows will do with it.


@pytest.mark.parametrize("evil", ["data.nc.", "data.nc ", "sub./x.nc"])
def test_rejects_a_name_windows_would_silently_rewrite(tmp_path, evil):
    """Measured on this machine: with only the earlier rules, a member named
    ``data.nc.`` was ACCEPTED and its bytes OVERWROTE the genuine ``data.nc``
    -- Windows strips the trailing dot, so the collision check never saw it,
    and ``fetch._open`` then read the attacker's payload as the CAMS cube.
    That is the exact damage this helper exists to prevent."""
    z = _zip(tmp_path / "rewrite.zip",
             [("data.nc", "GENUINE"), (evil, "ATTACKER")])
    dest = _dest(tmp_path)
    with pytest.raises(provenance.UnsafeArchiveError, match="space or dot"):
        provenance.safe_extract(z, dest)
    assert list(dest.rglob("*")) == []


@pytest.mark.parametrize("evil", ["NUL", "nul.nc", "sub/com1.nc"])
def test_rejects_a_windows_reserved_device_name(tmp_path, evil):
    """``dest/NUL`` is the null device in any directory: the write vanishes and
    safe_extract would return a path that does not exist (measured)."""
    z = _zip(tmp_path / "dev.zip", [("data.nc", "real"), (evil, "gone")])
    with pytest.raises(provenance.UnsafeArchiveError, match="reserved device"):
        provenance.safe_extract(z, _dest(tmp_path))


@pytest.mark.parametrize("members", [
    [("a", "I am a file"), ("a/b.nc", "needs a to be a dir")],
    [("a/b.nc", "needs a to be a dir"), ("a", "I am a file")],
])
def test_rejects_a_file_that_another_member_needs_to_be_a_directory(tmp_path, members):
    """Measured with only the earlier rules: FileExistsError in one order and
    PermissionError in the other, raised half way through the write, leaving a
    partial tree -- i.e. neither all-or-nothing nor an UnsafeArchiveError."""
    z = _zip(tmp_path / "clash.zip", members)
    dest = _dest(tmp_path)
    with pytest.raises(provenance.UnsafeArchiveError, match="needs that same path"):
        provenance.safe_extract(z, dest)
    assert list(dest.rglob("*")) == []


def test_a_corrupt_member_rolls_back_the_members_already_written(tmp_path):
    """The validator cannot see a bad CRC. All-or-nothing must still hold,
    because the caller's next move is to read the NetCDFs in this directory."""
    z = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("good.nc", "keep me")
        zf.writestr("bad.nc", "x" * 4096)
    raw = bytearray(z.read_bytes())
    raw[raw.rindex(b"xxxx")] = ord("y")          # break bad.nc's CRC
    z.write_bytes(bytes(raw))
    dest = _dest(tmp_path)
    with pytest.raises(zipfile.BadZipFile):
        provenance.safe_extract(z, dest)
    assert list(dest.rglob("*")) == []


def test_rejects_a_member_that_would_overwrite_the_archive_itself(tmp_path):
    """boundary.load_countries extracts into ``zip_path.parent``. Measured with
    only the earlier rules: the member REPLACED the downloaded Natural Earth
    zip, and load_countries caches on 'does the zip exist', so the substituted
    archive would be reused on every later run."""
    z = _zip(tmp_path / "ne.zip",
             [("ne_10m_admin_0_countries.shp", "shp"), ("ne.zip", "CLOBBER")])
    before = z.read_bytes()
    with pytest.raises(provenance.UnsafeArchiveError, match="archive being read"):
        provenance.safe_extract(z, z.parent)
    assert z.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ne.zip"]
