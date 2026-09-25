"""Deterministically compact the cited IMCCE and CDS astronomy tables.

The input directory is a disposable download cache. Runtime code only reads the
three generated binary files; it has no network or Python dependency.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/sidera/_source_cache"
OUTPUT = ROOT / "data/sidera"
RAD = 648000.0 / math.pi
DEG = math.pi / 180.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vsop() -> dict[str, int]:
    body_names = ("ear", "mer", "ven", "mar", "jup", "sat")
    records = bytearray()
    counts = {}
    for body, name in enumerate(body_names):
        source = SOURCE / f"VSOP87D.{name}"
        variable = power = None
        count = 0
        for line in source.read_text(encoding="ascii").splitlines():
            if "VSOP87 VERSION" in line:
                header = re.search(r"VARIABLE\s+(\d).*\*T\*\*(\d)", line)
                if header is None:
                    raise ValueError(f"unknown VSOP header: {line}")
                variable, power = map(int, header.groups())
                continue
            fields = line.split()
            if len(fields) < 3 or variable is None or power is None:
                raise ValueError(f"unknown VSOP term: {line}")
            amplitude, phase, frequency = map(float, fields[-3:])
            records.extend(struct.pack("<BBBddd", body, variable - 1, power, amplitude, phase, frequency))
            count += 1
        counts[name] = count
    (OUTPUT / "vsop87d.bin").write_bytes(struct.pack("<4sI", b"V87D", sum(counts.values())) + records)
    return counts


def lunar_arguments() -> tuple[list[list[float]], list[list[float]], list[float]]:
    c1, c2 = 60.0, 3600.0
    w = [
        [(218 + 18 / c1 + 59.95571 / c2) * DEG, 1732559343.73604 / RAD, -5.8883 / RAD, 0.006604 / RAD, -0.00003169 / RAD],
        [(83 + 21 / c1 + 11.67475 / c2) * DEG, 14643420.2632 / RAD, -38.2776 / RAD, -0.045047 / RAD, 0.00021301 / RAD],
        [(125 + 2 / c1 + 40.39816 / c2) * DEG, -6967919.3622 / RAD, 6.3622 / RAD, 0.007625 / RAD, -0.0003586 / RAD],
    ]
    earth = [(100 + 27 / c1 + 59.22059 / c2) * DEG, 129597742.2758 / RAD, -0.0202 / RAD, 0.000009 / RAD, 0.00000015 / RAD]
    peri = [(102 + 56 / c1 + 14.42753 / c2) * DEG, 1161.2283 / RAD, 0.5327 / RAD, -0.000138 / RAD, 0.0]
    delaunay = [[w[0][k] - earth[k], earth[k] - peri[k], w[0][k] - w[1][k], w[0][k] - w[2][k]] for k in range(5)]
    delaunay[0][0] += math.pi
    zeta = [w[0][0], w[0][1] + 5029.0966 / RAD]
    return w, delaunay, zeta


def planetary_arguments(earth: list[float]) -> list[list[float]]:
    start = [
        (252 + 15 / 60 + 3.25986 / 3600) * DEG,
        (181 + 58 / 60 + 47.28305 / 3600) * DEG,
        earth[0],
        (355 + 25 / 60 + 59.78866 / 3600) * DEG,
        (34 + 21 / 60 + 5.34212 / 3600) * DEG,
        (50 + 4 / 60 + 38.89694 / 3600) * DEG,
        (314 + 3 / 60 + 18.01841 / 3600) * DEG,
        (304 + 20 / 60 + 55.19575 / 3600) * DEG,
    ]
    rates = [538101628.68898, 210664136.43355, 0.0, 68905077.59284, 10925660.42861, 4399609.65932, 1542481.19393, 786550.32074]
    rates = [rate / RAD for rate in rates]
    rates[2] = earth[1]
    return [start, rates]


def lunar() -> dict[str, int]:
    w, delaunay, zeta = lunar_arguments()
    # Earth mean longitude is w1 - D; D's constant includes an added pi.
    earth = [w[0][0] - (delaunay[0][0] - math.pi), w[0][1] - delaunay[1][0]]
    planetary = planetary_arguments(earth)
    am = 0.074801329518
    alfa = 0.002571881335
    dtasm = 2 * alfa / (3 * am)
    delnu = 0.55604 / RAD / w[0][1]
    dele = 0.01789 / RAD
    delg = -0.08066 / RAD
    delnp = -0.06424 / RAD / w[0][1]
    delep = -0.12879 / RAD
    main = bytearray()
    secondary = bytearray()
    counts = {"main": 0, "secondary": 0, "discarded": 0}
    for number in range(1, 37):
        iv = (number - 1) % 3
        for line in (SOURCE / f"ELP{number}").read_text(encoding="ascii").splitlines()[1:]:
            if number <= 3:
                ilu = [int(line[i : i + 3]) for i in range(0, 12, 3)]
                coef = [float(line[14:27])] + [float(line[i:i+12]) for i in range(27, 99, 12)]
                amplitude = coef[0]
                if number == 3:
                    amplitude *= 1 - 2 * delnu / 3
                tgv = coef[1] + dtasm * coef[5]
                amplitude += tgv * (delnp - am * delnu) + coef[2] * delg + coef[3] * dele + coef[4] * delep
                phase = [sum(ilu[i] * delaunay[k][i] for i in range(4)) for k in range(5)]
                if iv == 2:
                    phase[0] += math.pi / 2
                main.extend(struct.pack("<Bd5d", iv, amplitude, *phase))
                counts["main"] += 1
                continue
            if 10 <= number <= 21:
                ipla = [int(line[i : i + 3]) for i in range(0, 33, 3)]
                phase0 = float(line[34:43]) * DEG
                amplitude = float(line[44:53])
                phase = [phase0, 0.0]
                for k in range(2):
                    if number < 16:
                        phase[k] += ipla[8] * delaunay[k][0] + ipla[9] * delaunay[k][2] + ipla[10] * delaunay[k][3]
                        phase[k] += sum(ipla[i] * planetary[k][i] for i in range(8))
                    else:
                        phase[k] += sum(ipla[i + 7] * delaunay[k][i] for i in range(4))
                        phase[k] += sum(ipla[i] * planetary[k][i] for i in range(7))
                time_power = int(13 <= number <= 15 or 19 <= number <= 21)
            else:
                integers = [int(line[i : i + 3]) for i in range(0, 15, 3)]
                iz, ilu = integers[0], integers[1:]
                phase0 = float(line[16:25]) * DEG
                amplitude = float(line[26:35])
                phase = [phase0, 0.0]
                for k in range(2):
                    phase[k] += iz * zeta[k] + sum(ilu[i] * delaunay[k][i] for i in range(4))
                time_power = 2 if 34 <= number <= 36 else int(7 <= number <= 9 or 25 <= number <= 27)
            secondary.extend(struct.pack("<BBddd", iv, time_power, amplitude, *phase))
            counts["secondary"] += 1
    (OUTPUT / "elp82b.bin").write_bytes(struct.pack("<4sII", b"ELP8", counts["main"], counts["secondary"]) + main + secondary)
    return counts


def catalog() -> int:
    records = bytearray()
    count = 0
    with gzip.open(SOURCE / "catalog.gz", "rt", encoding="ascii") as data:
        for row in data:
            if len(row) < 114 or not row[75:83].strip() or not row[102:107].strip():
                continue
            ra_h = int(row[75:77]) + int(row[77:79]) / 60 + float(row[79:83]) / 3600
            dec = int(row[84:86]) + int(row[86:88]) / 60 + int(row[88:90]) / 3600
            if row[83] == "-":
                dec = -dec
            magnitude = float(row[102:107])
            bv = float(row[109:114]) if row[109:114].strip() else 0.65
            records.extend(struct.pack("<ffff", ra_h * 15 * DEG, dec * DEG, magnitude, bv))
            count += 1
    (OUTPUT / "ybsc5.bin").write_bytes(struct.pack("<4sI", b"YBS5", count) + records)
    return count


def nutation() -> int:
    source = (SOURCE / "nut00b.c").read_text(encoding="ascii")
    table = source.split("} x[] = {", 1)[1].split("};", 1)[0]
    records = bytearray()
    for match in re.finditer(r"\{([^{}]+)\}", table):
        fields = [field.strip() for field in match.group(1).split(",")]
        if len(fields) != 11:
            raise ValueError(f"unexpected ERFA nutation term: {match.group(0)}")
        records.extend(struct.pack("<5b6d", *(int(n) for n in fields[:5]), *(float(n) for n in fields[5:])))
    count = len(records) // 53
    if count != 77:
        raise ValueError(f"expected 77 IAU 2000B terms, got {count}")
    (OUTPUT / "nut00b.bin").write_bytes(struct.pack("<4sI", b"N00B", count) + records)
    return count


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    v, e, c, n = vsop(), lunar(), catalog(), nutation()
    print("VSOP terms", v)
    print("ELP terms", e)
    print("stars", c)
    print("nutation terms", n)
    for name in ("vsop87d.bin", "elp82b.bin", "ybsc5.bin", "nut00b.bin"):
        p = OUTPUT / name
        print(name, p.stat().st_size, sha(p))
    manifest = [
        "# SIDERA source-asset manifest",
        "",
        "Runtime validity: UTC 2000-01-01 through 2050-12-31. Positions outside this interval are rejected.",
        "The coefficient sets are complete for the included bodies, with zero term truncation. The 40-site/epoch Horizons oracle measured maxima of Sun 7.88\", Moon 7.02\", planets 7.95\", phase 0.000023 and lunar semidiameter 0.091\". These are empirical maxima for the frozen vectors, not a global error guarantee.",
        "",
        "| Runtime asset | Bytes | SHA-256 | Description |",
        "| --- | ---: | --- | --- |",
    ]
    descriptions = {
        "vsop87d.bin": "IMCCE VSOP87D; 25,659 heliocentric ecliptic terms for Earth, Mercury, Venus, Mars, Jupiter and Saturn",
        "elp82b.bin": "IMCCE ELP2000-82B; 2,645 main and 35,227 secondary lunar terms",
        "ybsc5.bin": "CDS V/50 Yale Bright Star Catalogue, 5th edition; 9,096 J2000 RA/Dec, V, B−V records, no proper motion",
        "nut00b.bin": "ERFA IAU 2000B; 77 lunisolar terms, BSD license copied as ERFA-LICENSE",
        "delta_t.bin": "Monthly cubic fit of NASA/JPL Horizons TDB−UT1, five-day training samples; maximum sampled residual 1.1 microseconds and seven-day holdout residual 4.0 microseconds (2,662 samples); TT−TDB adds about 2 ms",
    }
    for name, description in descriptions.items():
        path = OUTPUT / name
        manifest.append(f"| `{name}` | {path.stat().st_size} | `{sha(path)}` | {description} |")
    manifest.extend([
        "", "| Source file | SHA-256 | Source |", "| --- | --- | --- |",
    ])
    for path in sorted(SOURCE.iterdir(), key=lambda p: p.name.lower()):
        name = path.name
        if name.startswith("VSOP87D."):
            url = f"https://ftp.imcce.fr/pub/ephem/planets/vsop87/{name}"
        elif name.startswith("ELP") or name in ("elp82b_1", "elp82b_2", "elp_README"):
            url = f"https://ftp.imcce.fr/pub/ephem/moon/elp82b/{'README' if name == 'elp_README' else name}"
        elif name.startswith("catalog"):
            url = f"https://cdsarc.cds.unistra.fr/ftp/V/50/{'ReadMe' if name == 'catalog_ReadMe' else name}"
        else:
            url = f"https://raw.githubusercontent.com/liberfa/erfa/master/src/{name}"
        manifest.append(f"| `{name}` | `{sha(path)}` | {url} |")
    manifest.extend([
        "", "The disposable source cache is regenerated by `tools/sidera_fetch_sources.py`; `tools/sidera_assets.py` compacts it. `tools/sidera_delta_t.py` and `tools/sidera_horizons.py` generate the independent Horizons assets.",
        "JPL oracle settings and the aggregate raw response digest are stored in `tests/data/horizons_vectors.dat`. The runtime assets do not call the network.",
        "", "References: [IMCCE VSOP87](https://ftp.imcce.fr/pub/ephem/planets/vsop87/), [IMCCE ELP82B](https://ftp.imcce.fr/pub/ephem/moon/elp82b/), [CDS Yale V/50](https://cdsarc.cds.unistra.fr/ftp/V/50/ReadMe), [ERFA](https://github.com/liberfa/erfa), [JPL Horizons](https://ssd.jpl.nasa.gov/horizons/).",
    ])
    (OUTPUT / "MANIFEST.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
