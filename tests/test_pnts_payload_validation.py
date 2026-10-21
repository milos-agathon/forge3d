import json
import struct
import pytest
from forge3d import tiles3d


def pnts(table, payload):
    encoded = json.dumps(table).encode()
    return struct.pack("<4s6I", b"pnts", 1, 28 + len(encoded) + len(payload), len(encoded), len(payload), 0, 0) + encoded + payload

@pytest.mark.parametrize("count,payload,extra", [
    (2, bytes(12), {}), (-1, b"", {}), (1.5, bytes(12), {}),
    (1, bytes(12), {"RGB": {"byteOffset": 12}}),
    (1, bytes(12), {"POSITION": {"byteOffset": -1}}),
])
def test_reject_malformed_semantics(monkeypatch, count, payload, extra):
    monkeypatch.setattr(tiles3d, "get_native_symbol", lambda _name: None)
    table = {"POINTS_LENGTH": count, "POSITION": {"byteOffset": 0}, **extra}
    with pytest.raises(ValueError, match="PNTS"):
        tiles3d.decode_pnts(pnts(table, payload))

def test_section_lengths_and_valid_payload(monkeypatch):
    monkeypatch.setattr(tiles3d, "get_native_symbol", lambda _name: None)
    data = pnts({"POINTS_LENGTH": 1, "POSITION": {"byteOffset": 0}}, struct.pack("<3f", 1,2,3))
    assert tiles3d.decode_pnts(data)["positions"].tolist() == [[1,2,3]]
    with pytest.raises(ValueError, match="PNTS section"):
        tiles3d.decode_pnts(data[:-1])
