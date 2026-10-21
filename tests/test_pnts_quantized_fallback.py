import json
import struct
import pytest
from forge3d import tiles3d


def pnts(table, payload):
    encoded = json.dumps(table).encode()
    return struct.pack("<4s6I", b"pnts", 1, 28 + len(encoded) + len(payload), len(encoded), len(payload), 0, 0) + encoded + payload

def test_quantized_does_not_return_zero_positions(monkeypatch):
    monkeypatch.setattr(tiles3d, "NATIVE_AVAILABLE", False)
    data = pnts({"POINTS_LENGTH": 1, "POSITION_QUANTIZED": {"byteOffset": 0},
                 "QUANTIZED_VOLUME_OFFSET": [10,20,30], "QUANTIZED_VOLUME_SCALE": [1,1,1]},
                struct.pack("<3H", 65535, 0, 0))
    with pytest.raises(ValueError, match="POSITION_QUANTIZED.*unsupported"):
        tiles3d.decode_pnts(data)
