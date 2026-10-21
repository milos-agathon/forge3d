import struct
import numpy as np
import pytest
from forge3d.codec import compress_dem, decompress_dem


def test_edge_pages_roundtrip_and_corrupt_coordinates():
    samples=np.arange(65*67,dtype=np.float32).reshape(65,67)
    encoded=compress_dem(samples,0.1)
    decoded,_=decompress_dem(encoded)
    assert decoded.shape == samples.shape
    np.testing.assert_allclose(decoded,samples,atol=0.1,rtol=0)
    index=struct.unpack_from("<Q",encoded,36)[0]
    for first,second in [(0,0),(1,0),(0,2)]:
        bad=bytearray(encoded)
        struct.pack_into("<I",bad,index,first)
        struct.pack_into("<I",bad,index+64,second)
        with pytest.raises(ValueError,match="(?i)(page|canonical|coordinate)"):
            decompress_dem(bytes(bad))
