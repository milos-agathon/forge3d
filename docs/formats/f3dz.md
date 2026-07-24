# F3DZ v1 byte format

F3DZ is forge3d's deterministic, page-independent, error-bounded single-channel
DEM format. All integers are little-endian. Every floating-point field is stored
as its IEEE-754 binary32 bit pattern. No implementation may infer or substitute
an error tolerance: the stream's `epsilon` is authoritative.

## Container prefix

The fixed header is 64 bytes:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | ASCII magic `F3DZ` |
| 4 | 2 | version (`1`) |
| 6 | 2 | fixed-header size (`64`) |
| 8 | 4 | flags: bit 0 progressive, bit 1 base-only |
| 12 | 4 | grid width |
| 16 | 4 | grid height |
| 20 | 2 | page/tile edge, at most 64 in v1 |
| 22 | 1 | default predictor; `255` means adaptive per page |
| 23 | 1 | reserved, zero |
| 24 | 4 | epsilon in metres, binary32 bits |
| 28 | 4 | page count |
| 32 | 4 | UTF-8 height-datum byte length |
| 36 | 8 | absolute page-index offset |
| 44 | 8 | absolute payload-region offset |
| 52 | 4 | CRC-32/ISO-HDLC of the fixed header with this field zero, followed by datum bytes |
| 56 | 8 | reserved, zero |

The UTF-8 height-datum bytes follow the fixed header. The page index follows the
datum immediately. `page_count` must equal
`ceil(width / tile_size) * ceil(height / tile_size)`.

## Page index

Each page-index record is exactly 64 bytes:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | page x |
| 4 | 4 | page y |
| 8 | 2 | page width |
| 10 | 2 | page height |
| 12 | 1 | selected predictor id |
| 13 | 1 | flags: bit 0 progressive, bit 1 base-only |
| 14 | 2 | reserved, zero |
| 16 | 8 | absolute payload offset |
| 24 | 4 | payload length |
| 28 | 4 | base-layer prefix length within the payload |
| 32 | 4 | CRC-32/ISO-HDLC of the complete stored payload |
| 36 | 4 | refined maximum absolute error, binary32 bits |
| 40 | 4 | base maximum absolute error, binary32 bits |
| 44 | 4 | sample count (`page_width * page_height`) |
| 48 | 4 | NaN/nodata count |
| 52 | 12 | reserved, zero |

Payload offsets are absolute so one index record is sufficient for an
independent range request. A reader validates the header CRC, all offsets and
lengths, and the selected page CRC before entropy decoding. Unknown flags,
inconsistent counts, non-finite error fields, and truncated ranges are hard
errors.

## Page payload

Every payload begins with ASCII `F3PG`, page-payload version `1`, flags, predictor
parameters, and byte lengths for the base and enhancement entropy layers. Each
entropy layer contains:

1. decoded byte count;
2. two initial 32-bit rANS states;
3. two byte-stream lengths;
4. a sparse `(symbol: u8, normalized_frequency: u16)` table whose frequencies
   sum exactly to `4096`;
5. lane 0 and lane 1 renormalization bytes.

The two lanes encode alternating bytes of the logical 32-bit token stream. A
normal texel is one little-endian zig-zag residual word. `0xffffffff` is the
explicit NaN/nodata escape and `0xfffffffe` is the exact finite-value escape;
either escape is followed by one word containing the source binary32 bits.
Escaped samples are excluded from later causal contexts, so neither nodata nor
a rare high-magnitude exact value contaminates prediction. Pages never
reference another page.

Prediction and reconstruction are defined in integer lattice coordinates.
For requested error `epsilon`, the refined lattice step is binary32
`2 * epsilon`; the base lattice step is binary32 `8 * epsilon`. Quantization is
round-to-nearest, ties-to-even. The encoder chooses the nearest lattice value
whose binary32 reconstruction is machine-checked within the declared bound.
Residuals are predicted from already reconstructed lattice values, never source
values. This feedback invariant prevents prediction error from accumulating.

Predictor ids are:

- `0`: 2D Lorenzo (`left + up - upper_left`), causal row-major;
- `1`: deterministic least-squares integer plane;
- `2`: previous/base layer, used by progressive enhancement;
- `3`: order-0, reserved for ablation and diagnostics.

A progressive payload stores a base layer bounded by `4 * epsilon`, then an
enhancement residual that refines the same texels to `epsilon`. A base-only
container is rewritten with valid offsets, lengths, CRCs, and base-quality
flags; it is not an arbitrarily byte-truncated invalid file.

## GPU execution contract

One workgroup owns one page. The two rANS lanes decode in parallel. Lorenzo
pages reconstruct in causal row-wave order; plane prediction and progressive
enhancement allow columns within a row to run in parallel after the entropy
barrier. The final binary32 values are written directly to the R32Float terrain
atlas. CPU and GPU use the same integer residuals and one binary32 multiply, so
their output bytes are required to match exactly.
