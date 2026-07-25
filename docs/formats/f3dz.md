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
| 13 | 1 | flags: bit 0 progressive, bit 1 base-only, bit 2 low-barrier GPU safe, bit 3 direct-token GPU safe |
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

Every payload begins with this 56-byte page header:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | ASCII magic `F3PG` |
| 4 | 2 | page version (`1`) |
| 6 | 2 | flags: bit 0 progressive, bit 1 base-only, bit 2 low-barrier GPU safe, bit 3 direct-token GPU safe |
| 8 | 1 | base predictor id |
| 9 | 1 | enhancement predictor id |
| 10 | 2 | reserved, zero |
| 12 | 12 | signed plane x slope, y slope, and intercept |
| 24 | 4 | base entropy-layer byte length |
| 28 | 4 | enhancement entropy-layer byte length |
| 32 | 4 | logical sample count |
| 36 | 4 | fine lattice step, binary32 bits |
| 40 | 4 | base lattice step, binary32 bits |
| 44 | 4 | exact base residual GCD scale |
| 48 | 4 | exact enhancement residual GCD scale |
| 52 | 4 | reserved, zero |

The non-zero GCD scales remove a common integer factor before entropy coding
and are multiplied back with checked integer arithmetic before prediction.
They do not change any reconstructed lattice value.

Each entropy layer contains:

1. decoded byte count;
2. two initial 32-bit rANS states;
3. two byte-stream lengths;
4. a canonical adaptive-static frequency table whose frequencies sum exactly
   to `4096`: alphabets up to 21 symbols use sorted
   `(symbol: u8, normalized_frequency: u16)` entries; larger alphabets use a
   256-bit presence bitmap followed by ascending `(frequency - 1)` values
   packed at 12 bits each;
5. lane 0 and lane 1 renormalization bytes.

The two lanes encode alternating bytes of a canonical variable-width token
stream. Every logical texel begins with one unsigned LEB128 value: `0` is the
explicit canonical NaN/nodata escape, `1` is the exact finite-value escape and
is followed by four little-endian source binary32 bytes, `2` is a run marker,
and any value `n >= 3` is the normal zig-zag residual `n - 3`. A run marker is
followed by a sample code (`0` for NaN, or residual `r + 1`) and a run length
of at least four, both unsigned LEB128. RAW values are never run encoded.
LEB128 values are minimally encoded and limited to five bytes. NaN therefore
needs no redundant payload: decode emits the canonical quiet-NaN bits
`0x7fc00000`. Escaped samples are excluded from later causal contexts, so
neither nodata nor a rare high-magnitude exact value contaminates prediction.
Pages never reference another page.

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

One workgroup owns one page. Invocations 0 and 1 concurrently expand the two
interleaved rANS lanes. Pages additionally marked direct-token-safe contain
exactly one canonical one-byte ordinary residual token per sample in every
stored layer; all 64 invocations may therefore parse distinct samples without
changing the CPU-decodable byte stream. The encoder uses this representation
only when its rANS layer is no larger than the normal canonical stream.
Lorenzo pages marked low-barrier-safe reconstruct with a checked
horizontal prefix scan followed by a checked vertical prefix scan. The encoder
sets that page flag for Lorenzo only after proving there are no escape samples,
all prefix intermediates fit `i32`, and the result is identical to the canonical
causal decoder. Plane and order-zero pages are independently reconstructible
and may use the same low-barrier pipeline. The GPU rechecks the predictor, flag,
token kind, and every arithmetic operation; a forged flag therefore fails
closed. Batches containing any unmarked page use the general fixed-wave Lorenzo
pipeline, which also supports escape samples. Progressive enhancement allows
columns within a row to run in parallel after the entropy barrier. The final
binary32 values are written directly to the R32Float terrain atlas. CPU and GPU
use the same integer residuals and one binary32 multiply, so their output bytes
are required to match exactly.

## TIFF/COG embedding

forge3d reserves private TIFF compression value `65003` for an F3DZ v1 tile.
The TIFF IFD must still declare one `32`-bit floating-point sample and tile
dimensions identical to the embedded F3DZ grid. The TIFF horizontal predictor
is not applied: F3DZ performs its own terrain-aware prediction. Readers decode
this branch through the same fail-closed CPU decoder and reject mismatched
dimensions or sample metadata.
