import fs from "node:fs";
import assert from "node:assert/strict";

const wasmPath = process.argv[2];
if (!wasmPath) throw new Error("usage: node wasm_orbis_loader.mjs PATH_TO_WASM");
const bytes = fs.readFileSync(wasmPath);
const module = await WebAssembly.compile(bytes);
const moduleImports = WebAssembly.Module.imports(module);
assert.ok(moduleImports.every((entry) => entry.module.startsWith("__wbindgen_")),
  "the production WebGPU compile graph may only add wasm-bindgen runtime imports");
const imports = {};
for (const entry of moduleImports) {
  imports[entry.module] ??= {};
  imports[entry.module][entry.name] = (..._args) => 0;
}
const { exports: e } = await WebAssembly.instantiate(module, imports);

assert.deepEqual([0, 1, 2, 3].map((field) => e.forge3d_orbis_global_tile_bound(2, 2, 1, field)),
  [0, 0, 90, 45], "wasm uses the production COG global-tile geography contract");

assert.equal(e.forge3d_orbis_loader_init(0, 2), 0);
assert.equal(e.forge3d_orbis_loader_init(1025, 2), 0,
  "public max_in_flight cap is enforced before queue/allocation construction");
assert.equal(e.forge3d_orbis_loader_init(1, 4096), 0, "oversize config rejected");
const handle = e.forge3d_orbis_loader_init(1, 2);
const foreign = e.forge3d_orbis_loader_init(1, 2);
assert.notEqual(handle, 0);
assert.notEqual(foreign, 0);
assert.notEqual(handle, foreign, "numeric handles are registry identities");
assert.equal(e.forge3d_orbis_registry_stat(0), 0);
assert.equal(e.forge3d_orbis_loader_request(0x7fffffff, 1, 0, 0), 0);
assert.equal(e.forge3d_orbis_loader_request(handle, 0xffffffff, 0, 0), 0,
  "invalid LOD is rejected before any shift or queue mutation");
assert.equal(e.forge3d_orbis_loader_request(handle, 1, 2, 0), 0,
  "x outside the LOD axis is rejected");
assert.equal(e.forge3d_orbis_loader_cancel(handle, 1, 0xffffffff, 0), 0);
assert.equal(e.forge3d_orbis_loader_resolve_page(handle, 32, 0, 0, 0), -1);
assert.equal(e.forge3d_orbis_loader_request(handle, 1, 0, 0), 1);
assert.equal(e.forge3d_orbis_loader_request(handle, 1, 0, 0), 0, "dedup");
assert.equal(e.forge3d_orbis_loader_request(handle, 1, 1, 0), 0, "bounded");
assert.equal(e.forge3d_orbis_loader_poll_request(handle), 1);
const ticket = [0, 1, 2, 3, 4].map((field) =>
  e.forge3d_orbis_loader_last_request(handle, field),
);
const [lod, x, y, generationLo, generationHi] = ticket;
assert.equal(generationLo, 1, "invalid tile IDs do not consume request generations");
assert.equal(generationHi, 0);

assert.equal(
  e.forge3d_orbis_alloc_heights(handle, lod, x, y, generationLo, generationHi, 2, 3),
  0,
  "payload dimensions must match configured dimensions",
);
assert.equal(
  e.forge3d_orbis_alloc_heights(handle, lod, x, y, generationLo + 1, generationHi, 2, 2),
  0,
  "foreign generation rejected before allocation",
);
const allocation = e.forge3d_orbis_alloc_heights(
  handle, lod, x, y, generationLo, generationHi, 2, 2,
);
assert.notEqual(allocation, 0);
assert.equal(e.forge3d_orbis_loader_last_request(handle, 3), 0,
  "successful allocation consumes the polled request mailbox");
assert.equal(
  e.forge3d_orbis_alloc_heights(handle, lod, x, y, generationLo, generationHi, 2, 2),
  0,
  "only one live allocation is allowed per loader ticket",
);
assert.equal(e.forge3d_orbis_registry_stat(0), 1);
assert.equal(e.forge3d_orbis_registry_stat(1), 20);
assert.equal(e.forge3d_orbis_heights_pointer(foreign, allocation), 0);
assert.equal(e.forge3d_orbis_free_heights(foreign, allocation), 0);
assert.equal(e.forge3d_orbis_loader_complete(foreign, allocation), 0);
const pointer = e.forge3d_orbis_heights_pointer(handle, allocation);
const coveragePointer = e.forge3d_orbis_coverage_pointer(handle, allocation);
assert.notEqual(pointer, 0);
assert.notEqual(coveragePointer, 0);
new Float32Array(e.memory.buffer, pointer, 4).set([10, 20, 30, 40]);
new Uint8Array(e.memory.buffer, coveragePointer, 4).set([255, 0, 64, 255]);
assert.equal(e.forge3d_orbis_loader_complete(handle, allocation), 1);
assert.equal(
  e.forge3d_orbis_alloc_heights(handle, lod, x, y, generationLo, generationHi, 2, 2),
  0,
  "a successfully submitted ticket is no longer allocatable",
);
assert.equal(e.forge3d_orbis_registry_stat(0), 1, "submitted payload remains accounted");
assert.equal(e.forge3d_orbis_loader_request(handle, lod, x, y), 0,
  "a submitted payload retains unique tile ownership until it is drained");
assert.equal(e.forge3d_orbis_loader_complete(handle, allocation), 0, "double completion safe");
assert.equal(e.forge3d_orbis_free_heights(handle, allocation), 0, "completion reclaimed payload");
assert.equal(e.forge3d_orbis_loader_drain_completion(handle), 1);
assert.equal(e.forge3d_orbis_loader_completion_meta(handle, 5), 4);
assert.equal(e.forge3d_orbis_loader_completion_sample(handle, 2), 30);
assert.equal(e.forge3d_orbis_loader_completion_coverage_sample(handle, 1), 0);
assert.equal(e.forge3d_orbis_loader_completion_coverage_sample(handle, 2), 64,
  "partial coverage survives completion into the shared residency/page-table path");
assert.equal(e.forge3d_orbis_registry_stat(0), 1, "drained payload remains accounted while readable");
assert.equal(e.forge3d_orbis_loader_drain_completion(handle), 0,
  "drain with no new terminal must not return stale completion as new");
assert.equal(e.forge3d_orbis_registry_stat(0), 1,
  "repeated empty drain cannot free a still-readable completion");
assert.equal(e.forge3d_orbis_loader_resolve_page(handle, lod, x, y, 0), lod);
assert.equal(e.forge3d_orbis_loader_resolve_page(handle, lod + 1, x * 2, y * 2, 0), lod,
  "production page-table path resolves a missing child to its resident ancestor");
assert.equal(e.forge3d_orbis_loader_pending(handle), 0);
assert.equal(e.forge3d_orbis_loader_release_completion(handle), 1);
assert.equal(e.forge3d_orbis_loader_release_completion(handle), 0, "double release safe");
assert.equal(e.forge3d_orbis_registry_stat(0), 0);

assert.equal(e.forge3d_orbis_loader_request(handle, 1, 1, 0), 1);
assert.equal(e.forge3d_orbis_loader_poll_request(handle), 1);
const cancelLo = e.forge3d_orbis_loader_last_request(handle, 3);
const cancelHi = e.forge3d_orbis_loader_last_request(handle, 4);
const cancelledAllocation = e.forge3d_orbis_alloc_heights(
  handle, 1, 1, 0, cancelLo, cancelHi, 2, 2,
);
assert.notEqual(cancelledAllocation, 0);
assert.equal(e.forge3d_orbis_registry_stat(0), 1);
assert.equal(e.forge3d_orbis_loader_cancel(handle, 1, 1, 0), 1);
assert.equal(e.forge3d_orbis_alloc_heights(
  handle, 1, 1, 0, cancelLo, cancelHi, 2, 2,
), 0, "cancelled ticket is not allocatable");
assert.equal(e.forge3d_orbis_registry_stat(0), 0, "cancel reclaims registered payload");
assert.equal(e.forge3d_orbis_loader_pending(handle), 1, "cancel retains capacity");
assert.equal(e.forge3d_orbis_loader_complete(handle, cancelledAllocation), 0);
assert.equal(e.forge3d_orbis_free_heights(handle, cancelledAllocation), 0, "stale payload reclaimed");
assert.equal(e.forge3d_orbis_loader_poll_cancellation(handle), 1);
assert.equal(e.forge3d_orbis_loader_last_cancellation(handle, 3), cancelLo);
assert.equal(e.forge3d_orbis_loader_ack_cancellation(handle, 1, 1, 0, cancelLo, cancelHi), 1);
assert.equal(e.forge3d_orbis_alloc_heights(
  handle, 1, 1, 0, cancelLo, cancelHi, 2, 2,
), 0, "acknowledged cancellation is not allocatable");
e.forge3d_orbis_loader_drain_completion(handle);
assert.equal(e.forge3d_orbis_loader_pending(handle), 0, "ack releases capacity");

// Replacing a readable completion releases exactly the old payload while the
// new one remains accounted and readable.
const replacementHandle = e.forge3d_orbis_loader_init(2, 1);
const submitOne = (tileX, value) => {
  assert.equal(e.forge3d_orbis_loader_request(replacementHandle, 2, tileX, 0), 1);
  assert.equal(e.forge3d_orbis_loader_poll_request(replacementHandle), 1);
  const fields = [0, 1, 2, 3, 4].map((field) =>
    e.forge3d_orbis_loader_last_request(replacementHandle, field));
  const id = e.forge3d_orbis_alloc_heights(replacementHandle, ...fields, 1, 1);
  new Float32Array(e.memory.buffer,
    e.forge3d_orbis_heights_pointer(replacementHandle, id), 1)[0] = value;
  new Uint8Array(e.memory.buffer,
    e.forge3d_orbis_coverage_pointer(replacementHandle, id), 1)[0] = 255;
  assert.equal(e.forge3d_orbis_loader_complete(replacementHandle, id), 1);
};
submitOne(0, 11);
assert.equal(e.forge3d_orbis_loader_drain_completion(replacementHandle), 1);
submitOne(1, 22);
assert.equal(e.forge3d_orbis_registry_stat(0), 2);
assert.equal(e.forge3d_orbis_loader_drain_completion(replacementHandle), 1);
assert.equal(e.forge3d_orbis_registry_stat(0), 1, "replacement releases exactly old completion");
assert.equal(e.forge3d_orbis_loader_completion_sample(replacementHandle, 0), 22);
assert.equal(e.forge3d_orbis_loader_drain_completion(replacementHandle), 0);
assert.equal(e.forge3d_orbis_registry_stat(0), 1);
assert.equal(e.forge3d_orbis_loader_release_completion(replacementHandle), 1);
assert.equal(e.forge3d_orbis_loader_destroy(replacementHandle), 1);
assert.equal(e.forge3d_orbis_registry_stat(0), 0);

// Browser fetch/decode errors are real bounded terminals. They reclaim a
// registered payload, retain capacity until drain acknowledgement, and allow
// a fresh generation to retry afterward.
assert.equal(e.forge3d_orbis_loader_request(handle, 2, 2, 1), 1);
assert.equal(e.forge3d_orbis_loader_poll_request(handle), 1);
const errorFields = [0, 1, 2, 3, 4].map((field) =>
  e.forge3d_orbis_loader_last_request(handle, field));
const errorAllocation = e.forge3d_orbis_alloc_heights(handle, ...errorFields, 2, 2);
assert.notEqual(errorAllocation, 0);
assert.equal(e.forge3d_orbis_loader_error(handle, ...errorFields, 7), 1);
assert.equal(e.forge3d_orbis_registry_stat(0), 0, "error reclaims registered payload");
assert.equal(e.forge3d_orbis_loader_pending(handle), 1, "error retains bounded capacity until drain");
assert.equal(e.forge3d_orbis_alloc_heights(handle, ...errorFields, 2, 2), 0,
  "terminal error ticket is not allocatable");
assert.equal(e.forge3d_orbis_loader_drain_completion(handle), 0,
  "completion drain cannot silently consume an error terminal");
assert.equal(e.forge3d_orbis_loader_poll_error(handle), 1);
assert.equal(e.forge3d_orbis_loader_poll_error(handle), 1, "unacknowledged error remains visible");
assert.deepEqual([0, 1, 2, 3, 4].map((field) =>
  e.forge3d_orbis_loader_last_error(handle, field)), errorFields);
assert.equal(e.forge3d_orbis_loader_last_error(handle, 5), 7);
assert.equal(e.forge3d_orbis_loader_ack_error(
  handle, ...errorFields.slice(0, 3), errorFields[3] + 1, errorFields[4],
), 0, "foreign error generation cannot release capacity");
assert.equal(e.forge3d_orbis_loader_ack_error(handle, ...errorFields), 1);
assert.equal(e.forge3d_orbis_loader_pending(handle), 0, "error ack releases capacity");
assert.equal(e.forge3d_orbis_loader_request(handle, 2, 2, 1), 1);
assert.equal(e.forge3d_orbis_loader_poll_request(handle), 1);
const retryFields = [0, 1, 2, 3, 4].map((field) =>
  e.forge3d_orbis_loader_last_request(handle, field));
assert.notDeepEqual(retryFields.slice(3), errorFields.slice(3), "retry gets a fresh generation");
const retryAllocation = e.forge3d_orbis_alloc_heights(handle, ...retryFields, 2, 2);
assert.notEqual(retryAllocation, 0);
assert.equal(e.forge3d_orbis_free_heights(handle, retryAllocation), 1);
assert.equal(e.forge3d_orbis_loader_cancel(handle, 2, 2, 1), 1);
assert.equal(e.forge3d_orbis_loader_poll_cancellation(handle), 1);
assert.equal(e.forge3d_orbis_loader_ack_cancellation(handle, ...retryFields), 1);
assert.equal(e.forge3d_orbis_loader_drain_completion(handle), 0);

// Exercise the exact shared HeightPageResidency implementation through the
// production serialized page-table ABI: preserve root, evict cold leaf.
const residencyHandle = e.forge3d_orbis_loader_init(1, 1);
const completeTile = (handle, lod, x, y, value) => {
  assert.equal(e.forge3d_orbis_loader_request(handle, lod, x, y), 1);
  assert.equal(e.forge3d_orbis_loader_poll_request(handle), 1);
  const fields = [0, 1, 2, 3, 4].map((field) => e.forge3d_orbis_loader_last_request(handle, field));
  const id = e.forge3d_orbis_alloc_heights(handle, ...fields, 1, 1);
  assert.notEqual(id, 0);
  const ptr = e.forge3d_orbis_heights_pointer(handle, id);
  const coveragePtr = e.forge3d_orbis_coverage_pointer(handle, id);
  new Float32Array(e.memory.buffer, ptr, 1)[0] = value;
  new Uint8Array(e.memory.buffer, coveragePtr, 1)[0] = 255;
  assert.equal(e.forge3d_orbis_loader_complete(handle, id), 1);
  assert.equal(e.forge3d_orbis_loader_drain_completion(handle), 1);
  assert.equal(e.forge3d_orbis_loader_release_completion(handle), 1);
};
completeTile(residencyHandle, 0, 0, 0, 1);
completeTile(residencyHandle, 2, 0, 0, 2);
assert.equal(e.forge3d_orbis_loader_resolve_page(residencyHandle, 0, 0, 0, 0), 0,
  "max_in_flight=1 reserves an independent pinned-root slot");
assert.equal(e.forge3d_orbis_loader_resolve_page(residencyHandle, 2, 0, 0, 0), 2,
  "root and one requested leaf are simultaneously resident");
assert.equal(e.forge3d_orbis_loader_resolve_page(residencyHandle, 0, 0, 0, 0), 0,
  "root touch uses the shared runtime LRU");
completeTile(residencyHandle, 2, 3, 0, 3);
assert.equal(e.forge3d_orbis_loader_resolve_page(residencyHandle, 2, 0, 0, 0), 0,
  "evicted cold leaf resolves through the pinned root");
assert.equal(e.forge3d_orbis_loader_resolve_page(residencyHandle, 2, 3, 0, 0), 2);
assert.equal(e.forge3d_orbis_loader_destroy(residencyHandle), 1);

// Aggregate accounting includes paired height+coverage payloads. Three 20 MiB
// allocations fit the 64 MiB cap; a fourth is rejected without ledger drift.
const largeHandles = [];
for (let i = 0; i < 4; i += 1) {
  const h = e.forge3d_orbis_loader_init(1, 2048);
  assert.notEqual(h, 0);
  assert.equal(e.forge3d_orbis_loader_request(h, 2, i, 0), 1);
  assert.equal(e.forge3d_orbis_loader_poll_request(h), 1);
  const fields = [0, 1, 2, 3, 4].map((field) => e.forge3d_orbis_loader_last_request(h, field));
  const id = e.forge3d_orbis_alloc_heights(h, ...fields, 2048, 2048);
  assert.equal(id !== 0, i < 3, "aggregate 64 MiB paired-payload cap is enforced");
  if (i === 0) {
    assert.equal(e.forge3d_orbis_loader_complete(h, id), 1);
    assert.equal(e.forge3d_orbis_loader_drain_completion(h), 1);
    assert.equal(e.forge3d_orbis_loader_drain_completion(h), 0);
    assert.equal(e.forge3d_orbis_registry_stat(1), 20 * 1024 * 1024,
      "cross-loader cap still counts a readable completion after empty drain");
  }
  largeHandles.push(h);
}
assert.equal(e.forge3d_orbis_registry_stat(0), 3);
assert.equal(e.forge3d_orbis_registry_stat(1), 60 * 1024 * 1024);
for (const h of largeHandles) assert.equal(e.forge3d_orbis_loader_destroy(h), 1);
assert.equal(e.forge3d_orbis_registry_stat(0), 0, "destroy reclaims all aggregate payloads");

assert.equal(e.forge3d_orbis_loader_destroy(handle), 1);
assert.equal(e.forge3d_orbis_loader_destroy(handle), 0, "stale handle safe");
assert.equal(e.forge3d_orbis_loader_destroy(foreign), 1);
assert.equal(e.forge3d_orbis_registry_stat(0), 0);
console.log("ORBIS wasm registry ABI PASS");
