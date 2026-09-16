// TERRA-DETERMINATA browser WebGPU canary.
//
// Runs the SAME WGSL sources as the native legs:
//   src/shaders/includes/determinism.wgsl + src/shaders/det_probe.wgsl  (compute)
//   src/shaders/includes/determinism.wgsl + src/shaders/det_raster.wgsl (raster)
// served by scripts/run_browser_probe.py, which launches headless Chrome with
// --dump-dom and scrapes the JSON record written into <pre id="out">.
//
// No dependencies, no build step: plain WebGPU + crypto.subtle.

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes.slice().buffer);
  return Array.from(new Uint8Array(digest)).map(b => b.toString(16).padStart(2, "0")).join("");
}

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: HTTP ${res.status}`);
  return res.text();
}

async function runComputeProbe(device, source) {
  const module = device.createShaderModule({ label: "det_probe.shader", code: source });
  const pipeline = await device.createComputePipelineAsync({
    label: "det_probe.pipeline",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const OUTPUT_BYTES = 16 * 16;
  const output = device.createBuffer({
    label: "det_probe.output",
    size: OUTPUT_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: "det_probe.readback",
    size: OUTPUT_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: output } }],
  });
  const encoder = device.createCommandEncoder({ label: "det_probe.encoder" });
  const pass = encoder.beginComputePass({ label: "det_probe.pass" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(4, 4, 1);
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, OUTPUT_BYTES);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(readback.getMappedRange().slice(0));
  readback.unmap();
  return bytes;
}

// The native raster canary renders to rgba32float: raw f32 bytes are hashed,
// so a one-ULP fragment difference is visible. Match that byte layout here —
// rgba8unorm would quantize every channel to 1/255 and silently hide exactly
// the divergence this leg exists to catch. When the browser adapter lacks
// float32-renderable the fallback is reported via record.raster_format so the
// checker can treat it as a non-comparable (8-bit) raster leg.
async function runRasterProbe(device, source, format) {
  const module = device.createShaderModule({ label: "det_raster.shader", code: source });
  const pipeline = await device.createRenderPipelineAsync({
    label: "det_raster.pipeline",
    layout: "auto",
    vertex: { module, entryPoint: "vs_main" },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
  const SIZE = 64;
  const BYTES_PER_PIXEL = format === "rgba32float" ? 16 : 4;
  const ROW_BYTES = SIZE * BYTES_PER_PIXEL;
  const target = device.createTexture({
    label: "det_raster.target",
    size: { width: SIZE, height: SIZE },
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: "det_raster.readback",
    size: ROW_BYTES * SIZE,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({ label: "det_raster.encoder" });
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: target.createView(),
      loadOp: "clear",
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      storeOp: "store",
    }],
  });
  pass.setPipeline(pipeline);
  pass.draw(3, 1, 0, 0);
  pass.end();
  encoder.copyTextureToBuffer(
    { texture: target },
    { buffer: readback, bytesPerRow: ROW_BYTES, rowsPerImage: SIZE },
    { width: SIZE, height: SIZE },
  );
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(readback.getMappedRange().slice(0));
  readback.unmap();
  return bytes;
}

async function main() {
  const out = document.getElementById("out");
  const record = { status: "ok" };
  try {
    if (!navigator.gpu) throw new Error("navigator.gpu missing (WebGPU disabled)");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("requestAdapter returned null");
    const info = adapter.info || {};
    record.adapter = {
      name: info.description || info.device || info.vendor || "webgpu-adapter",
      vendor: info.vendor || null,
      architecture: info.architecture || null,
      device: info.device || null,
      device_type: "webgpu",
      backend: "BrowserWebGpu",
      // SwiftShader/llvmpipe are software renderers; physical hardware reports
      // real vendor strings. Truthfully flag software so the checker can
      // distinguish a software-rendered browser leg from a hardware one.
      software_fallback:
        /swiftshader|llvmpipe|software|warp/i.test(
          `${info.vendor || ""} ${info.device || ""} ${info.description || ""} ${info.architecture || ""}`
        ),
    };
    // rgba32float render targets require float32-renderable. Requesting
    // float32-filterable auto-enables it per spec; some adapters list only
    // the filterable name. Without either, fall back to rgba8unorm — honest
    // but not byte-comparable to the native raster leg.
    const feats = adapter.features;
    const requiredFeatures =
      feats && feats.has("float32-renderable") ? ["float32-renderable"]
      : feats && feats.has("float32-filterable") ? ["float32-filterable"]
      : [];
    const rasterFormat = requiredFeatures.length ? "rgba32float" : "rgba8unorm";
    const device = await adapter.requestDevice({ requiredFeatures });
    record.raster_format = rasterFormat;
    record.adapter_features = adapter.features ? [...adapter.features].sort() : [];

    const detWgsl = await fetchText("determinism.wgsl");
    const probeWgsl = await fetchText("det_probe.wgsl");
    const rasterWgsl = await fetchText("det_raster.wgsl");
    const probeSource = `${detWgsl}\n${probeWgsl}`;
    const rasterSource = `${detWgsl}\n${rasterWgsl}`;

    const probeBytes = await runComputeProbe(device, probeSource);
    record.probe_sha256 = await sha256Hex(probeBytes);
    record.probe_bytes_hex = Array.from(probeBytes).map(b => b.toString(16).padStart(2, "0")).join("");
    record.wgsl_sha256 = await sha256Hex(new TextEncoder().encode(probeSource));

    const rasterBytes = await runRasterProbe(device, rasterSource, rasterFormat);
    record.raster_sha256 = await sha256Hex(rasterBytes);
    record.raster_bytes_hex = Array.from(rasterBytes).map(b => b.toString(16).padStart(2, "0")).join("");
    record.raster_wgsl_sha256 = await sha256Hex(new TextEncoder().encode(rasterSource));
  } catch (err) {
    record.status = "failed";
    record.reason = String(err && err.message ? err.message : err);
  }
  out.textContent = JSON.stringify(record);
  document.title = record.status;
  // The runner waits on this POST: --dump-dom/--virtual-time-budget races
  // ahead of real GPU async work, so the record is pushed back to the local
  // server as soon as it is final.
  try {
    await fetch("/result", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(record),
    });
  } catch (_) { /* DOM dump still carries the record for manual runs */ }
}

main();
