/** The status bar must never claim "forge3d unknown" next to a working engine. */
import { engineLabel } from "./engineLabel.ts";
import type { EngineStatus } from "../bridge/types";

const base: EngineStatus = {
  available: true,
  python: "/usr/bin/python3",
  packageVersion: "1.49.0",
  viewerBinary: null,
  backend: "vulkan",
  adapter: "NVIDIA GeForce RTX 3070",
  detail: "",
  features: null,
  software: false,
} as EngineStatus;

const label = (patch: Partial<EngineStatus>) => engineLabel({ ...base, ...patch });

if (label({}).indexOf("forge3d 1.49.0") !== 0) throw new Error("version missing");
if (label({}).includes("RTX 3070") === false) throw new Error("adapter missing");
if (label({ packageVersion: null }).startsWith("forge3d ") === false) throw new Error("versionless label");
if (label({ packageVersion: "unknown" }).includes("unknown")) {
  throw new Error("an unknown version must not be printed");
}
if (label({ available: false }) !== "engine unavailable") throw new Error("unavailable label");
if (engineLabel(null) !== "probing engine…") throw new Error("probing label");
if (label({ software: true }).endsWith("software") === false) throw new Error("software marker");
if (label({ adapter: "Mesa llvmpipe" }).includes("Mesa llvmpipe") === false) {
  throw new Error("non-NVIDIA adapter kept verbatim");
}

console.log("engineLabel: all assertions passed");
