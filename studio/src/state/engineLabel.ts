import type { EngineStatus } from "../bridge/types";

/**
 * The status bar's one-line engine summary.
 *
 * The probe falls back to the installed package metadata when the engine
 * module has no `__version__`; if even that is missing it reports the literal
 * "unknown". Showing "forge3d unknown" next to a working engine reads as a
 * failure, so an unknown version is dropped instead of printed.
 */
export function engineLabel(engine: EngineStatus | null | undefined): string {
  if (!engine) return "probing engine…";
  if (!engine.available) return "engine unavailable";

  const version = engine.packageVersion?.trim() ?? "";
  const known = version.length > 0 && version.toLowerCase() !== "unknown";
  const parts: string[] = [known ? `forge3d ${version}` : "forge3d"];
  if (engine.backend) parts.push(engine.backend);
  if (engine.adapter) {
    parts.push(engine.adapter.replace(/^NVIDIA GeForce\s*/, "").slice(0, 44));
  }
  if (engine.software) parts.push("software");
  return parts.join(" · ");
}
