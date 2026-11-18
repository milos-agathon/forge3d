import type { CartographyLayer, Project } from "../bridge/types";

/**
 * Reading the map's stack.
 *
 * The stack is back to front, so the *last* matching visible layer is the one
 * on top — the one the reader sees. The helpers here are the shell's one
 * answer to "what does the map show?", so the viewport, the panel and the
 * inspector cannot disagree about it.
 */

/** The visible layers, in draw order. */
function visibleLayers(project: Project | null | undefined): CartographyLayer[] {
  return (project?.cartography ?? []).filter((layer) => layer.visible);
}

/** The topmost visible elevation ramp, or null. */
export function topRamp(project: Project | null | undefined): CartographyLayer | null {
  const ramps = visibleLayers(project).filter((layer) => layer.type === "elevationRamp");
  return ramps.length > 0 ? ramps[ramps.length - 1] : null;
}

/** The visible vector layers, in draw order. */
export function vectorLayers(project: Project | null | undefined): CartographyLayer[] {
  return visibleLayers(project).filter((layer) => layer.type === "vector");
}

/** The topmost visible imagery layer, or null. */
export function topImagery(project: Project | null | undefined): CartographyLayer | null {
  const layers = visibleLayers(project).filter((layer) => layer.type === "imagery");
  return layers.length > 0 ? layers[layers.length - 1] : null;
}

/** The visible labels layers, in draw order. */
export function labelLayers(project: Project | null | undefined): CartographyLayer[] {
  return visibleLayers(project).filter((layer) => layer.type === "labels");
}

/** The visible graticule layers, in draw order. */
export function graticuleLayers(project: Project | null | undefined): CartographyLayer[] {
  return visibleLayers(project).filter((layer) => layer.type === "graticule");
}

/** The topmost visible plate layer, or null. */
export function plateLayer(project: Project | null | undefined): CartographyLayer | null {
  const plates = visibleLayers(project).filter((layer) => layer.type === "plate");
  return plates.length > 0 ? plates[plates.length - 1] : null;
}

/** The visible polygon layers, in draw order. */
export function polygonLayers(project: Project | null | undefined): CartographyLayer[] {
  return visibleLayers(project).filter((layer) => layer.type === "polygon");
}
