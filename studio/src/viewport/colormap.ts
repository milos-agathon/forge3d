//! The engine's elevation ramp, for the parts of the app that are not the
//! renderer: the Material preview's terrain colours and the panel's strip.
//!
//! The data is the engine's own LUT, not a hand-copied palette: `colormapStops`
//! samples `forge3d.get_colormap(name).rgba`, which is **linear** RGB, so the
//! preview can feed it straight into vertex colours (three.js works in linear)
//! and only convert for CSS.

/** A ramp stop: linear RGB in 0..1, the engine's own values. */
export type RampStop = [number, number, number];

/** The engine's catalogue name for a colormap, for labels and tooltips. */
export function colormapTitle(name: string): string {
  return `forge3d:${name}`;
}

/** The studio's clay, in the renderer's working (linear) space. */
export const CLAY_LINEAR: RampStop = [0.257, 0.301, 0.371];

/**
 * The ramp colour at `t` in [0, 1], interpolated between the engine's stops.
 *
 * `strength` mixes it with `base` exactly as the renderer's `mix` albedo mode
 * does between the material set and the colormap: 0 is the material, 1 is the
 * ramp.
 */
export function rampColor(
  stops: RampStop[],
  t: number,
  strength = 1,
  base: RampStop = CLAY_LINEAR,
): RampStop {
  if (stops.length === 0) return base;
  const clamped = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  const scaled = clamped * (stops.length - 1);
  const index = Math.min(stops.length - 1, Math.floor(scaled));
  const next = Math.min(stops.length - 1, index + 1);
  const blend = scaled - index;
  const a = stops[index];
  const b = stops[next];
  const mix = Math.min(1, Math.max(0, Number.isFinite(strength) ? strength : 1));
  return [
    (a[0] + (b[0] - a[0]) * blend) * mix + base[0] * (1 - mix),
    (a[1] + (b[1] - a[1]) * blend) * mix + base[1] * (1 - mix),
    (a[2] + (b[2] - a[2]) * blend) * mix + base[2] * (1 - mix),
  ];
}

/** A linear colour as a CSS `rgb()` string (the panel's strip). */
export function linearToCss(colour: RampStop): string {
  const channel = (value: number) => {
    const clamped = Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
    const encoded =
      clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * Math.pow(clamped, 1 / 2.4) - 0.055;
    return Math.round(encoded * 255);
  };
  return `rgb(${channel(colour[0])} ${channel(colour[1])} ${channel(colour[2])})`;
}
