import { browserBridge } from "./browser";
import type { StudioBridge } from "./index";
import { tauriBridge } from "./tauri";

const isTauri =
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export const bridge: StudioBridge = isTauri ? tauriBridge : browserBridge;

export type { StudioBridge };
export * from "./types";
