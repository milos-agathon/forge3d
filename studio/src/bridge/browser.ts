import type { StudioBridge } from "./index";

/**
 * What the app gets when it is loaded outside the desktop shell.
 *
 * The UI does not mount in that case (`main.tsx` renders the notice instead),
 * so nothing here should ever run; it exists so a stray call fails with an
 * explanation rather than `invoke is not a function`.
 */
const NOTICE = "Forge3D Studio needs the desktop shell — run `npm run tauri:dev`.";

export const browserBridge = new Proxy({} as StudioBridge, {
  get(_target, property) {
    if (property === "kind") return "browser";
    return () => Promise.reject(new Error(NOTICE));
  },
});
