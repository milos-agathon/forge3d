import type { RenderView, Transform } from "../bridge/types";

/**
 * The viewport's live camera state, published every frame.
 *
 * `current` is the latest mapping of the viewport camera into the Forge3D
 * renderer's frame. The viewport writes it every frame; the render actions read
 * it when a frame or still is requested, so the engine reproduces the view the
 * user is actually looking at.
 *
 * `cameraTransform` is the viewport camera as a scene transform. The timeline's
 * path recorder samples it while a take is running, so a recorded key is the
 * pose on screen rather than a reconstruction.
 *
 * `current` is `null` when the scene has no terrain to render (the renderer is a
 * terrain pipeline) — callers fall back to the shell's default framing.
 */
export const viewStore: {
  current: RenderView | null;
  cameraTransform: Transform | null;
} = { current: null, cameraTransform: null };
