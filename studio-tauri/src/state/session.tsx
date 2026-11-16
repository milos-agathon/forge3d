import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { open as openDialog, save as saveDialog } from "@tauri-apps/plugin-dialog";

import { viewStore } from "../viewport/viewStore";
import { bridge } from "../bridge/env";
import { SessionContext, type SessionApi } from "./session-context";
import { base64ToArrayBuffer, inspectGltf, putAssetBytes } from "../viewport/assets";
import { findRecipe, type RecipeId } from "./recipes";
import {
  canCancelJob,
  clearFinishedJobs as dropFinishedJobs,
  enqueueJob,
  newRenderJob,
  nextQueuedJob,
  patchJob,
  type RenderJob,
} from "./renderQueue";
import type {
  Applied,
  AssetIntegrity,
  AutosaveStatus,
  CartographyLayer,
  Command,
  EngineStatus,
  RenderFrame,
  ScenePreset,
  ViewportMode,
  PrimitiveShape,
  RecoveryEntry,
  SessionState,
  ShellInfo,
  SolarResult,
  RenderView,
  Tool,
  ToolSpace,
} from "../bridge/types";
import { DEFAULT_PRIMITIVES } from "../bridge/types";

const PROJECT_FILTER = {
  name: "Forge3D Studio project",
  extensions: ["studio3d"],
};

const MESH_FILTER = {
  name: "glTF / GLB",
  extensions: ["gltf", "glb"],
};

export function SessionProvider({ children }: { children: ReactNode }) {
  const [info, setInfo] = useState<ShellInfo | null>(null);
  const [state, setState] = useState<SessionState | null>(null);
  const [autosave, setAutosave] = useState<AutosaveStatus | null>(null);
  const [recovery, setRecovery] = useState<RecoveryEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [tool, setTool] = useState<Tool>("select");
  const [space, setSpace] = useState<ToolSpace>("world");
  const [cameraView, setCameraView] = useState(false);
  /** Fly camera. UI state, not project state. */
  const [flyMode, setFlyMode] = useState(false);
  const [engine, setEngine] = useState<EngineStatus | null>(null);
  // The layer the cartography panel is editing. UI state, like the tool: the
  // map's stack is the model, this is which row the author is looking at.
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);
  const [terrainResolution, setTerrainResolution] = useState(96);
  const [presets, setPresets] = useState<ScenePreset[]>([]);
  const [newProjectOpen, setNewProjectOpen] = useState(false);
  const [render, setRender] = useState<{ frame: RenderFrame | null; busy: boolean; error: string | null }>({
    frame: null,
    busy: false,
    error: null,
  });
  const [assetIntegrity, setAssetIntegrity] = useState<AssetIntegrity[]>([]);
  /** Offline render jobs (stills + frame sequences), FIFO. */
  const [renderJobs, setRenderJobs] = useState<RenderJob[]>([]);
  /** Authoritative mirror of `renderJobs`: the worker reads it without a render. */
  const jobsRef = useRef<RenderJob[]>([]);
  /** Render settings captured when the job was queued. */
  const jobSpecs = useRef(new Map<string, { width: number; height: number }>());
  const cancelledJobs = useRef(new Set<string>());
  const queueBusy = useRef(false);
  const noticeTimer = useRef<number | null>(null);

  const flash = useCallback((message: string) => {
    setNotice(message);
    if (noticeTimer.current !== null) window.clearTimeout(noticeTimer.current);
    noticeTimer.current = window.setTimeout(() => setNotice(null), 2600);
  }, []);

  const fail = useCallback((cause: unknown) => {
    setError(cause instanceof Error ? cause.message : String(cause));
  }, []);

  const refreshRecovery = useCallback(async () => {
    try {
      setRecovery(await bridge.listRecovery());
    } catch (cause) {
      fail(cause);
    }
  }, [fail]);

  /**
   * Re-check referenced assets against the filesystem. Advisory: a missing
   * file does not block editing, and the viewport/renderer still report the
   * real read failure — this is what puts the "missing" badge on it.
   */
  const refreshAssetIntegrity = useCallback(async () => {
    try {
      setAssetIntegrity(await bridge.checkAssets());
    } catch {
      // Integrity is a convenience; never turn it into an error toast.
      setAssetIntegrity([]);
    }
  }, []);

  // Re-check when the asset list changes (import, relink, undo, open), not on
  // every snapshot — the playhead produces one per frame.
  const assetSignature = state
    ? state.project.assets
        .map((asset) => `${asset.id}:${asset.path}:${asset.byteSize}`)
        .join("|")
    : "";
  const hasProject = Boolean(state);
  useEffect(() => {
    if (!hasProject) {
      setAssetIntegrity([]);
      return;
    }
    void refreshAssetIntegrity();
  }, [assetSignature, hasProject, refreshAssetIntegrity]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [shellInfo, snapshot, autosaveStatus, entries] = await Promise.all([
          bridge.info(),
          bridge.state(),
          bridge.autosaveStatus(),
          bridge.listRecovery(),
        ]);
        if (cancelled) return;
        setInfo(shellInfo);
        setState(snapshot);
        setAutosave(autosaveStatus);
        setRecovery(entries);
      } catch (cause) {
        if (!cancelled) fail(cause);
      }
    })();
    return () => {
      cancelled = true;
      if (noticeTimer.current !== null) window.clearTimeout(noticeTimer.current);
    };
  }, [fail]);

  // One engine probe per session: the status bar reports whether the renderer
  // can run, without a whole workspace for it.
  useEffect(() => {
    void refreshEngine();
    bridge
      .scenePresets()
      .then(setPresets)
      .catch(() => undefined);
    // Mount-time probe only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keep "autosaved 14:03" honest without a websocket: one cheap poll.
  useEffect(() => {
    const timer = window.setInterval(() => {
      bridge
        .autosaveStatus()
        .then(setAutosave)
        .catch(() => undefined);
    }, 10_000);
    return () => window.clearInterval(timer);
  }, []);

  const guardUnsaved = useCallback(
    (action: string): boolean => {
      if (!state?.dirty) return true;
      return window.confirm(
        `"${state.project.name}" has unsaved changes. Continue and ${action}?`,
      );
    },
    [state],
  );

  const applyCommand = useCallback(
    async (command: Command): Promise<Applied | null> => {
      try {
        const result = await bridge.applyCommand(command);
        setState(result.state);
        return result.applied;
      } catch (cause) {
        fail(cause);
        return null;
      }
    },
    [fail],
  );

  const select = useCallback(async (ids: string[]) => {
    try {
      setState(await bridge.setSelection(ids));
    } catch (cause) {
      fail(cause);
    }
  }, [fail]);

  const setPlayhead = useCallback(
    async (frame: number) => {
      try {
        const result = await bridge.setPlayhead(frame);
        setState(result.state);
      } catch (cause) {
        fail(cause);
      }
    },
    [fail],
  );

  /**
   * Point a directional light from the engine's ephemeris. One undoable
   * `SetTransform`; the date and place are inputs, not project state.
   */
  const setSunFromSolar = useCallback(
    async (
      id: string,
      latitude: number,
      longitude: number,
      datetimeUtc: string,
    ): Promise<SolarResult | null> => {
      try {
        const result = await bridge.setSunFromSolar(
          id,
          latitude,
          longitude,
          datetimeUtc,
        );
        setState(result.state);
        flash(
          result.daytime
            ? `Sun ${result.azimuthDeg.toFixed(0)}° · ${result.elevationDeg.toFixed(1)}° elevation`
            : "Sun is below the horizon",
        );
        return result;
      } catch (cause) {
        fail(cause);
        return null;
      }
    },
    [fail, flash],
  );

  const addPrimitive = useCallback(
    async (shape: PrimitiveShape) => {
      const applied = await applyCommand({
        type: "createPrimitive",
        primitive: DEFAULT_PRIMITIVES[shape],
      });
      const id = applied?.affected[0];
      if (id) await select([id]);
    },
    [applyCommand, select],
  );

  const duplicateSelection = useCallback(async () => {
    const selection = state?.project.selection ?? [];
    if (selection.length === 0) return;
    const applied = await applyCommand({
      type: "batch",
      commands: selection.map((id) => ({ type: "duplicateObject" as const, id })),
    });
    if (applied && applied.affected.length > 0) {
      await select(applied.affected);
      flash(
        applied.affected.length === 1
          ? "Duplicated object"
          : `Duplicated ${applied.affected.length} objects`,
      );
    }
  }, [applyCommand, flash, select, state]);

  const deleteSelection = useCallback(async () => {
    const scene = state?.project.scene;
    const selection = state?.project.selection ?? [];
    if (!scene || selection.length === 0) return;

    // Children before parents, so deleting a subtree in one batch is legal.
    const ordered = [...selection].sort(
      (a, b) =>
        scene.objects.findIndex((object) => object.id === b) -
        scene.objects.findIndex((object) => object.id === a),
    );
    const applied = await applyCommand({
      type: "batch",
      commands: ordered.map((id) => ({ type: "deleteObject" as const, id })),
    });
    if (applied) {
      await select([]);
      flash(
        applied.affected.length === 1
          ? "Deleted object"
          : `Deleted ${applied.affected.length} objects`,
      );
    }
  }, [applyCommand, flash, select, state]);

  const importMesh = useCallback(async () => {
    try {
      const picked = await openDialog({ multiple: false, filters: [MESH_FILTER] });
      const path = typeof picked === "string" ? picked : null;
      if (!path) return;

      setBusy(true);
      const payload = await bridge.readMeshFile(path);
      const buffer = base64ToArrayBuffer(payload.base64);
      // Parse before registering: a corrupt file must not reach the project.
      const { meshes } = await inspectGltf(buffer, payload.name);
      putAssetBytes(payload.path, buffer);

      const applied = await applyCommand({
        type: "importGltf",
        name: payload.name,
        path: payload.path,
        kind: payload.kind,
        byteSize: payload.byteSize,
        node: null,
      });
      const id = applied?.affected[0];
      if (id) await select([id]);
      flash(`Imported ${payload.name} (${meshes} mesh${meshes === 1 ? "" : "es"})`);
    } catch (cause) {
      fail(cause);
    } finally {
      setBusy(false);
    }
  }, [applyCommand, fail, flash, select]);

  const refreshEngine = useCallback(
    async (force = false) => {
      try {
        setEngine(await bridge.forge3dProbe(force));
      } catch {
        // A missing engine is reported through the status bar, not a failure.
      }
    },
    [],
  );

  // The renderer costs seconds per frame, so requests are serialised and only
  // the newest queued view matters: dragging the camera must not build a queue.
  const pendingRender = useRef<{
    width: number;
    height: number;
    quality: "preview" | "final";
    /** The viewport camera at the moment the frame was requested. */
    view: RenderView | undefined;
  } | null>(null);
  const renderInFlight = useRef(false);

  const pumpRender = useCallback(async () => {
    if (renderInFlight.current) return;
    renderInFlight.current = true;
    try {
      while (pendingRender.current) {
        const request = pendingRender.current;
        pendingRender.current = null;
        setRender((current) => ({ ...current, busy: true, error: null }));
        try {
          const frame = await bridge.renderFrame(request.width, request.height, request.quality, request.view);
          setRender({ frame, busy: false, error: null });
        } catch (cause) {
          setRender({
            frame: null,
            busy: false,
            error: cause instanceof Error ? cause.message : String(cause),
          });
        }
      }
    } finally {
      renderInFlight.current = false;
    }
  }, []);

  const renderViewport = useCallback(
    async (width: number, height: number, quality: "preview" | "final" = "preview") => {
      pendingRender.current = { width, height, quality, view: viewStore.current ?? undefined };
      await pumpRender();
    },
    [pumpRender],
  );

  /**
   * Drop the last Rendered frame. A frame belongs to the camera and scene that
   * produced it, so a mode switch or project change must not leave it covering
   * the live view until the next render lands.
   */
  const clearRenderFrame = useCallback(() => {
    setRender({ frame: null, busy: false, error: null });
  }, []);

  // A new project (or a close) has no rendered frame yet.
  const projectId = state?.project.id ?? null;
  useEffect(() => {
    setRender({ frame: null, busy: false, error: null });
  }, [projectId]);

  const selectedMesh = useCallback(() => {
    const project = state?.project;
    const object = project?.scene.objects.find((entry) =>
      project.selection.includes(entry.id),
    );
    return object && object.kind.type === "mesh" ? object : null;
  }, [state]);

  const importMaterialTexture = useCallback(async () => {
    const mesh = selectedMesh();
    if (!mesh || mesh.kind.type !== "mesh") return;
    try {
      const picked = await openDialog({
        multiple: false,
        filters: [{ name: "Image", extensions: ["png", "jpg", "jpeg"] }],
      });
      const path = typeof picked === "string" ? picked : null;
      if (!path) return;
      // Asset registration rejects empty files (`byte_size == 0` reads as
      // "empty"), so stat the real size up front — the webview cannot.
      const byteSize = await bridge.fileSize(path);
      const asset = {
        id: crypto.randomUUID(),
        name: path.split(/[\\/]/).pop() ?? "texture.png",
        path,
        kind: "orthophoto" as const,
        byteSize,
      };
      const applied = await applyCommand({
        type: "batch",
        commands: [
          { type: "addAsset", asset },
          {
            type: "setMeshMaterial",
            id: mesh.id,
            material: { ...mesh.kind.material, texture: asset.id },
          },
        ],
      });
      if (applied) flash("Texture attached");
    } catch (cause) {
      fail(cause);
    }
  }, [applyCommand, fail, flash, selectedMesh]);

  const clearMaterialTexture = useCallback(async () => {
    const mesh = selectedMesh();
    if (!mesh || mesh.kind.type !== "mesh") return;
    await applyCommand({
      type: "setMeshMaterial",
      id: mesh.id,
      material: { ...mesh.kind.material, texture: null },
    });
    flash("Texture removed");
  }, [applyCommand, flash, selectedMesh]);

  /**
   * Pick a Radiance `.hdr` and make it the project's environment. Registered
   * as an asset first (same batch), so the render settings never reference an
   * id the project does not carry.
   */
  const importEnvironment = useCallback(async () => {
    const project = state?.project;
    if (!project) return;
    try {
      const picked = await openDialog({
        multiple: false,
        filters: [{ name: "Radiance HDR environment", extensions: ["hdr"] }],
      });
      const path = typeof picked === "string" ? picked : null;
      if (!path) return;
      const byteSize = await bridge.fileSize(path);
      const asset = {
        id: crypto.randomUUID(),
        name: path.split(/[\\/]/).pop() ?? "environment.hdr",
        path,
        kind: "hdr" as const,
        byteSize,
      };
      const applied = await applyCommand({
        type: "batch",
        commands: [
          { type: "addAsset", asset },
          {
            type: "setRenderSettings",
            render: { ...project.render, environment: asset.id },
          },
        ],
      });
      if (applied) flash(`Environment set to ${asset.name}`);
    } catch (cause) {
      fail(cause);
    }
  }, [applyCommand, fail, flash, state]);

  const clearEnvironment = useCallback(async () => {
    const project = state?.project;
    if (!project) return;
    await applyCommand({
      type: "setRenderSettings",
      render: { ...project.render, environment: null },
    });
    flash("Environment cleared");
  }, [applyCommand, flash, state]);

  const relinkAsset = useCallback(
    async (assetId: string) => {
      try {
        const picked = await openDialog({ multiple: false });
        const path = typeof picked === "string" ? picked : null;
        if (!path) return;
        // Same non-empty invariant as AddAsset: stat, don't guess.
        await applyCommand({
          type: "setAssetPath",
          id: assetId,
          path,
          byteSize: await bridge.fileSize(path),
        });
        flash("Asset relinked");
      } catch (cause) {
        fail(cause);
      }
    },
    [applyCommand, fail, flash],
  );

  const importTerrainDem = useCallback(async () => {
    try {
      const picked = await openDialog({
        multiple: false,
        filters: [{ name: "GeoTIFF DEM", extensions: ["tif", "tiff"] }],
      });
      const path = typeof picked === "string" ? picked : null;
      if (!path) return;
      setBusy(true);
      setState(await bridge.importTerrainDem(path));
      flash(`Terrain DEM set to ${path.split(/[\\/]/).pop()}`);
    } catch (cause) {
      fail(cause);
    } finally {
      setBusy(false);
    }
  }, [fail, flash]);

  // The engine's registry: the same resolution for every import site. The
  // shell resolves the name (the engine fetches it, hash-verified, and
  // caches it), and the file that lands is then imported like any other.
  const importTerrainDemByName = useCallback(
    async (name: string) => {
      try {
        setBusy(true);
        const path = await bridge.datasetPath(name);
        setState(await bridge.importTerrainDem(path));
        flash(`Terrain DEM set from the engine's dataset "${name}"`);
      } catch (cause) {
        fail(cause);
      } finally {
        setBusy(false);
      }
    },
    [fail, flash],
  );

  const addImageryLayerByName = useCallback(
    async (name: string) => {
      try {
        setBusy(true);
        const path = await bridge.datasetPath(name);
        setState(await bridge.addImageryLayer(path, 0.82));
        flash(`Imagery layer added from the engine's dataset "${name}"`);
      } catch (cause) {
        fail(cause);
      } finally {
        setBusy(false);
      }
    },
    [fail, flash],
  );

  const addVectorLayerByName = useCallback(
    async (name: string) => {
      try {
        setBusy(true);
        const path = await bridge.datasetPath(name);
        setState(await bridge.addVectorLayer(path, [0.9, 0.29, 0.1, 1.0], 2));
        flash(`Vector layer added from the engine's dataset "${name}"`);
      } catch (cause) {
        fail(cause);
      } finally {
        setBusy(false);
      }
    },
    [fail, flash],
  );

  const addVectorLayer = useCallback(async () => {
    try {
      const picked = await openDialog({
        multiple: false,
        filters: [
          { name: "Vector layer", extensions: ["gpkg", "geojson", "json"] },
        ],
      });
      const path = typeof picked === "string" ? picked : null;
      if (!path) return;
      setBusy(true);
      setState(await bridge.addVectorLayer(path, [0.9, 0.29, 0.1, 1.0], 2));
      flash("Vector layer added");
    } catch (cause) {
      fail(cause);
    } finally {
      setBusy(false);
    }
  }, [fail, flash]);

  const addPolygonLayer = useCallback(
    async (
      path?: string,
      colour: [number, number, number, number] = [0.2, 0.55, 0.75, 0.7],
    ) => {
      try {
        let target = path;
        if (!target) {
          const picked = await openDialog({
            multiple: false,
            filters: [{ name: "Polygon layer", extensions: ["gpkg", "geojson", "json"] }],
          });
          target = typeof picked === "string" ? picked : undefined;
        }
        if (!target) return;
        setBusy(true);
        setState(await bridge.addPolygonLayer(target, colour));
        flash("Polygon layer added");
      } catch (cause) {
        fail(cause);
      } finally {
        setBusy(false);
      }
    },
    [fail, flash],
  );

  const addImageryLayer = useCallback(async () => {
    try {
      const picked = await openDialog({
        multiple: false,
        filters: [{ name: "Raster overlay", extensions: ["tif", "tiff", "png"] }],
      });
      const path = typeof picked === "string" ? picked : null;
      if (!path) return;
      setBusy(true);
      setState(await bridge.addImageryLayer(path, 0.82));
      flash("Imagery layer added");
    } catch (cause) {
      fail(cause);
    } finally {
      setBusy(false);
    }
  }, [fail, flash]);

  // A layer's id is minted here: the command refuses a duplicate id, so a
  // second elevation ramp (or plate, or label layer) needs its own.
  const addCartographyLayer = useCallback(
    async (layer: CartographyLayer, index?: number) => {
      const added = { ...layer, id: layer.id || crypto.randomUUID() };
      await applyCommand({ type: "addCartographyLayer", layer: added, index });
      setSelectedLayerId(added.id);
    },
    [applyCommand],
  );

  const addRampLayer = useCallback(
    async (colormap: string) => {
      await addCartographyLayer({
        id: crypto.randomUUID(),
        name: "Elevation ramp",
        visible: true,
        type: "elevationRamp",
        colormap,
        strength: 1,
        legend: false,
      });
    },
    [addCartographyLayer],
  );

  const addPlateLayer = useCallback(
    async (title: string) => {
      await addCartographyLayer({
        id: crypto.randomUUID(),
        name: "Map plate",
        visible: true,
        type: "plate",
        title,
        northArrow: true,
        scaleBar: true,
        legend: true,
      });
    },
    [addCartographyLayer],
  );

  const addGraticuleLayer = useCallback(async () => {
    await addCartographyLayer({
      id: crypto.randomUUID(),
      name: "Graticule",
      visible: true,
      type: "graticule",
      intervalDeg: 1,
      colour: [0.1, 0.1, 0.12, 0.75],
      widthPx: 1,
      labels: true,
    });
  }, [addCartographyLayer]);

  const addLabelLayer = useCallback(
    async (source: string, field: string) => {
      await addCartographyLayer({
        id: crypto.randomUUID(),
        name: "Labels",
        visible: true,
        type: "labels",
        source,
        field,
        colour: [0.08, 0.08, 0.1, 1],
        sizePx: 14,
      });
    },
    [addCartographyLayer],
  );

  const updateCartographyLayer = useCallback(
    async (layer: CartographyLayer) => {
      await applyCommand({ type: "updateCartographyLayer", id: layer.id, layer });
    },
    [applyCommand],
  );

  const removeCartographyLayer = useCallback(
    async (id: string) => {
      await applyCommand({ type: "removeCartographyLayer", id });
      setSelectedLayerId((current) => (current === id ? null : current));
    },
    [applyCommand],
  );

  const moveCartographyLayer = useCallback(
    async (id: string, to: number) => {
      await applyCommand({ type: "moveCartographyLayer", id, to });
    },
    [applyCommand],
  );

  const applyRecipe = useCallback(
    async (id: RecipeId) => {
      const project = state?.project;
      if (!project) {
        flash("no project is open");
        return;
      }
      const recipe = findRecipe(id);
      if (!recipe) {
        flash(`unknown look recipe: ${String(id)}`);
        return;
      }
      try {
        setBusy(true);
        await applyCommand({
          type: "batch",
          commands: recipe.build(project, null),
        });
        flash(`Look applied: ${recipe.name}`);
      } catch (cause) {
        fail(cause);
      } finally {
        setBusy(false);
      }
    },
    [applyCommand, fail, flash, state],
  );

  const applyMapLook = useCallback(async () => {
    const project = state?.project;
    const terrain = project?.scene.objects.find((object) => object.kind.type === "terrain");
    if (!project || !terrain || terrain.kind.type !== "terrain") {
      flash("This project has no terrain to frame");
      return;
    }
    try {
      setBusy(true);
      await applyCommand({ type: "applyMapLook", id: terrain.id });
      // The map framing lives on the scene camera, so the viewport has to look
      // through it rather than keep its own orbit.
      setCameraView(true);
      flash("Map look applied: relief fitted, north-west sun, hillshade on");
    } catch (cause) {
      fail(cause);
    } finally {
      setBusy(false);
    }
  }, [applyCommand, fail, flash, state]);

  /**
   * Offline render queue.
   *
   * The viewport's preview scheduler is newest-wins (a preview is disposable);
   * a still or a frame sequence is a job, so it runs FIFO to completion with
   * honest progress and a cancel that stops between frames. Each job captures
   * the render size when it is queued, so a later settings edit cannot
   * retarget work the user already started.
   */
  const commitJobs = useCallback((next: RenderJob[]) => {
    jobsRef.current = next;
    setRenderJobs(next);
  }, []);

  const updateJob = useCallback(
    (id: string, patch: Partial<Omit<RenderJob, "id">>) => {
      commitJobs(patchJob(jobsRef.current, id, patch));
    },
    [commitJobs],
  );

  const pumpJobs = useCallback(async () => {
    if (queueBusy.current) return;
    queueBusy.current = true;
    try {
      for (;;) {
        const job = nextQueuedJob(jobsRef.current);
        if (!job) break;
        updateJob(job.id, { status: "running" });
        try {
          if (job.kind === "still") {
            const frame = await bridge.renderStill(
              job.output || null,
              viewStore.current ?? undefined,
            );
            // A still cannot be interrupted mid-frame; a cancel requested
            // while it ran still writes the file, so report the truth.
            const cancelled = cancelledJobs.current.has(job.id);
            updateJob(job.id, {
              status: cancelled ? "cancelled" : "done",
              done: 1,
              output: frame.path ?? job.output,
            });
            flash(frame.path ? `Saved ${frame.path}` : "Rendered");
          } else {
            const spec = jobSpecs.current.get(job.id);
            const width = spec?.width ?? 1920;
            const height = spec?.height ?? 1080;
            let index = 0;
            for (; index < job.total; index += 1) {
              if (cancelledJobs.current.has(job.id)) break;
              const output = `${job.output}/frame_${String(index).padStart(4, "0")}.png`;
              await bridge.renderAnimationFrame({
                frame: index,
                width,
                height,
                quality: "final",
                output,
              });
              updateJob(job.id, { done: index + 1 });
            }
            if (index < job.total) {
              updateJob(job.id, { status: "cancelled" });
              flash(`Export stopped at frame ${index} of ${job.total}`);
            } else {
              updateJob(job.id, { status: "done" });
              flash(`Exported ${job.total} frames to ${job.output}`);
            }
          }
        } catch (cause) {
          // A frame that fails because the job was cancelled (or its project
          // closed under it) is a cancellation, not an error to shout about.
          const cancelled = cancelledJobs.current.has(job.id);
          updateJob(job.id, {
            status: cancelled ? "cancelled" : "failed",
            error: cancelled
              ? null
              : cause instanceof Error
                ? cause.message
                : String(cause),
          });
          if (!cancelled) fail(cause);
        } finally {
          jobSpecs.current.delete(job.id);
          cancelledJobs.current.delete(job.id);
        }
      }
    } finally {
      queueBusy.current = false;
    }
  }, [fail, flash, updateJob]);

  const enqueueRenderJob = useCallback(
    (job: RenderJob, spec: { width: number; height: number }) => {
      jobSpecs.current.set(job.id, spec);
      commitJobs(enqueueJob(jobsRef.current, job));
      void pumpJobs();
    },
    [commitJobs, pumpJobs],
  );

  /** Queue one still; its path is already chosen (null in the preview). */
  const enqueueStill = useCallback(
    (path: string | null) => {
      const render = state?.project.render;
      enqueueRenderJob(
        newRenderJob({
          kind: "still",
          label: path ? `Still → ${path.split(/[\\/]/).pop()}` : "Still (preview)",
          total: 1,
          output: path ?? "",
        }),
        { width: render?.width ?? 1920, height: render?.height ?? 1080 },
      );
    },
    [enqueueRenderJob, state],
  );

  /**
   * Queue the whole timeline as a PNG sequence. The shell evaluates a clone
   * per frame — the same studio-core `evaluate` the playhead uses — so the
   * session's playhead and dirty state never move during export.
   */
  const exportFrames = useCallback(
    async (directory: string) => {
      const snapshot = state;
      if (!snapshot) return;
      const { timeline, render: renderSettings } = snapshot.project;
      const total = timeline.length + 1;
      const name = directory.split(/[\\/]/).filter(Boolean).pop() ?? "sequence";
      enqueueRenderJob(
        newRenderJob({
          kind: "sequence",
          label: `Sequence · ${total} frames → ${name}`,
          total,
          output: directory,
        }),
        { width: renderSettings.width, height: renderSettings.height },
      );
    },
    [enqueueRenderJob, state],
  );

  /** Stop the running sequence and drop anything queued behind it. */
  const cancelExport = useCallback(() => {
    for (const job of jobsRef.current) {
      if (job.kind !== "sequence") continue;
      if (job.status === "running") {
        cancelledJobs.current.add(job.id);
        flash("Stopping after the current frame…");
      } else if (job.status === "queued") {
        // Also raise the flag: the worker may have picked it up between our
        // read and this write, in which case it stops after the current frame.
        cancelledJobs.current.add(job.id);
        updateJob(job.id, { status: "cancelled" });
      }
    }
  }, [flash, updateJob]);

  const cancelRenderJob = useCallback(
    (id: string) => {
      const job = jobsRef.current.find((entry) => entry.id === id);
      if (!job) return;
      if (job.status === "queued") {
        // Same race as cancelExport: the worker may already be inside it.
        cancelledJobs.current.add(id);
        updateJob(id, { status: "cancelled" });
        return;
      }
      if (job.status === "running" && canCancelJob(job)) {
        cancelledJobs.current.add(id);
        flash("Stopping after the current frame…");
      }
    },
    [flash, updateJob],
  );

  const clearFinishedJobs = useCallback(() => {
    commitJobs(dropFinishedJobs(jobsRef.current));
  }, [commitJobs]);

  // Closing a project abandons its offline work: every job reads the session's
  // project, so there is nothing left to render.
  useEffect(() => {
    if (hasProject) return;
    for (const job of jobsRef.current) {
      if (job.status === "queued") updateJob(job.id, { status: "cancelled" });
      else if (job.status === "running") cancelledJobs.current.add(job.id);
    }
  }, [hasProject, updateJob]);

  const runningSequence = renderJobs.find(
    (job) => job.kind === "sequence" && job.status === "running",
  );
  const exporting = runningSequence
    ? { done: runningSequence.done, total: runningSequence.total }
    : null;

  const setViewportMode = useCallback(
    async (mode: ViewportMode) => {
      const project = state?.project;
      if (!project) return;
      await applyCommand({
        type: "setViewport",
        viewport: { ...project.viewport, mode },
      });
    },
    [applyCommand, state],
  );

  const setHillshade = useCallback(
    async (hillshade: boolean) => {
      const project = state?.project;
      if (!project) return;
      await applyCommand({
        type: "setViewport",
        viewport: { ...project.viewport, hillshade },
      });
    },
    [applyCommand, state],
  );

  const actions = useMemo<SessionApi["actions"]>(
    () => ({
      selectCartographyLayer(id: string | null) {
        setSelectedLayerId(id);
      },

      async newProject() {
        // Kept for programmatic use (keyboard shortcut opens the dialog instead).
        await actions.createProject(null);
      },

      requestNewProject() {
        if (!guardUnsaved("start a new project")) return;
        if (presets.length === 0) {
          bridge
            .scenePresets()
            .then(setPresets)
            .catch(() => undefined);
        }
        setNewProjectOpen(true);
      },

      dismissNewProject() {
        setNewProjectOpen(false);
      },

      async createProject(presetId) {
        setBusy(true);
        setNewProjectOpen(false);
        try {
          const snapshot = await bridge.newProject(null, presetId);
          setState(snapshot);
          await refreshRecovery();
          flash("New project");
        } catch (cause) {
          fail(cause);
        } finally {
          setBusy(false);
        }
      },

      async openProject() {
        if (!guardUnsaved("open another project")) return;
        try {
          const picked = await openDialog({
            multiple: false,
            filters: [PROJECT_FILTER],
          });
          const path = typeof picked === "string" ? picked : null;
          if (!path) return;
          setBusy(true);
          setState(await bridge.openProject(path));
          await refreshRecovery();
          flash("Project opened");
        } catch (cause) {
          fail(cause);
        } finally {
          setBusy(false);
        }
      },

      async renderStill() {
        // The still goes where the user points the save dialog; the folder
        // defaults to <project>/renders/<name>-<timestamp>.png so a render
        // can always be traced to its settings (JSON sidecar next to it).
        // It is queued like every offline render, so one worker owns the
        // engine and progress/cancel are uniform.
        if (!state) return;
        const name = (state.project.name || "untitled").replace(/[^A-Za-z0-9-_]+/g, "-");
        const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
        const slash = Math.max(
          state.path?.lastIndexOf("/") ?? -1,
          state.path?.lastIndexOf("\\") ?? -1,
        );
        const dir = state.path && slash > 0 ? state.path.slice(0, slash) : null;
        const picked = await saveDialog({
          defaultPath: dir
            ? `${dir}/renders/${name}-${stamp}.png`
            : `${name}-${stamp}.png`,
          filters: [{ name: "PNG image", extensions: ["png"] }],
        });
        if (!picked) return;
        enqueueStill(picked);
      },

      exportFrames,
      cancelExport,
      cancelRenderJob,
      clearFinishedJobs,

      async saveProject(saveAs = false) {
        if (!state) return;
        try {
          let path: string | null = null;
          if (saveAs || !state.path) {
            const picked = await saveDialog({
              defaultPath: `${state.project.name || "untitled"}.studio3d`,
              filters: [PROJECT_FILTER],
            });
            if (!picked) return;
            path = picked;
          }
          setBusy(true);
          const saved = await bridge.saveProject(path);
          setState(saved);
          await refreshRecovery();
          flash(`Saved ${saved.path ?? "project"}`);
        } catch (cause) {
          fail(cause);
        } finally {
          setBusy(false);
        }
      },

      async closeProject() {
        if (!guardUnsaved("close the project")) return;
        setBusy(true);
        try {
          setState(await bridge.closeProject());
          flash("Project closed");
        } catch (cause) {
          fail(cause);
        } finally {
          setBusy(false);
        }
      },

      apply: applyCommand,

      async undo() {
        try {
          setState((await bridge.undo()).state);
        } catch (cause) {
          fail(cause);
        }
      },

      async redo() {
        try {
          setState((await bridge.redo()).state);
        } catch (cause) {
          fail(cause);
        }
      },

      select,
      setPlayhead,
      setSunFromSolar,
      addPrimitive,
      duplicateSelection,
      deleteSelection,
      importMesh,

      refreshEngine,
      importMaterialTexture,
      clearMaterialTexture,
      importEnvironment,
      clearEnvironment,
      relinkAsset,
      refreshAssetIntegrity,
      importTerrainDem,
      importTerrainDemByName,
      addImageryLayerByName,
      addVectorLayerByName,
      addVectorLayer,
      addPolygonLayer,
      addImageryLayer,
      addCartographyLayer,
      addRampLayer,
      addPlateLayer,
      addGraticuleLayer,
      addLabelLayer,
      updateCartographyLayer,
      removeCartographyLayer,
      moveCartographyLayer,
      applyMapLook,
      applyRecipe,
      setViewportMode,
      setHillshade,
      renderViewport,
      clearRenderFrame,

      async restoreRecovery(id) {
        setBusy(true);
        try {
          setState(await bridge.restoreRecovery(id));
          await refreshRecovery();
          flash("Recovered unsaved work");
        } catch (cause) {
          fail(cause);
        } finally {
          setBusy(false);
        }
      },

      async discardRecovery(id) {
        try {
          await bridge.discardRecovery(id);
          await refreshRecovery();
          flash("Recovery snapshot discarded");
        } catch (cause) {
          fail(cause);
        }
      },

      dismissError() {
        setError(null);
      },

      notify(message) {
        flash(message);
      },
    }),
    [
      addPrimitive,
      applyCommand,
      applyMapLook,
      applyRecipe,
      deleteSelection,
      duplicateSelection,
      fail,
      flash,
      guardUnsaved,
      importMesh,
      importMaterialTexture,
      clearMaterialTexture,
      importEnvironment,
      clearEnvironment,
      relinkAsset,
      refreshAssetIntegrity,
      importTerrainDem,
      importTerrainDemByName,
      addImageryLayerByName,
      addVectorLayerByName,
      addVectorLayer,
      addPolygonLayer,
      addImageryLayer,
      addCartographyLayer,
      addRampLayer,
      addPlateLayer,
      addGraticuleLayer,
      addLabelLayer,
      updateCartographyLayer,
      removeCartographyLayer,
      moveCartographyLayer,
      refreshEngine,
      refreshRecovery,
      enqueueStill,
      renderViewport,
      clearRenderFrame,
      exportFrames,
      cancelExport,
      cancelRenderJob,
      clearFinishedJobs,
      select,
      setPlayhead,
      setSunFromSolar,
      setViewportMode,
      setHillshade,
      state,
    ],
  );

  const value: SessionApi = {
    presets,
    newProjectOpen,
    info,
    state,
    autosave,
    recovery,
    error,
    notice,
    busy,
    tool,
    setTool,
    selectedLayerId,
    space,
    setSpace,
    cameraView,
    setCameraView,
    flyMode,
    setFlyMode,
    engine,
    render,
    exporting,
    assetIntegrity,
    renderJobs,
    terrainResolution,
    setTerrainResolution,
    actions,
  };

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

