import { useEffect, useRef, useState } from "react";
import type {
  Asset,
  CapabilityLevel,
  CapabilityTable,
  Material,
  ParameterCapability,
  CameraObject,
  LightKind,
  LightObject,
  Primitive,
  RenderSettings,
  SceneObject,
  SolarResult,
  Transform,
} from "../bridge/types";
import { objectKindLabel } from "../bridge/types";
import { bridge } from "../bridge/env";
import { base64ToArrayBuffer, listGltfNodes } from "../viewport/assets";
import { useSession } from "../state/session-context";
import { CartographyProperties } from "./CartographyProperties";
import { ChoiceField, NumberField, RangeField, Section, TextField, Toggle } from "./fields";
import type { CloudSettings } from "../bridge/types";

// ---------------------------------------------------------------- capability

let capabilityTablePromise: Promise<CapabilityTable | null> | null = null;

/** The shell's static capability table, fetched once per app run. */
function useCapabilities(): CapabilityTable | null {
  const [table, setTable] = useState<CapabilityTable | null>(null);
  useEffect(() => {
    capabilityTablePromise ??= bridge.capabilityTable().catch(() => null);
    let live = true;
    void capabilityTablePromise.then((value) => {
      if (live) setTable(value);
    });
    return () => {
      live = false;
    };
  }, []);
  return table;
}

const LEVEL_GLYPH: Record<CapabilityLevel, string> = {
  full: "✓",
  partial: "~",
  unsupported: "✗",
};

/** Per-parameter support badges: glyph = level, hover = what it drives.
 *  Phase 5's "capability-tagged parameters", straight from studio-core. */
function CapabilityBadges({ rows }: { rows: ParameterCapability[] | undefined }) {
  if (!rows || rows.length === 0) return null;
  return (
    <div className="capability-badges">
      {rows.map((row) => (
        <span
          key={row.parameter}
          className="badge"
          title={`${LEVEL_GLYPH[row.level]} ${row.level} — ${row.detail}`}
        >
          {LEVEL_GLYPH[row.level]} {row.parameter}
        </span>
      ))}
    </div>
  );
}

function TransformEditor({
  object,
  transform,
  disabled,
}: {
  object: SceneObject;
  transform: Transform;
  disabled: boolean;
}) {
  const { actions } = useSession();

  const commit = (next: Transform) => {
    void actions.apply({ type: "setTransform", id: object.id, transform: next });
  };

  const axis = (
    key: "translation" | "rotationDeg" | "scale",
    field: "x" | "y" | "z",
  ) => (value: number) =>
    commit({ ...transform, [key]: { ...transform[key], [field]: value } });

  return (
    <>
      <Section title="Transform">
        <div className="triple">
          <NumberField label="X" value={transform.translation.x} disabled={disabled} onCommit={axis("translation", "x")} />
          <NumberField label="Y" value={transform.translation.y} disabled={disabled} onCommit={axis("translation", "y")} />
          <NumberField label="Z" value={transform.translation.z} disabled={disabled} onCommit={axis("translation", "z")} />
        </div>
        <span className="group-label">Translation</span>

        <div className="triple">
          <NumberField label="X" value={transform.rotationDeg.x} disabled={disabled} onCommit={axis("rotationDeg", "x")} />
          <NumberField label="Y" value={transform.rotationDeg.y} disabled={disabled} onCommit={axis("rotationDeg", "y")} />
          <NumberField label="Z" value={transform.rotationDeg.z} disabled={disabled} onCommit={axis("rotationDeg", "z")} />
        </div>
        <span className="group-label">Rotation (degrees)</span>

        <div className="triple">
          <NumberField label="X" value={transform.scale.x} step={0.05} disabled={disabled} onCommit={axis("scale", "x")} />
          <NumberField label="Y" value={transform.scale.y} step={0.05} disabled={disabled} onCommit={axis("scale", "y")} />
          <NumberField label="Z" value={transform.scale.z} step={0.05} disabled={disabled} onCommit={axis("scale", "z")} />
        </div>
        <span className="group-label">Scale</span>
        {object.kind.type === "terrain" && (
          <p className="hint">
            Terrain is georeferenced — its position and rotation are fixed so
            the Solid view and the renderer frame the same ground. Use Vertical
            scale (Terrain section) for relief.
          </p>
        )}
      </Section>

      <Section title="Object">
        <Toggle
          label="Visible"
          checked={object.visible}
          onChange={(visible) =>
            void actions.apply({ type: "setVisibility", id: object.id, visible })
          }
        />
        <Toggle
          label="Locked"
          checked={object.locked}
          onChange={(locked) =>
            void actions.apply({ type: "setLocked", id: object.id, locked })
          }
        />
      </Section>
    </>
  );
}

function MeshEditor({ object, primitive }: { object: SceneObject; primitive: Primitive }) {
  const { actions } = useSession();
  const extent = primitive.type === "sphere" ? primitive.radius : primitive.size;

  const setShape = (shape: string) => {
    const withShape: Primitive =
      shape === "sphere"
        ? { type: "sphere", radius: extent }
        : shape === "plane"
          ? { type: "plane", size: extent }
          : { type: "cube", size: extent };
    void actions.apply({ type: "setPrimitive", id: object.id, primitive: withShape });
  };

  const setExtent = (value: number) => {
    const next: Primitive =
      primitive.type === "sphere"
        ? { type: "sphere", radius: value }
        : primitive.type === "plane"
          ? { type: "plane", size: value }
          : { type: "cube", size: value };
    void actions.apply({ type: "setPrimitive", id: object.id, primitive: next });
  };

  return (
    <Section title="Mesh">
      <label className="field">
        <span className="field-label">Primitive</span>
        <select
          className="input input--select"
          value={primitive.type}
          onChange={(event) => setShape(event.target.value)}
        >
          <option value="cube">Cube</option>
          <option value="sphere">Sphere</option>
          <option value="plane">Plane</option>
        </select>
      </label>
      <NumberField
        label={primitive.type === "sphere" ? "Radius" : "Size"}
        value={extent}
        step={0.1}
        min={0.01}
        onCommit={setExtent}
      />
    </Section>
  );
}

function CameraEditor({
  object,
  camera,
  active,
}: {
  object: SceneObject;
  camera: CameraObject;
  active: boolean;
}) {
  const { actions } = useSession();
  const caps = useCapabilities();
  const commit = (next: CameraObject) =>
    void actions.apply({ type: "setCamera", id: object.id, camera: next });

  return (
    <Section
      title="Camera"
      actions={
        active ? (
          <span className="badge badge--active">active</span>
        ) : (
          <button
            type="button"
            className="text-button"
            onClick={() => void actions.apply({ type: "setActiveCamera", id: object.id })}
          >
            Set active
          </button>
        )
      }
    >
      <NumberField
        label="Field of view"
        value={camera.fovDeg}
        step={1}
        min={1}
        max={179}
        onCommit={(fovDeg) => commit({ ...camera, fovDeg })}
      />
      <div className="pair">
        <NumberField
          label="Near"
          value={camera.near}
          step={0.05}
          min={0.001}
          onCommit={(near) => commit({ ...camera, near })}
        />
        <NumberField
          label="Far"
          value={camera.far}
          step={10}
          onCommit={(far) => commit({ ...camera, far })}
        />
      </div>
      <CapabilityBadges rows={caps?.camera} />
    </Section>
  );
}

function localDateTimeInput(): string {
  const now = new Date();
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}T${pad(
    now.getHours(),
  )}:${pad(now.getMinutes())}`;
}

/**
 * Bakes the engine's solar ephemeris into the light's rotation. The datetime
 * input is local; the browser converts it to the UTC string the engine wants.
 * Nothing is stored: re-applying a different time is the way to change it.
 */
function SolarPositionEditor({ object }: { object: SceneObject }) {
  const { actions } = useSession();
  const [latitude, setLatitude] = useState(0);
  const [longitude, setLongitude] = useState(0);
  const [when, setWhen] = useState(localDateTimeInput);
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<SolarResult | null>(null);

  const apply = async () => {
    const local = new Date(when);
    if (Number.isNaN(local.getTime())) {
      actions.notify("Enter a valid date and time");
      return;
    }
    setPending(true);
    try {
      const utc = local.toISOString();
      const applied = await actions.setSunFromSolar(
        object.id,
        latitude,
        longitude,
        utc,
      );
      if (applied) setResult(applied);
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="solar-position">
      <span className="group-label">Sun from date &amp; place</span>
      <div className="pair">
        <NumberField
          label="Latitude"
          value={latitude}
          step={0.1}
          min={-90}
          max={90}
          onCommit={setLatitude}
        />
        <NumberField
          label="Longitude"
          value={longitude}
          step={0.1}
          min={-180}
          max={180}
          onCommit={setLongitude}
        />
      </div>
      <label className="field">
        <span className="field-label">Local date &amp; time</span>
        <input
          className="input"
          type="datetime-local"
          value={when}
          onChange={(event) => setWhen(event.target.value)}
        />
      </label>
      <button
        type="button"
        className="button button--small"
        disabled={pending || !when}
        title="Rotate the sun to this place and time — one undo step"
        onClick={() => void apply()}
      >
        {pending ? "Locating…" : "Set sun"}
      </button>
      {result && (
        <p className="hint">
          Azimuth {result.azimuthDeg.toFixed(1)}° · elevation{" "}
          {result.elevationDeg.toFixed(1)}°
          {!result.daytime && " · below the horizon"}
        </p>
      )}
    </div>
  );
}

function LightEditor({ object, light }: { object: SceneObject; light: LightObject }) {
  const { actions } = useSession();
  const caps = useCapabilities();
  const commit = (next: LightObject) =>
    void actions.apply({ type: "setLight", id: object.id, light: next });

  const channel = (index: 0 | 1 | 2) => (value: number) => {
    const color: [number, number, number] = [...light.color] as [number, number, number];
    color[index] = value;
    commit({ ...light, color });
  };

  const rotation = object.transform.rotationDeg;
  const setRotation = (axis: "x" | "y" | "z", value: number) =>
    void actions.apply({
      type: "setTransform",
      id: object.id,
      transform: {
        ...object.transform,
        rotationDeg: { ...rotation, [axis]: value },
      },
    });

  return (
    <Section title="Sun" id="light-sun">
      <label className="field">
        <span className="field-label">Type</span>
        <select
          className="input input--select"
          value={light.kind}
          title="Both views light with a directional sun; point lights are not implemented in either view"
          onChange={(event) => commit({ ...light, kind: event.target.value as LightKind })}
        >
          <option value="directional">Directional</option>
          {/* Neither the Solid view nor the renderer implements point lights;
              creating one would silently diverge the two (renders fall back
              to the default sun). Kept selectable *from* for legacy scenes. */}
          <option value="point" disabled>Point — unsupported</option>
        </select>
      </label>
      <NumberField
        label="Intensity"
        value={light.intensity}
        step={0.1}
        min={0}
        onCommit={(intensity) => commit({ ...light, intensity })}
      />
      <div className="triple">
        <NumberField label="R" value={light.color[0]} step={0.02} min={0} max={1} onCommit={channel(0)} />
        <NumberField label="G" value={light.color[1]} step={0.02} min={0} max={1} onCommit={channel(1)} />
        <NumberField label="B" value={light.color[2]} step={0.02} min={0} max={1} onCommit={channel(2)} />
      </div>
      <span className="group-label">Colour (linear RGB)</span>
      <div className="triple">
        <NumberField label="Pitch" value={rotation.x} step={1} min={-90} max={90} disabled={light.kind !== "directional"} onCommit={(value) => setRotation("x", value)} />
        <NumberField label="Yaw" value={rotation.y} step={1} min={-180} max={180} disabled={light.kind !== "directional"} onCommit={(value) => setRotation("y", value)} />
        <NumberField label="Roll" value={rotation.z} step={1} min={-180} max={180} disabled={light.kind !== "directional"} onCommit={(value) => setRotation("z", value)} />
      </div>
      <span className="group-label">Direction (degrees)</span>
      {light.kind === "directional" && <SolarPositionEditor object={object} />}
      <CapabilityBadges rows={caps?.light} />
    </Section>
  );
}

function AssetsPanel() {
  const { state, actions, assetIntegrity } = useSession();
  const project = state?.project;
  if (!project || project.assets.length === 0) return null;
  const integrity = new Map(assetIntegrity.map((entry) => [entry.id, entry]));
  const problems = assetIntegrity.filter(
    (entry) => entry.actualBytes === null || entry.actualBytes !== entry.expectedBytes,
  );
  return (
    <Section title="Assets">
      {problems.length > 0 && (
        <p className="hint hint--boxed">
          {problems.length === 1
            ? "1 referenced file needs attention."
            : `${problems.length} referenced files need attention.`}{" "}
          Missing files keep their path; relink them before rendering.
        </p>
      )}
      <ul className="multi-list">
        {project.assets.map((asset) => {
          const usage =
            project.scene.objects.filter(
              (object) =>
                (object.kind.type === "mesh" &&
                  (object.kind.source.type === "gltf"
                    ? object.kind.source.asset === asset.id
                    : object.kind.material.texture === asset.id)) ||
                (object.kind.type === "terrain" && object.kind.asset === asset.id) ||
                project.cartography.some(
                  (layer) => layer.type !== "relief" && layer.asset === asset.id,
                ),
            ).length + (project.render.environment === asset.id ? 1 : 0);
          const check = integrity.get(asset.id);
          const exists = check ? check.actualBytes !== null : true;
          const sizeMatches = check
            ? check.actualBytes === check.expectedBytes
            : true;
          return (
            <li key={asset.id}>
              <span title={check?.error ?? asset.path}>{asset.name}</span>
              {check && !exists && (
                <span className="badge badge--missing" title={check.error ?? undefined}>
                  missing
                </span>
              )}
              {check && exists && !sizeMatches && (
                <span
                  className="badge badge--changed"
                  title={`recorded ${check.expectedBytes} bytes, found ${check.actualBytes}`}
                >
                  changed
                </span>
              )}
              <span className="mono">
                {usage} use{usage === 1 ? "" : "s"}
              </span>
              <button
                type="button"
                className="text-button"
                onClick={() => void actions.relinkAsset(asset.id)}
              >
                Relink…
              </button>
            </li>
          );
        })}
      </ul>
      <p className="hint">
        Files are checked when the project opens and whenever an asset is
        imported or relinked. Relinking points the asset at a new path without
        touching scene references.
      </p>
    </Section>
  );
}

/** The project's render settings plus a whole-object commit (every settings
 *  edit is one command, like `setChannelKeys`). */
function useRenderSettings() {
  const { state, actions } = useSession();
  const latest = useRef(state?.project.render);
  latest.current = state?.project.render;
  if (!state || !latest.current) return null;
  const commit = (patch: Partial<RenderSettings>) => {
    // Two edits can land before the first command returns; build the next
    // settings on what this hook last sent, not on the snapshot React has not
    // seen yet (otherwise the earlier edit is silently dropped).
    const next = { ...latest.current!, ...patch };
    latest.current = next;
    void actions.apply({ type: "setRenderSettings", render: next });
  };
  return { project: state.project, render: latest.current, commit };
}

/** Render tab: what the still and the Rendered frame are made of. */
export function RenderSettingsEditor() {
  const { engine } = useSession();
  const settings = useRenderSettings();
  if (!settings) return null;
  const { render, commit } = settings;

  return (
    <>
      <Section title="Output" id="render-output">
        <div className="pair">
          {/* The render service caps a frame at 4096 px per side, so the
              field must not offer a size the renderer will refuse. */}
          <NumberField label="Width" value={render.width} step={16} min={64} max={4096} onCommit={(width) => commit({ width: Math.round(width) })} />
          <NumberField label="Height" value={render.height} step={16} min={64} max={4096} onCommit={(height) => commit({ height: Math.round(height) })} />
        </div>
        <NumberField label="Exposure" value={render.exposure} step={0.05} min={0.05} max={8} onCommit={(exposure) => commit({ exposure })} />
        <p className="hint">
          The renderer tone-maps with ACES at this exposure; the previews use
          the same value, so a still matches what you were looking at.
        </p>
      </Section>

      <Section title="Shadows" id="render-shadows">
        <Toggle label="Sun shadows" checked={render.shadows} onChange={(shadows) => commit({ shadows })} />
        <ChoiceField
          label="Filter"
          value={render.shadowTechnique}
          options={[
            { value: "pcf", label: "PCF", title: "Fixed-width filter: crisp and cheap" },
            { value: "pcss", label: "PCSS", title: "Blocker search: softer contact shadows, roughly 2× the cost" },
          ]}
          onChange={(shadowTechnique) => commit({ shadowTechnique })}
        />
        <ChoiceField
          label="Map size"
          value={String(render.shadowResolution)}
          options={[512, 1024, 2048, 4096].map((size) => ({
            value: String(size),
            label: String(size),
            title: `${size}² shadow map: ${size < 2048 ? "faster, chunkier edges" : "crisper, more memory and fill"}`,
          }))}
          onChange={(value) => commit({ shadowResolution: Number(value) })}
          hint="Applies to previews too: the Rendered viewport re-renders on orbit, so a big map costs frame time."
        />
        <Toggle
          label="Terrain sun shadows"
          checked={render.sunVisibility}
          onChange={(sunVisibility) => commit({ sunVisibility })}
        />
        {render.sunVisibility && (
          <>
            <NumberField
              label="Sun reach (m)"
              value={render.sunVisibilityDistance}
              step={100}
              min={100}
              max={20000}
              onCommit={(sunVisibilityDistance) =>
                commit({ sunVisibilityDistance })
              }
            />
            <ChoiceField
              label="Ray steps"
              value={String(render.sunVisibilitySteps)}
              options={[16, 32, 48, 64].map((steps) => ({
                value: String(steps),
                label: String(steps),
                title: `${steps} steps per sun ray: more follows a ridge line further before missing it`,
              }))}
              onChange={(value) =>
                commit({ sunVisibilitySteps: Number(value) })
              }
              hint="A ray march per pixel over the heightfield: a ridge finally shadows the valley behind it. The preview bakes the same march once per DEM change; it rebuilds when you move the sun."
            />
          </>
        )}
      </Section>

      <Section title="Post" id="render-post">
        <Toggle
          label="Bloom the highlights"
          checked={render.bloom.enabled}
          onChange={(enabled) => commit({ bloom: { ...render.bloom, enabled } })}
        />
        {render.bloom.enabled && (
          <>
            <NumberField
              label="Threshold"
              value={render.bloom.threshold}
              step={0.05}
              min={0}
              max={1}
              onCommit={(threshold) =>
                commit({ bloom: { ...render.bloom, threshold } })
              }
            />
            <NumberField
              label="Softness"
              value={render.bloom.softness}
              step={0.05}
              min={0}
              max={1}
              onCommit={(softness) =>
                commit({ bloom: { ...render.bloom, softness } })
              }
            />
            <NumberField
              label="Intensity"
              value={render.bloom.intensity}
              step={0.05}
              min={0}
              max={2}
              onCommit={(intensity) =>
                commit({ bloom: { ...render.bloom, intensity } })
              }
            />
          </>
        )}
        <NumberField
          label="Vignette"
          value={render.vignette}
          step={0.05}
          min={0}
          max={1}
          onCommit={(vignette) => commit({ vignette })}
        />
        <p className="hint">
          Bloom and vignette are applied to the finished frame, after the
          engine's tone-map. The Rendered preview and the still show them; the
          Solid and Material previews do not — like AOVs and denoise.
        </p>
      </Section>

      <Section title="Stills" id="render-stills">
        <Toggle
          label="AOVs next to the still (albedo · normal · depth)"
          checked={render.stillAov.albedo || render.stillAov.normal || render.stillAov.depth}
          onChange={(value) =>
            commit({ stillAov: { albedo: value, normal: value, depth: value } })
          }
        />
        <Toggle
          label="Denoise the still with OIDN"
          checked={render.stillDenoise}
          onChange={(stillDenoise) => commit({ stillDenoise })}
        />
        {render.stillDenoise && engine?.available && engine.denoise === false && (
          <p className="hint hint--boxed">
            The engine's interpreter has no OIDN runtime, so a denoised still
            fails instead of rendering. Install it with{" "}
            <code>pip install -e .[denoise]</code> for{" "}
            <code>{engine.python}</code>.
          </p>
        )}
        <p className="hint">
          Both apply to sequence frames as well (they are final renders) and to
          viewport previews never.
        </p>

      </Section>
      <Section title="Fog" id="atmosphere-fog">
        <Toggle
          label="Haze"
          checked={render.fog.enabled}
          onChange={(enabled) => commit({ fog: { ...render.fog, enabled } })}
        />
        {render.fog.enabled && (
          <>
            <NumberField
              label="Density"
              value={render.fog.density}
              step={0.05}
              min={0}
              max={100}
              onCommit={(density) => commit({ fog: { ...render.fog, density } })}
            />
            <NumberField
              label="Reach"
              value={render.fog.reach}
              step={0.05}
              min={0.01}
              max={1}
              onCommit={(reach) => commit({ fog: { ...render.fog, reach } })}
            />
            {/* The engine scatters the sky's colour when the sky pass is on
                (`select(fog_inscatter, sky_color, sky_enabled)`), so a tint
                would be a control that changes nothing in the default state:
                measured 0 samples with the sky on, 30,127 with it off. */}
            {render.sky ? (
              <p className="hint">
                The haze takes its colour from the analytic sky. Turn the sky
                background off (Lighting → Sky) to tint the fog itself.
              </p>
            ) : (
              <>
                <div className="pair">
                  {(["R", "G", "B"] as const).map((channel, index) => (
                    <NumberField
                      key={channel}
                      label={channel}
                      value={render.fog.inscatter[index]}
                      step={0.05}
                      min={0}
                      max={1}
                      onCommit={(value) => {
                        const inscatter = [...render.fog.inscatter] as [
                          number,
                          number,
                          number,
                        ];
                        inscatter[index] = value;
                        commit({ fog: { ...render.fog, inscatter } });
                      }}
                    />
                  ))}
                </div>
                <p className="hint">
                  Inscatter is the colour the haze scatters: white reads as
                  neutral mist, a warm tint as dust, a cool one as sea air.
                </p>
              </>
            )}
            <p className="hint">
              Height-based fog, applied after the lighting and before the
              tone-map. Reach is how far up the terrain the haze climbs, as a
              fraction of its height — 0.5 is halfway up the mountain.
            </p>
          </>
        )}
      </Section>

      <Section title="Clouds" id="atmosphere-clouds">
        <Toggle
          label="Clouds"
          checked={render.clouds.enabled}
          onChange={(enabled) =>
            commit({ clouds: { ...render.clouds, enabled } })
          }
        />
        {render.clouds.enabled && (
          <>
            <NumberField
              label="Coverage"
              value={render.clouds.coverage}
              step={0.05}
              min={0}
              max={1}
              onCommit={(coverage) =>
                commit({ clouds: { ...render.clouds, coverage } })
              }
            />
            <NumberField
              label="Density"
              value={render.clouds.density}
              step={0.05}
              min={0}
              max={1}
              onCommit={(density) =>
                commit({ clouds: { ...render.clouds, density } })
              }
            />
            <NumberField
              label="Shadow strength"
              value={render.clouds.shadowStrength}
              step={0.05}
              min={0}
              max={1}
              onCommit={(shadowStrength) =>
                commit({ clouds: { ...render.clouds, shadowStrength } })
              }
            />
            <ChoiceField
              label="Field resolution"
              value={render.clouds.quality}
              options={[
                { value: "low", label: "Low" },
                { value: "medium", label: "Medium" },
                { value: "high", label: "High" },
                { value: "ultra", label: "Ultra" },
              ]}
              onChange={(quality) =>
                commit({
                  clouds: { ...render.clouds, quality: quality as CloudSettings["quality"] },
                })
              }
            />
            <p className="hint">
              The engine&apos;s own cloud field, projected along the sun onto
              the ground. The shadows land on the terrain, so a ridge keeps its
              own light. Measured on the Mount Fuji DEM: the clouds change
              368,996 samples (mean 0.498).
            </p>
          </>
        )}
      </Section>
    </>
  );
}

/**
 * Atmosphere: the weather on the terrain.
 *
 * Fog and water live here rather than in the Render tab: they are things in
 * the world (haze in the air, a lake in the valley), not properties of the
 * camera or the output. Clouds belong here too when the engine's cloud
 * shadows reach this render path.
 */
export function AtmosphereSection() {
  const settings = useRenderSettings();
  if (!settings) return null;
  const { project, render, commit } = settings;
  // The water level is an absolute elevation, so the user needs to know the
  // terrain's range to pick one.
  const terrain = project.scene.objects.find((object) => object.kind.type === "terrain");
  const terrainRange =
    terrain && terrain.kind.type === "terrain" ? terrain.kind.elevationRange : null;

  return (
    <>
      <Section title="Water" id="atmosphere-water">
        <Toggle
          label="Flood the low ground"
          checked={render.water.enabled}
          onChange={(enabled) => commit({ water: { ...render.water, enabled } })}
        />
        {render.water.enabled && (
          <>
            {terrainRange && (
              <div className="readout readout--compact">
                <dt>Terrain</dt>
                <dd>
                  {Math.round(terrainRange[0]).toLocaleString()}–{Math.round(terrainRange[1]).toLocaleString()} m
                </dd>
              </div>
            )}
            <NumberField
              label="Level (m)"
              value={render.water.level}
              step={1}
              min={0}
              onCommit={(level) => commit({ water: { ...render.water, level } })}
            />
            <NumberField
              label="Slope threshold"
              value={render.water.slopeThreshold}
              step={0.005}
              min={0}
              max={1}
              onCommit={(slopeThreshold) =>
                commit({ water: { ...render.water, slopeThreshold } })
              }
            />
            <NumberField
              label="Roughness"
              value={render.water.roughness}
              step={0.05}
              min={0}
              max={1}
              onCommit={(roughness) =>
                commit({ water: { ...render.water, roughness } })
              }
            />
            <p className="hint">
              The engine derives the water mask from the DEM: ground below the
              level that is flat enough to hold it. The level is an absolute
              elevation in the DEM's own metres, so it has to sit inside the
              terrain's range — 0 picks the lowest 15% automatically, and a
              level below the terrain falls back to that too. The slope
              threshold is metres of rise per DEM cell — steeper ground stays
              dry.
            </p>
          </>
        )}
      </Section>

    </>
  );
}

/** Light tab: the backdrop sky (render-only — the environment lights). */
export function SkySection() {
  const settings = useRenderSettings();
  if (!settings) return null;
  const { project, render, commit } = settings;

  // A dangling id (asset removed) must not lock the panel to "HDRI": the
  // renderer drops the missing map and draws the analytic sky.
  const hasHdri =
    render.environment !== null &&
    project.assets.some((asset) => asset.id === render.environment);
  const aether = render.skyModel === "aether";
  const shippedBake =
    Math.abs(render.skyTurbidity - 2) < 1e-6 && Math.abs(render.skyGroundAlbedo - 0.3) < 1e-6;

  return (
    <Section title="Sky" id="light-sky">
      <Toggle
        label="Sky background"
        checked={render.sky}
        onChange={(sky) => commit({ sky })}
      />
      {hasHdri ? (
        <>
          <div className="readout readout--compact">
            <dt>Model</dt>
            <dd>HDRI (project environment)</dd>
          </div>
          <p className="hint">
            The project's HDRI is the sky, so its model and exposure live in{" "}
            <strong>Environment</strong> below. Remove the HDRI to draw an
            analytic sky instead.
          </p>
        </>
      ) : (
        <>
          <ChoiceField
            label="Model"
            value={render.skyModel}
            options={[
              {
                value: "hosek",
                label: "Hosek",
                title: "Hosek–Wilkie: the engine's default daylight fit",
              },
              {
                value: "preetham",
                label: "Preetham",
                title: "Preetham: the older fit; reads bluer at the same exposure",
              },
              {
                value: "aether",
                label: "AETHER",
                title: shippedBake
                  ? "Spectral sky from the shipped LUT (turbidity 2.0, ground 0.3)"
                  : "AETHER needs a baked LUT for these values; the shipped bake is 2.0 / 0.3",
                disabled: !shippedBake,
              },
            ]}
            onChange={(skyModel) => commit({ skyModel })}
          />
          <RangeField
            label="Turbidity"
            value={render.skyTurbidity}
            min={1}
            max={10}
            step={0.1}
            disabled={aether}
            onCommit={(skyTurbidity) => commit({ skyTurbidity })}
          />
          <RangeField
            label="Ground albedo"
            value={render.skyGroundAlbedo}
            min={0}
            max={1}
            step={0.01}
            disabled={aether}
            onCommit={(skyGroundAlbedo) => commit({ skyGroundAlbedo })}
          />
          <NumberField
            label="Sky exposure"
            value={render.skyExposure}
            step={0.01}
            min={0}
            max={4}
            onCommit={(skyExposure) => commit({ skyExposure })}
          />
          {aether && (
            <p className="hint">
              AETHER renders from the shipped bake, so turbidity and ground
              albedo are fixed at 2.0 / 0.3. Pick another model to change them.
            </p>
          )}
          <p className="hint">
            The backdrop only: the environment map below (or the built-in studio
            sky) is what lights the terrain. Sky exposure moves the sky pass, not
            the scene.
          </p>
        </>
      )}
    </Section>
  );
}

/** Light tab: the scene's environment — the map that lights everything. */
export function EnvironmentSettings() {
  const { actions } = useSession();
  const caps = useCapabilities();
  const settings = useRenderSettings();
  if (!settings) return null;
  const { project, render, commit } = settings;
  const linked = render.environment !== null;
  const environment = linked
    ? project.assets.find((asset) => asset.id === render.environment)
    : undefined;

  return (
    <Section title="Environment" id="light-environment">
      <NumberField
        label="Environment / IBL intensity"
        value={render.iblIntensity}
        step={0.05}
        min={0}
        max={4}
        onCommit={(iblIntensity) => commit({ iblIntensity })}
      />
      <div className="readout readout--compact">
        <dt>Map</dt>
        <dd>
          {environment?.name ??
            (linked ? "missing asset — relink or remove" : "built-in studio sky")}
        </dd>
      </div>
      <div className="inline-actions">
        <button
          type="button"
          className="button button--small"
          title="Pick a Radiance .hdr for image-based lighting; both Material and Rendered use it"
          onClick={() => void actions.importEnvironment()}
        >
          {environment ? "Replace HDRI…" : "Add HDRI…"}
        </button>
        {linked && (
          <button
            type="button"
            className="button button--small"
            onClick={() => void actions.clearEnvironment()}
          >
            Remove
          </button>
        )}
      </div>
      <RangeField
        label="Background intensity"
        value={render.environmentBackgroundIntensity}
        min={0}
        max={1}
        step={0.01}
        disabled={!environment}
        onCommit={(environmentBackgroundIntensity) =>
          commit({ environmentBackgroundIntensity })
        }
      />
      {!environment ? (
        <p className="hint">
          The built-in sky gives the terrain its blue fill and the viewport its
          neutral backdrop. Add an HDRI to light and backdrop your own
          environment, and to control the backdrop brightness here.
        </p>
      ) : (
        <p className="hint">
          Rendered draws this map as its sky; Material uses it behind the scene
          and as the light. Background intensity moves only the backdrop.
        </p>
      )}
      <CapabilityBadges rows={caps?.environment} />
    </Section>
  );
}

function MaterialEditor({ object, material }: { object: SceneObject; material: Material }) {
  const { actions, state } = useSession();
  const caps = useCapabilities();
  const texture = material.texture
    ? state?.project.assets.find((asset) => asset.id === material.texture)
    : undefined;
  const commit = (next: Material) =>
    void actions.apply({ type: "setMeshMaterial", id: object.id, material: next });
  const channel = (index: 0 | 1 | 2) => (value: number) => {
    const baseColor: [number, number, number] = [...material.baseColor] as [number, number, number];
    baseColor[index] = value;
    commit({ ...material, baseColor });
  };
  return (
    <Section title="Material">
      <div className="triple">
        <NumberField label="R" value={material.baseColor[0]} step={0.02} min={0} max={1} onCommit={channel(0)} />
        <NumberField label="G" value={material.baseColor[1]} step={0.02} min={0} max={1} onCommit={channel(1)} />
        <NumberField label="B" value={material.baseColor[2]} step={0.02} min={0} max={1} onCommit={channel(2)} />
      </div>
      <span className="group-label">Base colour (linear RGB)</span>
      <div className="pair">
        <NumberField label="Roughness" value={material.roughness} step={0.05} min={0} max={1} onCommit={(roughness) => commit({ ...material, roughness })} />
        <NumberField label="Metallic" value={material.metallic} step={0.05} min={0} max={1} onCommit={(metallic) => commit({ ...material, metallic })} />
      </div>
      <div className="readout readout--compact">
        <dt>Texture</dt>
        <dd>{texture?.name ?? "none"}</dd>
      </div>
      <div className="inline-actions">
        <button
          type="button"
          className="button button--small"
          onClick={() => void actions.importMaterialTexture()}
        >
          {texture ? "Replace texture…" : "Add texture…"}
        </button>
        {texture && (
          <button
            type="button"
            className="button button--small"
            onClick={() => void actions.clearMaterialTexture()}
          >
            Remove
          </button>
        )}
      </div>
      <CapabilityBadges rows={caps?.meshMaterial} />
      <p className="hint">
        Studio renders this material; the Forge3D terrain pipeline does not
        consume it, and the capability report says so.
      </p>
    </Section>
  );
}

function TerrainEditor({
  terrain,
  asset,
}: {
  terrain: Extract<SceneObject["kind"], { type: "terrain" }>;
  asset: Asset | undefined;
}) {
  const { actions, terrainResolution, setTerrainResolution, state } = useSession();
  const hillshade = state?.project.viewport?.hillshade ?? true;
  const elevation =
    terrain.elevationRange === null
      ? "—"
      : `${terrain.elevationRange[0].toFixed(0)}–${terrain.elevationRange[1].toFixed(0)} m`;
  const footprint =
    terrain.worldSpanXz === null
      ? "—"
      : terrain.worldSpanXz.map((value) => `${(value / 1000).toFixed(2)} km`).join(" × ");
  const pixels =
    terrain.dimensions === null ? "—" : `${terrain.dimensions[0]} × ${terrain.dimensions[1]} px`;

  return (
    <Section title="Terrain">
      <div className="readout">
        <dt>DEM</dt>
        <dd>{asset?.name ?? "missing asset"}</dd>
        <dt>Grid</dt>
        <dd className="mono">{pixels}</dd>
        <dt>Footprint</dt>
        <dd className="mono">{footprint}</dd>
        <dt>Elevation</dt>
        <dd className="mono">{elevation}</dd>
      </div>

      <Toggle
        label="Hillshade (relief shading)"
        checked={hillshade}
        onChange={(value) => void actions.setHillshade(value)}
      />
      <p className="hint">
        Preview aid — the still does not bake a hillshade; it decides its own
        highlight response. Nothing here changes the output frame.
      </p>
      <label className="field">
        <span className="field-label">
          Solid mesh · {terrainResolution} × {terrainResolution}
        </span>
        <input
          className="slider"
          type="range"
          min={32}
          max={512}
          step={32}
          value={terrainResolution}
          onChange={(event) => setTerrainResolution(Number(event.target.value))}
        />
      </label>

      <div className="inline-actions">
        <button
          type="button"
          className="button button--small"
          onClick={() => void actions.importTerrainDem()}
        >
          Change DEM…
        </button>
      </div>

      <p className="hint">
        Relief, the elevation ramp and the draped imagery are in the{" "}
        <strong>Map</strong> tab — they belong to the terrain's cartography, not
        to the object's transform.
      </p>

      <p className="hint">
        The slider is the heightfield detail the Solid view builds from this DEM
        (32 = blocky, 512 = close to the source grid). The renderer always reads
        the original GeoTIFF, so Rendered is unaffected by this setting — and
        every render uses the camera you are looking through, so orbit and press
        Re-render to see it from there.
      </p>
    </Section>
  );
}


/** Map tab: the terrain's cartography — relief, elevation ramp, imagery. */
export function MapSection() {
  const { state } = useSession();
  const terrainObject = state?.project.scene.objects.find(
    (entry) => entry.kind.type === "terrain",
  );
  if (!terrainObject) {
    return (
      <p className="hint">
        This project has no terrain. Add a DEM from Object → Terrain (Change
        DEM…) to get relief, a colormap and draped imagery.
      </p>
    );
  }
  return <CartographyProperties />;
}

function ImportedMesh({ object, asset }: { object: SceneObject; asset: Asset | undefined }) {
  const { actions } = useSession();
  const [nodes, setNodes] = useState<string[] | null>(null);
  const [nodeError, setNodeError] = useState<string | null>(null);
  const source = object.kind.type === "mesh" ? object.kind.source : null;
  const current = source?.type === "gltf" ? source.node : null;

  // Node names come from the file itself (parse through the shell, cached by
  // path in assets.ts) — they are derived data, never stored in the project.
  useEffect(() => {
    if (!asset?.path) {
      setNodes(null);
      return;
    }
    let live = true;
    setNodeError(null);
    listGltfNodes(asset.path, (path) =>
      bridge.readMeshFile(path).then((payload) => base64ToArrayBuffer(payload.base64)),
    )
      .then((names) => {
        if (live) setNodes(names);
      })
      .catch((cause) => {
        if (live) {
          setNodes([]);
          setNodeError(cause instanceof Error ? cause.message : String(cause));
        }
      });
    return () => {
      live = false;
    };
  }, [asset?.path]);

  return (
    <Section title="Mesh">
      <div className="readout">
        <dt>Source</dt>
        <dd>glTF/GLB asset</dd>
        <dt>Asset</dt>
        <dd>{asset?.name ?? "missing asset"}</dd>
        <dt>File</dt>
        <dd className="mono" title={asset?.path}>
          {asset?.path ?? "—"}
        </dd>
        <dt>Size</dt>
        <dd className="mono">
          {asset ? `${(asset.byteSize / 1024).toFixed(1)} KB` : "—"}
        </dd>
      </div>
      {nodes && nodes.length > 0 && (
        <label className="field">
          <span className="field-label">Node</span>
          <select
            className="input input--select"
            value={current ?? ""}
            title="Draw the whole file, or only the selected node's subtree"
            onChange={(event) =>
              void actions.apply({
                type: "setGltfNode",
                id: object.id,
                node: event.target.value || null,
              })
            }
          >
            <option value="">Whole file</option>
            {current && !nodes.includes(current) && (
              <option value={current}>{current} (missing)</option>
            )}
            {nodes.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
      )}
      {nodeError && <p className="hint">{nodeError}</p>}
      <p className="hint">
        Imported meshes are moved, rotated, and scaled like any other object;
        Solid draws the file's own materials — Node picks which part of the
        file this object shows. A Studio-side material override for glTF
        assets is not implemented yet; asset relinking lives in the Assets
        panel.
      </p>
    </Section>
  );
}

/**
 * Object tab: the selected object's own properties. Scene-wide and
 * light-rig settings live in their own tabs, so this stays about the thing
 * that is selected.
 */
export function ObjectProperties() {
  const { state, actions } = useSession();
  if (!state) return null;

  const { project } = state;
  const selectedObjects = project.scene.objects.filter((object) =>
    project.selection.includes(object.id),
  );

  if (selectedObjects.length === 0) {
    return (
      <Section title="Object" id="object-none">
        <p className="hint">
          Nothing is selected. Pick an object in the scene tree (or the
          viewport) to edit its transform, material and kind-specific settings.
        </p>
        <p className="hint">
          Render output and shadows are in the <strong>Render</strong> tab, the
          sun and environment in <strong>Lighting</strong>, and the project
          itself in <strong>Scene</strong>.
        </p>
      </Section>
    );
  }

  if (selectedObjects.length > 1) {
    return (
      <Section title="Object" id="object-multi">
        <p className="hint">
          {selectedObjects.length} objects selected. Transform editing is
          one object at a time for now.
        </p>
        <ul className="multi-list">
          {selectedObjects.map((object) => (
            <li key={object.id}>
              <span>{object.name}</span>
              <span className="mono">{objectKindLabel(object.kind)}</span>
            </li>
          ))}
        </ul>
      </Section>
    );
  }

  const object = selectedObjects[0];
  const locked = object.locked;
  const terrainKind = object.kind.type === "terrain" ? object.kind : null;
  const terrainAsset = terrainKind
    ? project.assets.find((asset) => asset.id === terrainKind.asset)
    : undefined;
  const meshSource = object.kind.type === "mesh" ? object.kind.source : null;
  const meshAsset =
    meshSource?.type === "gltf"
      ? project.assets.find((asset) => asset.id === meshSource.asset)
      : undefined;

  return (
    <>
      <Section title="Identity" id="object-identity">
        <TextField
          label="Name"
          value={object.name}
          onCommit={(name) =>
            void actions.apply({ type: "renameObject", id: object.id, name })
          }
        />
        <div className="readout readout--compact">
          <dt>ID</dt>
          <dd className="mono">{object.id.slice(0, 8)}</dd>
          <dt>Parent</dt>
          <dd className="mono">
            {object.parent
              ? project.scene.objects.find((entry) => entry.id === object.parent)?.name ??
                "missing"
              : "scene root"}
          </dd>
        </div>
      </Section>

      <TransformEditor
        object={object}
        transform={object.transform}
        disabled={locked || object.kind.type === "terrain"}
      />

      {meshSource?.type === "primitive" && (
        <MeshEditor object={object} primitive={meshSource.primitive} />
      )}
      {/* A Studio material only reaches primitives today (the glTF loader
          draws the file's own materials), so showing the editor for glTF
          assets would be a control that does nothing — the shared material
          library owns it (README "not yet implemented"). */}
      {object.kind.type === "mesh" && meshSource?.type === "primitive" && (
        <MaterialEditor object={object} material={object.kind.material} />
      )}
      {meshSource?.type === "gltf" && (
        <ImportedMesh object={object} asset={meshAsset} />
      )}
      {object.kind.type === "terrain" && (
        <TerrainEditor terrain={object.kind} asset={terrainAsset} />
      )}
      {object.kind.type === "camera" && (
        <CameraEditor
          object={object}
          camera={object.kind}
          active={project.scene.activeCamera === object.id}
        />
      )}
      {object.kind.type === "light" && (
        <Section title="Light" id="object-light">
          <p className="hint">
            Intensity, colour, direction and solar placement are the scene's
            lighting rig — they live in the <strong>Lighting</strong> tab so the
            same sun is edited in one place.
          </p>
        </Section>
      )}
    </>
  );
}

/** Scene tab: the project and scene facts, and the asset list. */
export function ProjectFacts() {
  const { state } = useSession();
  const project = state?.project;
  if (!state || !project) return null;

  return (
    <>
      <Section title="Project" id="scene-project">
        <div className="readout">
          <dt>Name</dt>
          <dd>{project.name}</dd>
          <dt>Format</dt>
          <dd className="mono">
            {project.format} v{project.schemaVersion}
          </dd>
          <dt>Project ID</dt>
          <dd className="mono">{project.id.slice(0, 8)}</dd>
          <dt>Scene ID</dt>
          <dd className="mono">{project.scene.id.slice(0, 8)}</dd>
          <dt>Revision</dt>
          <dd className="mono">{state.revision}</dd>
          <dt>Saved</dt>
          <dd>{state.dirty ? "unsaved changes" : "saved"}</dd>
        </div>
      </Section>
      <Section title="Scene" id="scene-facts">
        <div className="readout">
          <dt>Objects</dt>
          <dd>{project.scene.objects.length}</dd>
          <dt>Meshes</dt>
          <dd>{project.scene.objects.filter((o) => o.kind.type === "mesh").length}</dd>
          <dt>Terrains</dt>
          <dd>{project.scene.objects.filter((o) => o.kind.type === "terrain").length}</dd>
          <dt>Cameras</dt>
          <dd>{project.scene.objects.filter((o) => o.kind.type === "camera").length}</dd>
          <dt>Lights</dt>
          <dd>{project.scene.objects.filter((o) => o.kind.type === "light").length}</dd>
        </div>
      </Section>
      <AssetsPanel />
    </>
  );
}

/** Light tab: the scene's sun, edited wherever the selection happens to be. */
export function LightRig() {
  const { state } = useSession();
  const sun = state?.project.scene.objects.find(
    (object) =>
      object.kind.type === "light" && object.kind.kind === "directional" && object.visible,
  );

  if (!sun || sun.kind.type !== "light") {
    return (
      <Section title="Sun" id="light-sun">
        <p className="hint">
          The scene has no visible directional light. Both the preview and the
          renderer need one: add it from the scene tree, or undo the change that
          hid it.
        </p>
      </Section>
    );
  }

  return (
    <>
      <LightEditor object={sun} light={sun.kind} />
      <p className="hint">
        This is the same sun Solid, Material and Rendered light with. Solar
        placement writes one rotation, so a single undo puts the old sun back.
      </p>
    </>
  );
}
