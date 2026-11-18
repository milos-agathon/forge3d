import { useSession } from "../state/session-context";
import { vectorLayers } from "../state/cartography";
import type { CartographyLayer } from "../bridge/types";
import { ChoiceField, NumberField, Section, TextField, Toggle } from "./fields";

/**
 * The selected layer's properties.
 *
 * One panel for the whole stack: the layer the author picked decides what is
 * editable, and the common controls (the name, the eye) are on every layer.
 */
export function CartographyProperties() {
  const { state, actions, selectedLayerId } = useSession();
  const layers = state?.project.cartography ?? [];
  const layer = layers.find((entry) => entry.id === selectedLayerId) ?? null;

  if (!layer) {
    // No layer picked: the stack itself is the list to pick from, so the
    // properties panel works in the scene workspace too (where the layer
    // panel is not on screen).
    return (
      <Section title="Map layers" id="cartography-list">
        <div className="readout readout--compact">
          <dt>Layers</dt>
          <dd>{layers.length}</dd>
        </div>
        {layers.length === 0 && (
          <p className="hint">
            This map has no layers yet. Add one in the Cartography workspace.
          </p>
        )}
        <div className="inline-actions">
          {[...layers].reverse().map((entry) => (
            <button
              key={entry.id}
              type="button"
              className="button button--small"
              onClick={() => actions.selectCartographyLayer(entry.id)}
            >
              {entry.name}
            </button>
          ))}
        </div>
        <p className="hint">
          A map is a stack of layers: the terrain&apos;s relief at the bottom,
          the elevation ramp and the imagery over it, the vector layers and
          their labels, and the plate at the front.
        </p>
      </Section>
    );
  }

  const update = (patch: Partial<CartographyLayer>) =>
    void actions.updateCartographyLayer({ ...layer, ...patch });

  const asset = layer.asset
    ? state?.project.assets.find((entry) => entry.id === layer.asset)
    : undefined;

  return (
    <>
      <Section title={layer.name} id={`cartography-${layer.type}`}>
        <TextField
          label="Name"
          value={layer.name}
          onCommit={(name) => update({ name })}
        />
        <Toggle
          label="Visible"
          checked={layer.visible}
          onChange={(visible) => update({ visible })}
        />
        {asset && (
          <div className="readout readout--compact">
            <dt>File</dt>
            <dd>{asset.name}</dd>
          </div>
        )}
        {layer.type === "polygon" && (
          <>
            <div className="pair">
              {(["R", "G", "B"] as const).map((channel, index) => (
                <NumberField
                  key={channel}
                  label={channel}
                  value={layer.colour?.[index] ?? 0}
                  step={0.05}
                  min={0}
                  max={1}
                  onCommit={(value) => {
                    const colour = [...(layer.colour ?? [0.2, 0.55, 0.75, 1])] as [
                      number,
                      number,
                      number,
                      number,
                    ];
                    colour[index] = value;
                    update({ colour });
                  }}
                />
              ))}
            </div>
            <NumberField
              label="Opacity"
              value={layer.colour?.[3] ?? 1}
              step={0.05}
              min={0}
              max={1}
              onCommit={(value) => {
                const colour = [...(layer.colour ?? [0.2, 0.55, 0.75, 1])] as [
                  number,
                  number,
                  number,
                  number,
                ];
                colour[3] = value;
                update({ colour });
              }}
            />
          </>
        )}
        {layer.type === "vector" && (
          <>
            <TextField
              label="Legend of (attribute)"
              value={layer.legendField ?? ""}
              onCommit={(legendField) =>
                update({ legendField: legendField.trim() || undefined })
              }
            />
            <p className="hint">
              One swatch per class on the plate, and the lines take the same
              colours. Empty = one colour for the whole layer.
            </p>
            <NumberField
              label="Width (px)"
              value={layer.widthPx ?? 2}
              step={1}
              min={1}
              max={20}
              onCommit={(widthPx) => update({ widthPx })}
            />
            <div className="pair">
              {(["R", "G", "B"] as const).map((channel, index) => (
                <NumberField
                  key={channel}
                  label={channel}
                  value={layer.colour?.[index] ?? 0}
                  step={0.05}
                  min={0}
                  max={1}
                  onCommit={(value) => {
                    const colour = [...(layer.colour ?? [0.9, 0.29, 0.1, 1])] as [
                      number,
                      number,
                      number,
                      number,
                    ];
                    colour[index] = value;
                    update({ colour });
                  }}
                />
              ))}
            </div>
          </>
        )}
        {layer.type === "labels" && (
          <>
            <ChoiceField
              label="Names the features of"
              value={layer.source ?? ""}
              options={[
                ...vectorLayers(state?.project).map((entry) => ({
                  value: entry.id,
                  label: entry.name,
                })),
              ]}
              onChange={(source) => update({ source })}
              hint={
                vectorLayers(state?.project).length === 0
                  ? "Add a vector layer first: the labels name its features."
                  : undefined
              }
            />
            <TextField
              label="Label field"
              value={layer.field ?? ""}
              onCommit={(field) => update({ field })}
            />
            <NumberField
              label="Label size (px)"
              value={layer.sizePx ?? 14}
              step={1}
              min={8}
              max={48}
              onCommit={(sizePx) => update({ sizePx })}
            />
            <div className="pair">
              {(["R", "G", "B"] as const).map((channel, index) => (
                <NumberField
                  key={channel}
                  label={channel}
                  value={layer.colour?.[index] ?? 0}
                  step={0.05}
                  min={0}
                  max={1}
                  onCommit={(value) => {
                    const colour = [...(layer.colour ?? [0.08, 0.08, 0.1, 1])] as [
                      number,
                      number,
                      number,
                      number,
                    ];
                    colour[index] = value;
                    update({ colour });
                  }}
                />
              ))}
            </div>
            <p className="hint">
              The attribute that names a feature (`name`, `name.en`, ...). The
              engine's own label plan places them and its text compositor draws
              them, so a name outside the packaged atlas is rejected rather
              than drawn blank.
            </p>
          </>
        )}
        {layer.type === "graticule" && (
          <>
            <NumberField
              label="Interval (degrees)"
              value={layer.intervalDeg ?? 1}
              step={0.05}
              min={0.05}
              max={30}
              onCommit={(intervalDeg) => update({ intervalDeg })}
            />
            <NumberField
              label="Width (px)"
              value={layer.widthPx ?? 1}
              step={1}
              min={1}
              max={20}
              onCommit={(widthPx) => update({ widthPx })}
            />
            <div className="pair">
              {(["R", "G", "B"] as const).map((channel, index) => (
                <NumberField
                  key={channel}
                  label={channel}
                  value={layer.colour?.[index] ?? 0}
                  step={0.05}
                  min={0}
                  max={1}
                  onCommit={(value) => {
                    const colour = [...(layer.colour ?? [0.1, 0.1, 0.12, 0.75])] as [
                      number,
                      number,
                      number,
                      number,
                    ];
                    colour[index] = value;
                    update({ colour });
                  }}
                />
              ))}
            </div>
            <Toggle
              label="Coordinate annotations"
              checked={layer.labels ?? true}
              onChange={(labels) => update({ labels })}
            />
            <p className="hint">
              The engine generates the grid of meridians and parallels for the
              map&apos;s own bounds, and the engine&apos;s label plan places the
              coordinate annotations where each line leaves the frame.
            </p>
          </>
        )}
        {layer.type === "imagery" && (
          <>
            <NumberField
              label="Opacity"
              value={layer.opacity ?? 0.82}
              step={0.02}
              min={0}
              max={1}
              onCommit={(opacity) => update({ opacity })}
            />
            <Toggle
              label="Keep the source colours"
              checked={layer.preserveColors ?? false}
              onChange={(preserveColors) => update({ preserveColors })}
            />
            <Toggle
              label="Despeckle"
              checked={layer.despeckle ?? false}
              onChange={(despeckle) => update({ despeckle })}
            />
          </>
        )}
        {layer.type === "elevationRamp" && (
          <>
            <ChoiceField
              label="Colormap"
              value={layer.colormap ?? "viridis"}
              options={[
                { value: "viridis", label: "Viridis" },
                { value: "magma", label: "Magma" },
                { value: "vik", label: "Vik" },
              ]}
              onChange={(colormap) => update({ colormap })}
            />
            <NumberField
              label="Strength"
              value={layer.strength ?? 1}
              step={0.05}
              min={0}
              max={1}
              onCommit={(strength) => update({ strength })}
            />
            <Toggle
              label="Elevation legend"
              checked={layer.legend ?? false}
              onChange={(legend) => update({ legend })}
            />
          </>
        )}
        {layer.type === "plate" && (
          <>
            <TextField
              label="Title"
              value={layer.title ?? ""}
              onCommit={(title) => update({ title })}
            />
            <TextField
              label="Caption"
              value={layer.caption ?? ""}
              onCommit={(caption) => update({ caption })}
            />
            <Toggle
              label="North arrow"
              checked={layer.northArrow ?? true}
              onChange={(northArrow) => update({ northArrow })}
            />
            <Toggle
              label="Scale bar"
              checked={layer.scaleBar ?? true}
              onChange={(scaleBar) => update({ scaleBar })}
            />
            <Toggle
              label="Legend"
              checked={layer.legend ?? true}
              onChange={(legend) => update({ legend })}
            />
          </>
        )}
        {layer.type === "relief" && <ReliefFacts />}
        <div className="inline-actions">
          <button
            type="button"
            className="button button--small button--danger"
            onClick={() => void actions.removeCartographyLayer(layer.id)}
          >
            Remove this layer
          </button>
        </div>
      </Section>
    </>
  );
}

/** The terrain's own facts: what the relief layer is made of. */
function ReliefFacts() {
  const { state, actions } = useSession();
  const terrainObject = state?.project.scene.objects.find(
    (entry) => entry.kind.type === "terrain",
  );
  const terrain =
    terrainObject && terrainObject.kind.type === "terrain" ? terrainObject.kind : null;
  if (!terrainObject || !terrain) return null;
  const reliefPercent =
    terrain.elevationRange && terrain.worldSpanXz
      ? ((terrain.elevationRange[1] - terrain.elevationRange[0]) /
          Math.max(terrain.worldSpanXz[0], terrain.worldSpanXz[1])) *
        100
      : null;
  return (
    <>
      <div className="readout readout--compact">
        <dt>Elevation</dt>
        <dd>
          {terrain.elevationRange === null
            ? "—"
            : `${terrain.elevationRange[0].toFixed(0)}–${terrain.elevationRange[1].toFixed(0)} m`}
        </dd>
        <dt>Footprint</dt>
        <dd>
          {terrain.worldSpanXz === null
            ? "—"
            : terrain.worldSpanXz
                .map((value) => `${(value / 1000).toFixed(1)} km`)
                .join(" × ")}
        </dd>
        <dt>Relief</dt>
        <dd>
          {reliefPercent === null ? "—" : `${reliefPercent.toFixed(1)}% of the footprint`}
        </dd>
      </div>
      <NumberField
        label="Vertical scale"
        value={terrain.zScale}
        step={0.1}
        min={0.1}
        max={50}
        onCommit={(zScale) =>
          void actions.apply({ type: "setTerrainZScale", id: terrainObject.id, zScale })
        }
      />
      <div className="inline-actions">
        <button
          type="button"
          className="button button--small"
          onClick={() =>
            void actions.apply({ type: "fitTerrainRelief", id: terrainObject.id })
          }
        >
          Fit the DEM&apos;s relief
        </button>
        <button
          type="button"
          className="button button--small button--primary"
          onClick={() => void actions.applyMapLook()}
        >
          Map look
        </button>
      </div>
    </>
  );
}
