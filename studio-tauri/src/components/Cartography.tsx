import { useMemo, useState } from "react";

import { useSession } from "../state/session-context";
import type { CartographyLayer } from "../bridge/types";

/**
 * The map's layer stack.
 *
 * The model holds the layers back to front (the first is drawn first); the
 * panel shows them the way a paint program does — the front at the top. Each
 * row has an eye, a swatch of what the layer draws, its name and its kind.
 */

const KIND_LABEL: Record<CartographyLayer["type"], string> = {
  relief: "relief",
  elevationRamp: "ramp",
  imagery: "imagery",
  vector: "vector",
  polygon: "polygon",
  labels: "labels",
  graticule: "grid",
  plate: "plate",
};

/** The swatch of a layer: what it draws, as a colour. */
function swatch(layer: CartographyLayer): string {
  switch (layer.type) {
    case "elevationRamp":
      return "linear-gradient(135deg, #2b3550, #c9a06a)";
    case "imagery":
      return "#6f7f5a";
    case "vector":
      return `rgba(${Math.round((layer.colour?.[0] ?? 0.9) * 255)}, ${Math.round(
        (layer.colour?.[1] ?? 0.29) * 255,
      )}, ${Math.round((layer.colour?.[2] ?? 0.1) * 255)}, 1)`;
    case "polygon":
      return `rgba(${Math.round((layer.colour?.[0] ?? 0.2) * 255)}, ${Math.round(
        (layer.colour?.[1] ?? 0.55) * 255,
      )}, ${Math.round((layer.colour?.[2] ?? 0.75) * 255)}, 1)`;
    case "labels":
      return "#1c1f26";
    case "graticule":
      return `rgba(${Math.round((layer.colour?.[0] ?? 0.1) * 255)}, ${Math.round(
        (layer.colour?.[1] ?? 0.1) * 255,
      )}, ${Math.round((layer.colour?.[2] ?? 0.12) * 255)}, 1)`;
    case "plate":
      return "#e8ebf0";
    default:
      return "#6d675c";
  }
}

export function LayerStack() {
  const { state, actions, selectedLayerId } = useSession();
  const layers = state?.project.cartography ?? [];
  // The panel shows the front at the top; the model is back to front.
  const ordered = useMemo(() => [...layers].reverse(), [layers]);

  return (
    <div className="layerstack">
      <div className="layerstack__head">
        <h2>Layers</h2>
        <span>{layers.length}</span>
      </div>
      <ul className="layerstack__list">
        {ordered.map((layer) => {
          const index = layers.findIndex((entry) => entry.id === layer.id);
          return (
            <li
              key={layer.id}
              className={`layerrow${
                layer.id === selectedLayerId ? " is-selected" : ""
              }${layer.visible ? "" : " is-hidden"}`}
            >
              <button
                type="button"
                className="layerrow__eye"
                title={layer.visible ? "Hide this layer" : "Show this layer"}
                aria-label={layer.visible ? "Hide this layer" : "Show this layer"}
                onClick={() =>
                  void actions.updateCartographyLayer({
                    ...layer,
                    visible: !layer.visible,
                  })
                }
              />
              <button
                type="button"
                className="layerrow__main"
                onClick={() => actions.selectCartographyLayer(layer.id)}
              >
                <span
                  className="layerrow__swatch"
                  style={{ background: swatch(layer) }}
                  aria-hidden="true"
                />
                <span className="layerrow__name">{layer.name}</span>
                <span className="layerrow__kind">{KIND_LABEL[layer.type]}</span>
              </button>
              <span className="layerrow__move">
                <button
                  type="button"
                  title="Move up"
                  aria-label="Move up"
                  disabled={index >= layers.length - 1}
                  onClick={() => void actions.moveCartographyLayer(layer.id, index + 1)}
                >
                  ↑
                </button>
                <button
                  type="button"
                  title="Move down"
                  aria-label="Move down"
                  disabled={index <= 0}
                  onClick={() => void actions.moveCartographyLayer(layer.id, index - 1)}
                >
                  ↓
                </button>
              </span>
            </li>
          );
        })}
      </ul>
      <AddLayer />
    </div>
  );
}

/** The add row: the layer kinds the map can hold. */
function AddLayer() {
  const { state, actions } = useSession();
  const [open, setOpen] = useState(false);
  const vectors = (state?.project.cartography ?? []).filter(
    (layer) => layer.type === "vector",
  );

  return (
    <div className="layerstack__add">
      <button
        type="button"
        className="button button--small"
        onClick={() => setOpen((value) => !value)}
      >
        + Add layer
      </button>
      {open && (
        <div className="layerstack__menu">
          <button type="button" onClick={() => void actions.addImageryLayer()}>
            Imagery…
          </button>
          <button type="button" onClick={() => void actions.addVectorLayer()}>
            Vector layer…
          </button>
          <button type="button" onClick={() => void actions.addPolygonLayer()}>
            Polygon layer…
          </button>
          {vectors.length > 0 && (
            <button
              type="button"
              onClick={() => void actions.addLabelLayer(vectors[0].id, "name")}
            >
              Labels…
            </button>
          )}
          <button type="button" onClick={() => void actions.addRampLayer("viridis")}>
            Elevation ramp
          </button>
          <button type="button" onClick={() => void actions.addGraticuleLayer()}>
            Graticule
          </button>
          <button type="button" onClick={() => void actions.addPlateLayer("")}>
            Map plate
          </button>
        </div>
      )}
    </div>
  );
}
