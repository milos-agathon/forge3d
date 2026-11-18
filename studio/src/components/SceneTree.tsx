import { useMemo } from "react";

import type { SceneObject } from "../bridge/types";
import { objectKindLabel } from "../bridge/types";
import { useSession } from "../state/session-context";
import { EyeIcon, EyeOffIcon, LockIcon, UnlockIcon, objectIcon } from "./icons";

interface Row {
  object: SceneObject;
  depth: number;
}

function flatten(objects: SceneObject[]): Row[] {
  const byParent = new Map<string | null, SceneObject[]>();
  for (const object of objects) {
    const list = byParent.get(object.parent) ?? [];
    list.push(object);
    byParent.set(object.parent, list);
  }
  const rows: Row[] = [];
  const seen = new Set<string>();
  const walk = (parent: string | null, depth: number) => {
    for (const object of byParent.get(parent) ?? []) {
      if (seen.has(object.id)) continue;
      seen.add(object.id);
      rows.push({ object, depth });
      walk(object.id, depth + 1);
    }
  };
  walk(null, 0);
  // Defensive: an object whose parent vanished still shows up in the tree.
  for (const object of objects) {
    if (!seen.has(object.id)) rows.push({ object, depth: 0 });
  }
  return rows;
}

export function SceneTree() {
  const { state, actions } = useSession();
  const scene = state?.project.scene;
  const selection = useMemo(
    () => new Set(state?.project.selection ?? []),
    [state],
  );
  // Computed, not memoised: the tree is small, and a bridge is free to return
  // a snapshot whose object identity is reused. Correctness beats a micro-optimisation.
  const rows = flatten(scene?.objects ?? []);

  if (!scene) return null;

  const select = (event: React.MouseEvent, id: string) => {
    const additive = event.metaKey || event.ctrlKey || event.shiftKey;
    const current = state?.project.selection ?? [];
    if (!additive) {
      void actions.select([id]);
      return;
    }
    void actions.select(
      current.includes(id) ? current.filter((entry) => entry !== id) : [...current, id],
    );
  };

  return (
    <div className="panel">
      <header className="panel__header">
        <h2>Scene</h2>
        <span className="panel__meta">{scene.objects.length} objects</span>
      </header>

      <div className="tree" role="tree" aria-label="Scene hierarchy">
        {rows.map(({ object, depth }) => {
          const Icon = objectIcon(object.kind.type);
          const selected = selection.has(object.id);
          return (
            <div
              key={object.id}
              role="treeitem"
              aria-selected={selected}
              className={`tree-row ${selected ? "is-selected" : ""}`}
              style={{ paddingLeft: 8 + depth * 14 }}
              onClick={(event) => select(event, object.id)}
            >
              <Icon size={14} className={`tree-icon tree-icon--${object.kind.type}`} />
              <span className={`tree-name ${object.visible ? "" : "is-hidden"}`}>
                {object.name}
              </span>
              <span className="tree-kind">{objectKindLabel(object.kind)}</span>
              <button
                type="button"
                className="tree-action"
                title={object.visible ? "Hide object" : "Show object"}
                aria-label={object.visible ? "Hide object" : "Show object"}
                onClick={(event) => {
                  event.stopPropagation();
                  void actions.apply({
                    type: "setVisibility",
                    id: object.id,
                    visible: !object.visible,
                  });
                }}
              >
                {object.visible ? <EyeIcon size={14} /> : <EyeOffIcon size={14} />}
              </button>
              <button
                type="button"
                className={`tree-action ${object.locked ? "is-on" : ""}`}
                title={object.locked ? "Unlock object" : "Lock object"}
                aria-label={object.locked ? "Unlock object" : "Lock object"}
                onClick={(event) => {
                  event.stopPropagation();
                  void actions.apply({
                    type: "setLocked",
                    id: object.id,
                    locked: !object.locked,
                  });
                }}
              >
                {object.locked ? <LockIcon size={14} /> : <UnlockIcon size={14} />}
              </button>
            </div>
          );
        })}
        {rows.length === 0 && (
          <p className="panel__empty">This scene has no objects yet.</p>
        )}
      </div>

      <footer className="panel__footer">
        <span className="mono">scene {scene.id.slice(0, 8)}</span>
        {scene.activeCamera && (
          <span className="mono">
            cam{" "}
            {scene.objects.find((object) => object.id === scene.activeCamera)?.name ?? "missing"}
          </span>
        )}
      </footer>
    </div>
  );
}
