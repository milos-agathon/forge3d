import { useEffect, useState } from "react";

import { bridge } from "../bridge/env";
import type { DatasetInfo } from "../bridge/types";
import { datasetEntries, DATASET_KINDS, type DatasetKind } from "../state/datasets";
import { CloseIcon } from "./icons";

/**
 * Open one of the engine's registry datasets as a layer or a DEM.
 *
 * The registry is the engine's own (`forge3d.datasets`), so the list is the
 * engine's truth about what it ships, and the fetch is the engine's too
 * (hash-verified, cached under `~/.forge3d/datasets`). Nothing downloads
 * until the user picks an entry; the resolution happens in the shell's
 * `dataset_path` command, the same call every import site uses.
 */
export function DatasetPicker({
  kind,
  label,
  onPick,
}: {
  kind: DatasetKind;
  /** The button's caption when the picker is closed. */
  label: string;
  onPick: (name: string) => void | Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [datasets, setDatasets] = useState<DatasetInfo[] | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    if (!open || datasets) return;
    let cancelled = false;
    bridge
      .datasetList()
      .then((entries) => {
        if (!cancelled) setDatasets(entries);
      })
      .catch((cause) => {
        if (!cancelled) {
          setProblem(cause instanceof Error ? cause.message : String(cause));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [datasets, open]);

  if (!open) {
    return (
      <button
        type="button"
        className="button button--small"
        onClick={() => setOpen(true)}
        title={`Open one of the engine's ${DATASET_KINDS[kind].toLowerCase()} datasets`}
      >
        {label}
      </button>
    );
  }

  const entries = datasetEntries(datasets, kind);
  const close = () => {
    setOpen(false);
    setProblem(null);
  };

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label={`Open an engine ${DATASET_KINDS[kind].toLowerCase()}`}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) close();
      }}
    >
      <div className="modal dataset-picker">
        <header className="modal__header">
          <h2>Engine datasets — {DATASET_KINDS[kind]}</h2>
          <button type="button" className="icon-button" aria-label="Cancel" onClick={close}>
            <CloseIcon size={14} />
          </button>
        </header>

        {problem && <p className="hint hint--error">{problem}</p>}
        {!problem && !datasets && <p className="hint">Reading the engine's registry…</p>}
        {datasets && entries.length === 0 && (
          <p className="hint">
            The engine's registry has no {DATASET_KINDS[kind].toLowerCase()} dataset.
          </p>
        )}

        <div className="dataset-list">
          {entries.map((entry) => (
            <button
              key={entry.name}
              type="button"
              className="dataset-list__row"
              onClick={() => {
                close();
                void onPick(entry.name);
              }}
              title={entry.description ?? entry.filename ?? entry.name}
            >
              <span className="dataset-list__name">{entry.name}</span>
              <span className="dataset-list__desc">
                {entry.description ?? entry.filename ?? "from the engine's registry"}
              </span>
              <span className="dataset-list__meta mono">
                {entry.path ? "local file" : "downloads on first use"}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
