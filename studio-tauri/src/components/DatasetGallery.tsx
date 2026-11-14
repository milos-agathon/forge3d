import { useEffect, useMemo, useState } from "react";

import { bridge } from "../bridge/env";
import type { DatasetInfo } from "../bridge/types";
import { thumbPixels } from "../viewport/datasetThumb";

/**
 * The engine's dataset registry as a pick-as-starter-scene gallery.
 *
 * Everything here already exists on the user's machine when the repo sample
 * assets are present; otherwise the click downloads once (hash-verified by
 * pooch) and caches under `~/.forge3d/datasets`. Nothing is fetched at idle —
 * only on the click.
 */
export function DatasetGallery({
  onPick,
  disabled,
}: {
  onPick: (presetId: string) => void;
  disabled: boolean;
}) {
  const [datasets, setDatasets] = useState<DatasetInfo[] | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
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
  }, []);

  const terrain = useMemo(
    () => (datasets ?? []).filter((entry) => entry.kind === "dem"),
    [datasets],
  );

  if (problem) {
    return <p className="hint">Engine datasets unavailable: {problem}</p>;
  }
  if (!datasets) return null;
  if (terrain.length === 0) return null;

  return (
    <>
      <div className="dataset-gallery__head">
        <h3>From the engine&rsquo;s datasets</h3>
        <p className="hint">
          Registry DEMs. Bundled with the repo when present; otherwise fetched
          once (hash-verified) on click.
        </p>
      </div>
      <div className="preset-grid">
        {terrain.map((entry) => (
          <DatasetCard
            key={entry.name}
            entry={entry}
            onPick={onPick}
            disabled={disabled}
          />
        ))}
      </div>
    </>
  );
}

function DatasetCard({
  entry,
  onPick,
  disabled,
}: {
  entry: DatasetInfo;
  onPick: (presetId: string) => void;
  disabled: boolean;
}) {
  const thumbnail = useDatasetThumbnail(entry.path);
  // The registry hands us a local path only when one resolves; the
  // thumbnail is rendered from it. Anything else shows the fetch glyph.
  const ready = Boolean(entry.path);
  return (
    <button
      type="button"
      className="preset dataset"
      disabled={disabled}
      onClick={() => onPick(`dataset:${entry.name}`)}
      title={
        entry.description ??
        (ready ? `${entry.filename}` : "Downloads on first use (hash-verified)")
      }
    >
      {thumbnail ? (
        <canvas
          className="dataset__thumb"
          width={thumbnail.width}
          height={thumbnail.height}
          ref={(node) => {
            if (!node) return;
            const ctx = node.getContext("2d");
            if (ctx) ctx.putImageData(thumbnail, 0, 0);
          }}
          aria-hidden="true"
        />
      ) : (
        <span className="dataset__placeholder" aria-hidden="true">
          {ready ? "…" : "⤓"}
        </span>
      )}
      <span className="preset__name">{entry.name}</span>
      <span className="preset__desc">{entry.description ?? entry.filename}</span>
      <span className="preset__meta mono">
        {ready ? (entry.filename ?? "local file") : "download on first use"}
      </span>
    </button>
  );
}

/** Elevation thumbnail from the shell's own heightfield preview; null until ready. */
function useDatasetThumbnail(path: string | undefined): ImageData | null {
  const [image, setImage] = useState<ImageData | null>(null);
  useEffect(() => {
    if (!path) {
      setImage(null);
      return;
    }
    let cancelled = false;
    bridge
      .terrainPreview(path, 48)
      .then((preview) => {
        if (cancelled) return;
        const res = preview.resolution;
        const pixels = thumbPixels(preview);
        setImage(new ImageData(pixels, res, res));
      })
      .catch(() => {
        // A missing raster must not break the gallery: the card stays usable.
      });
    return () => {
      cancelled = true;
    };
  }, [path]);
  return image;
}
