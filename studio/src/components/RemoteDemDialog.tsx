import { useState } from "react";

import { bridge } from "../bridge/env";
import type { RemoteCogInfo, RemoteCogPreview } from "../bridge/types";
import { useSession } from "../state/session-context";
import { thumbPixels } from "../viewport/datasetThumb";
import { CloseIcon } from "./icons";

const COMPRESSION_NAMES: Record<number, string> = {
  1: "none",
  5: "LZW",
  6: "JPEG",
  7: "old-style JPEG",
  8: "Deflate",
  50000: "Zstd",
};

/**
 * Open a remote DEM (a Cloud Optimized GeoTIFF) as a project.
 *
 * The probe is the engine's own range-reader: one HEAD request, no pixel
 * data, and it fails loudly on anything that is not a valid remote COG the
 * reader can open. Confirmation downloads the file once to the local cache
 * (`~/…/forge3d-cog-cache`), hash-keyed by URL, so reopening costs nothing.
 */
export function RemoteDemDialog() {
  const { actions, busy } = useSession();
  const [open, setOpen] = useState(false);
  const [url, setUrl] = useState("");
  const [probing, setProbing] = useState(false);
  const [info, setInfo] = useState<RemoteCogInfo | null>(null);
  const [preview, setPreview] = useState<RemoteCogPreview | null>(null);
  const [error, setError] = useState<string | null>(null);

  const close = () => {
    setOpen(false);
    setError(null);
    setInfo(null);
    setPreview(null);
    setProbing(false);
  };

  const probe = async () => {
    const target = url.trim();
    if (!target) return;
    setProbing(true);
    setError(null);
    setInfo(null);
    setPreview(null);
    try {
      setInfo(await bridge.remoteCogInfo(target));
      // The ladder's first rung: the overview, read through the engine's
      // range reader — the terrain shows up before the file is downloaded.
      // A preview that fails is not a probe failure: the summary still stands.
      try {
        setPreview(await bridge.remoteCogPreview(target, 96));
      } catch {
        setPreview(null);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setProbing(false);
    }
  };

  const pick = (id: string) => {
    // The fetch happens inside createProject; close first so the busy
    // state is not double-rendered.
    close();
    void actions.createProject(id);
  };

  if (!open) {
    return (
      <button
        type="button"
        className="preset preset--empty"
        disabled={busy}
        onClick={() => setOpen(true)}
        title="Open a remote DEM — a Cloud Optimized GeoTIFF served over HTTP(S)"
      >
        <span className="preset__name">Remote DEM (COG)…</span>
        <span className="preset__desc">
          Probe a remote Cloud Optimized GeoTIFF before anything downloads
        </span>
        <span className="preset__meta mono">http / https / file</span>
      </button>
    );
  }

  const compression = info ? COMPRESSION_NAMES[info.levels[0]?.compression] ?? `#${info.levels[0]?.compression}` : "";

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="Open a remote DEM"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) close();
      }}
    >
      <div className="modal remote-dem">
        <header className="modal__header">
          <h2>Open a remote DEM</h2>
          <button type="button" className="icon-button" aria-label="Cancel" onClick={close}>
            <CloseIcon size={14} />
          </button>
        </header>

        <p className="hint">
          An HTTP(S) URL to a Cloud Optimized GeoTIFF (a tiled, deflate or
          LZW GeoTIFF). The engine probes it with one HEAD request — nothing
          downloads until you confirm.
        </p>

        <label className="field">
          <span className="field-label">URL</span>
          <input
            className="input input--mono"
            type="text"
            autoFocus
            spellCheck={false}
            placeholder="https://example.com/dem.tif"
            value={url}
            onChange={(event) => {
              setUrl(event.target.value);
              setInfo(null);
              setError(null);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter") void probe();
            }}
          />
        </label>

        {info && (
          <div className="probe-summary mono">
            <div className="probe-row">
              <span>Shape</span>
              <span>
                {info.levels[0]?.width} × {info.levels[0]?.height} px · {info.levels[0]?.bitsPerSample}-bit · {compression}
              </span>
            </div>
            <div className="probe-row">
              <span>Streamable</span>
              <span>{info.streamable ? "yes (tiled, native range reads)" : "no"}</span>
            </div>
            <div className="probe-row">
              <span>Overview levels</span>
              <span>
                {info.overviewCount} · {info.totalTiles} tiles total · {info.levels[0]?.tileWidth}×{info.levels[0]?.tileHeight} px tiles
              </span>
            </div>
            {info.bounds && (
              <div className="probe-row">
                <span>Bounds</span>
                <span>
                  {info.bounds.map((value) => value.toFixed(4)).join(", ")}
                </span>
              </div>
            )}
          </div>
        )}

        {preview && (
          <figure className="cog-preview">
            <canvas
              width={preview.resolution}
              height={preview.resolution}
              ref={(node) => {
                if (!node) return;
                const ctx = node.getContext("2d");
                if (!ctx) return;
                const pixels = thumbPixels(preview);
                ctx.putImageData(
                  new ImageData(pixels, preview.resolution, preview.resolution),
                  0,
                  0,
                );
              }}
            />
            <figcaption className="hint">
              Preview from overview level {preview.level} (
              {preview.levelWidth} × {preview.levelHeight} px) — read through the
              engine&rsquo;s range reader; the full file downloads on open.
            </figcaption>
          </figure>
        )}

        {error && <p className="hint hint--error">Probe failed: {error}</p>}

        <footer className="modal__footer">
          <span className="hint">
            Downloads once to the local cache on open; reopening costs no fetch.
          </span>
          <div className="modal__footer-actions">
            <button type="button" className="button" onClick={close}>
              Cancel
            </button>
            {!info ? (
              <button
                type="button"
                className="button"
                disabled={probing || !url.trim()}
                onClick={() => void probe()}
              >
                {probing ? "Probing…" : "Probe"}
              </button>
            ) : (
              <button
                type="button"
                className="button button--primary"
                disabled={probing || busy}
                onClick={() => pick(`url:${url.trim()}`)}
              >
                Open DEM
              </button>
            )}
          </div>
        </footer>
      </div>
    </div>
  );
}
