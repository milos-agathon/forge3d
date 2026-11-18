import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import { bridge } from "./bridge/env";
import { SessionProvider } from "./state/session";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("#root is missing from index.html");

createRoot(root).render(
  <StrictMode>
    {bridge.kind === "tauri" ? (
      <SessionProvider>
        <App />
      </SessionProvider>
    ) : (
      <div className="app">
        <main className="workspace workspace--empty">
          <div className="empty">
            <div className="empty__card">
              <span className="empty__mark" aria-hidden="true">
                F3
              </span>
              <h1>Forge3D Studio</h1>
              <p className="empty__lede">
                This page only works inside the desktop shell, where projects,
                files and the renderer live.
              </p>
              <p className="hint hint--boxed">
                Start it with <code>npm run tauri:dev</code>.
              </p>
            </div>
          </div>
        </main>
      </div>
    )}
  </StrictMode>,
);
