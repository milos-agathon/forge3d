import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Tauri drives the dev server: fixed port, no screen clearing, and the Rust
// tree excluded from the file watcher (Cargo owns those rebuilds).
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: {
      ignored: ["**/src-tauri/**", "**/crates/**", "**/target/**", "**/node_modules/**"],
    },
  },
  build: {
    target: "es2022",
    sourcemap: false,
  },
});
