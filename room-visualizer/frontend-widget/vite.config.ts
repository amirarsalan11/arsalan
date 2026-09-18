import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Milestone 1 scope: standard Vite + React setup only.
// Iframe isolation / postMessage wiring with the SDK is implemented
// in the "SDK + Iframe" milestone, not here.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
  build: {
    outDir: "dist",
  },
});
