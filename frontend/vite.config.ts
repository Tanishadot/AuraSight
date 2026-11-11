import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const backend = process.env.AURASIGHT_BACKEND ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": { target: backend, changeOrigin: false },
      "/ws": { target: backend.replace(/^http/, "ws"), ws: true },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
