import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// BASE is "/ml-atlas/" on GitHub Pages and "./" for the downloadable copy.
export default defineConfig({
  base: process.env.BASE ?? "./",
  plugins: [react()],
  worker: { format: "es" },
  build: { target: "es2022", chunkSizeWarningLimit: 1500 },
});
