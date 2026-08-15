import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig(({ mode }) => ({
  root: "github",
  base: mode === "development" ? "/" : process.env.GITHUB_PAGES_BASE || "/",
  plugins: [react()],
  build: {
    outDir: "../dist-pages",
    emptyOutDir: true,
  },
}));
