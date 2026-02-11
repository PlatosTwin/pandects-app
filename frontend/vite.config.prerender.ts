import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

export default defineConfig({
  plugins: [react()],
  build: {
    ssr: path.resolve(__dirname, "prerender/entry.tsx"),
    outDir: "dist/prerender",
    target: "node22",
    rollupOptions: {
      output: {
        entryFileNames: "prerender.mjs",
      },
    },
    minify: false,
    sourcemap: false,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./client"),
      "@shared": path.resolve(__dirname, "./shared"),
      "@branding": path.resolve(__dirname, "../branding"),
    },
  },
  ssr: {
    noExternal: true,
  },
});
