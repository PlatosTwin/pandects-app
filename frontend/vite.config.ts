import { defineConfig, loadEnv, Plugin } from "vite";
import react from "@vitejs/plugin-react-swc";
import fs from "fs";
import path from "path";
import { createServer } from "./server";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const host = env.VITE_DEV_HOST?.trim() || "localhost";
  const httpsCert = env.VITE_DEV_HTTPS_CERT?.trim() || "";
  const httpsKey = env.VITE_DEV_HTTPS_KEY?.trim() || "";
  const https =
    httpsCert && httpsKey
      ? { cert: httpsCert, key: httpsKey }
      : undefined;

  return {
    server: {
      host,
      port: 8080,
      https,
      proxy: {
        "/api": {
          target: "http://localhost:5113",
          changeOrigin: true,
          secure: false,
        },
      },
    },
    build: {
      outDir: "dist/spa",
    },
    plugins: [react(), expressPlugin(), criticalCssPlugin()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./client"),
        "@shared": path.resolve(__dirname, "./shared"),
      },
    },
  };
});

function expressPlugin(): Plugin {
  return {
    name: "express-plugin",
    apply: "serve", // Only apply during development (serve mode)
    configureServer(server) {
      const app = createServer();

      // Add Express app as middleware to Vite dev server
      server.middlewares.use(app);
    },
  };
}

function criticalCssPlugin(): Plugin {
  const criticalPath = path.resolve(__dirname, "client/critical.css");
  const criticalCss = fs.existsSync(criticalPath)
    ? fs.readFileSync(criticalPath, "utf8").trim()
    : "";

  return {
    name: "critical-css",
    apply: "build",
    transformIndexHtml(html) {
      let nextHtml = html;

      if (criticalCss) {
        nextHtml = nextHtml.replace(
          "</head>",
          `<style>${criticalCss}</style></head>`,
        );
      }

      return nextHtml.replace(/<link\s+[^>]*rel="stylesheet"[^>]*>/g, (match) => {
        const hrefMatch = match.match(/href="([^"]+\.css)"/);
        if (!hrefMatch) return match;
        const href = hrefMatch[1];
        const preservedAttrs = match
          .replace(/<link\s+/g, "")
          .replace(/>/g, "")
          .replace(/rel="stylesheet"/g, "")
          .replace(/href="[^"]+"/g, "")
          .trim();
        const attrString = preservedAttrs ? ` ${preservedAttrs}` : "";
        return [
          `<link rel="preload" as="style" href="${href}"${attrString}>`,
          `<link rel="stylesheet" href="${href}"${attrString} media="print" onload="this.media='all'">`,
          `<noscript><link rel="stylesheet" href="${href}"${attrString}></noscript>`,
        ].join("");
      });
    },
  };
}
