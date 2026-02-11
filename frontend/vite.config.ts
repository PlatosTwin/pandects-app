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
      // Enable compression in dev mode for better performance testing
      middlewareMode: false,
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
      sourcemap: false,
      minify: "esbuild",
      // Target modern browsers to reduce legacy JavaScript
      // This eliminates the need for polyfills and reduces bundle size
      target: "esnext",
      // CSS is automatically minified when minify is set
      rollupOptions: {
        output: {
          // Better code splitting to reduce unused JavaScript
          manualChunks: (id) => {
            // Separate vendor chunks for better caching
            if (id.includes("node_modules")) {
              // Large UI libraries
              if (
                id.includes("@radix-ui") ||
                id.includes("lucide-react") ||
                id.includes("framer-motion")
              ) {
                return "vendor-ui";
              }
              // React and core dependencies
              if (
                id.includes("react") ||
                id.includes("react-dom") ||
                id.includes("react-router")
              ) {
                return "vendor-react";
              }
              // Query and state management
              if (id.includes("@tanstack/react-query")) {
                return "vendor-query";
              }
              // Other vendor code
              return "vendor";
            }
          },
          // Optimize chunk file names for better caching
          chunkFileNames: "assets/js/[name]-[hash].js",
          entryFileNames: "assets/js/[name]-[hash].js",
          assetFileNames: "assets/[ext]/[name]-[hash].[ext]",
        },
      },
      // Increase chunk size warning limit since we're doing manual chunking
      chunkSizeWarningLimit: 1000,
    },
    plugins: [react(), expressPlugin(), criticalCssPlugin()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./client"),
        "@shared": path.resolve(__dirname, "./shared"),
        "@branding": path.resolve(__dirname, "../branding"),
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
    transformIndexHtml(html, ctx) {
      let nextHtml = html;

      if (criticalCss) {
        nextHtml = nextHtml.replace(
          "</head>",
          `<style>${criticalCss}</style></head>`,
        );
      }

      // Add DNS prefetch and preconnect for external resources
      const dnsPrefetch = `
  <link rel="dns-prefetch" href="https://www.googletagmanager.com" />
  <link rel="preconnect" href="https://www.googletagmanager.com" crossorigin />
  <link rel="preconnect" href="https://api.pandects.org" crossorigin />`;

      nextHtml = nextHtml.replace("</head>", `${dnsPrefetch}</head>`);

      // Add preload hints for LCP image only if not already present (index.html may have them)
      const hasLogoPreload = (nextHtml.match(/<link[^>]+>/g) || []).some(
        (link) => link.includes("rel=\"preload\"") && link.includes("logo-128") && link.includes("as=\"image\"")
      );
      if (ctx?.bundle && !hasLogoPreload) {
        const preloadHints: string[] = [];
        for (const [, chunk] of Object.entries(ctx.bundle)) {
          if (chunk.type === "asset") {
            const assetFileName = chunk.fileName;
            if (assetFileName.includes("logo-128")) {
              if (assetFileName.endsWith(".webp")) {
                preloadHints.push(
                  `  <link rel="preload" href="/${assetFileName}" as="image" type="image/webp" />`,
                );
              } else if (assetFileName.endsWith(".png")) {
                preloadHints.push(
                  `  <link rel="preload" href="/${assetFileName}" as="image" type="image/png" />`,
                );
              }
            }
          }
        }
        if (preloadHints.length > 0) {
          nextHtml = nextHtml.replace(
            "</head>",
            `\n${preloadHints.join("\n")}\n</head>`,
          );
        }
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
    writeBundle(options, bundle) {
      // After bundle is written, inject logo preload hints only if not already present
      const htmlPath = path.join(options.dir || "dist/spa", "index.html");
      if (fs.existsSync(htmlPath)) {
        let html = fs.readFileSync(htmlPath, "utf-8");
        const hasLogoPreload = (html.match(/<link[^>]+>/g) || []).some(
          (link) => link.includes("rel=\"preload\"") && link.includes("logo-128") && link.includes("as=\"image\"")
        );
        if (hasLogoPreload) return;

        const preloadHints: string[] = [];
        for (const [, chunk] of Object.entries(bundle)) {
          if (chunk.type === "asset") {
            const assetFileName = chunk.fileName;
            if (assetFileName.includes("logo-128")) {
              if (assetFileName.endsWith(".webp")) {
                preloadHints.push(
                  `  <link rel="preload" href="/${assetFileName}" as="image" type="image/webp" />`,
                );
              } else if (assetFileName.endsWith(".png")) {
                preloadHints.push(
                  `  <link rel="preload" href="/${assetFileName}" as="image" type="image/png" />`,
                );
              }
            }
          }
        }
        if (preloadHints.length > 0) {
          html = html.replace(
            "</head>",
            `\n${preloadHints.join("\n")}\n</head>`,
          );
          fs.writeFileSync(htmlPath, html, "utf-8");
        }
      }
    },
  };
}
