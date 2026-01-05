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
      sourcemap: false,
      minify: "esbuild",
      // Target modern browsers to reduce legacy JavaScript
      // This eliminates the need for polyfills and reduces bundle size
      target: "esnext",
      // CSS is automatically minified when minify is set
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
    transformIndexHtml(html, ctx) {
      let nextHtml = html;

      if (criticalCss) {
        nextHtml = nextHtml.replace(
          "</head>",
          `<style>${criticalCss}</style></head>`,
        );
      }

      // Add DNS prefetch for external resources (keeping only what was safe before)
      const dnsPrefetch = `
  <link rel="dns-prefetch" href="https://www.googletagmanager.com" />
  <link rel="preconnect" href="https://www.googletagmanager.com" crossorigin />`;

      nextHtml = nextHtml.replace("</head>", `${dnsPrefetch}</head>`);

      // Add preload hints for LCP image using bundle info if available
      if (ctx?.bundle) {
        const preloadHints: string[] = [];
        for (const [fileName, chunk] of Object.entries(ctx.bundle)) {
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
      // After bundle is written, inject preload hints into HTML if bundle info wasn't available in transformIndexHtml
      const htmlPath = path.join(options.dir || "dist/spa", "index.html");
      if (fs.existsSync(htmlPath)) {
        let html = fs.readFileSync(htmlPath, "utf-8");
        const preloadHints: string[] = [];
        
        for (const [fileName, chunk] of Object.entries(bundle)) {
          if (chunk.type === "asset") {
            const assetFileName = chunk.fileName;
            if (assetFileName.includes("logo-128")) {
              if (assetFileName.endsWith(".webp")) {
                if (!html.includes(`rel="preload" href="/${assetFileName}"`)) {
                  preloadHints.push(
                    `  <link rel="preload" href="/${assetFileName}" as="image" type="image/webp" />`,
                  );
                }
              } else if (assetFileName.endsWith(".png")) {
                if (!html.includes(`rel="preload" href="/${assetFileName}"`)) {
                  preloadHints.push(
                    `  <link rel="preload" href="/${assetFileName}" as="image" type="image/png" />`,
                  );
                }
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
