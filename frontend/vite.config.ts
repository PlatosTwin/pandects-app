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
      rollupOptions: {
        output: {
          manualChunks: (id) => {
            // Separate large icon library (check before generic "react" check)
            if (id.includes("lucide-react")) {
              return "vendor-icons";
            }
            // Keep React, React DOM, and React Router together in same chunk
            // This prevents React context errors that occur when they're split
            if (
              id.includes("react-router") ||
              id.includes("react-dom") ||
              (id.includes("react") && id.includes("node_modules"))
            ) {
              return "vendor-react";
            }
            // Separate React Query
            if (id.includes("@tanstack/react-query")) {
              return "vendor-query";
            }
            // Separate Radix UI components
            if (id.includes("@radix-ui")) {
              return "vendor-radix";
            }
            // Separate charting library (recharts)
            if (id.includes("recharts")) {
              return "vendor-charts";
            }
            // Separate animation library
            if (id.includes("framer-motion")) {
              return "vendor-animations";
            }
            // Separate other node_modules
            if (id.includes("node_modules")) {
              return "vendor";
            }
          },
          chunkFileNames: "assets/js/[name]-[hash].js",
          entryFileNames: "assets/js/[name]-[hash].js",
          assetFileNames: (assetInfo) => {
            const info = assetInfo.name?.split(".") || [];
            const ext = info[info.length - 1];
            if (/png|jpe?g|svg|gif|tiff|bmp|ico|webp/i.test(ext)) {
              return "assets/images/[name]-[hash][extname]";
            }
            if (/css/i.test(ext)) {
              return "assets/css/[name]-[hash][extname]";
            }
            return "assets/[name]-[hash][extname]";
          },
        },
      },
      chunkSizeWarningLimit: 1000,
      target: "esnext",
      cssCodeSplit: true,
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

      // Add DNS prefetch and preconnect for external resources
      const resourceHints = `
  <link rel="dns-prefetch" href="https://www.googletagmanager.com" />
  <link rel="preconnect" href="https://www.googletagmanager.com" crossorigin />
  <link rel="preconnect" href="https://pandects.org" />
  <link rel="preconnect" href="https://api.pandects.org" crossorigin />`;

      nextHtml = nextHtml.replace("</head>", `${resourceHints}</head>`);

      // Add preload hints for critical resources using bundle info if available
      if (ctx?.bundle) {
        const preloadHints: string[] = [];
        
        // Preload critical JavaScript bundles (main entry and vendor-react)
        for (const [fileName, chunk] of Object.entries(ctx.bundle)) {
          if (chunk.type === "chunk") {
            const chunkFileName = chunk.fileName;
            // Preload main entry point
            if (chunkFileName.includes("main") || chunkFileName.includes("index")) {
              preloadHints.push(
                `  <link rel="modulepreload" href="/${chunkFileName}" />`,
              );
            }
            // Preload vendor-react early (needed for hydration)
            if (chunkFileName.includes("vendor-react")) {
              preloadHints.push(
                `  <link rel="modulepreload" href="/${chunkFileName}" />`,
              );
            }
          } else if (chunk.type === "asset") {
            const assetFileName = chunk.fileName;
            // Preload LCP image
            if (assetFileName.includes("logo-128")) {
              if (assetFileName.endsWith(".webp")) {
                preloadHints.push(
                  `  <link rel="preload" href="/${assetFileName}" as="image" type="image/webp" fetchpriority="high" />`,
                );
              } else if (assetFileName.endsWith(".png")) {
                preloadHints.push(
                  `  <link rel="preload" href="/${assetFileName}" as="image" type="image/png" fetchpriority="high" />`,
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
          if (chunk.type === "chunk") {
            const chunkFileName = chunk.fileName;
            // Preload main entry point
            if (chunkFileName.includes("main") || chunkFileName.includes("index")) {
              if (!html.includes(`rel="modulepreload" href="/${chunkFileName}"`)) {
                preloadHints.push(
                  `  <link rel="modulepreload" href="/${chunkFileName}" />`,
                );
              }
            }
            // Preload vendor-react early
            if (chunkFileName.includes("vendor-react")) {
              if (!html.includes(`rel="modulepreload" href="/${chunkFileName}"`)) {
                preloadHints.push(
                  `  <link rel="modulepreload" href="/${chunkFileName}" />`,
                );
              }
            }
          } else if (chunk.type === "asset") {
            const assetFileName = chunk.fileName;
            // Preload LCP image
            if (assetFileName.includes("logo-128")) {
              if (assetFileName.endsWith(".webp")) {
                if (!html.includes(`rel="preload" href="/${assetFileName}"`)) {
                  preloadHints.push(
                    `  <link rel="preload" href="/${assetFileName}" as="image" type="image/webp" fetchpriority="high" />`,
                  );
                }
              } else if (assetFileName.endsWith(".png")) {
                if (!html.includes(`rel="preload" href="/${assetFileName}"`)) {
                  preloadHints.push(
                    `  <link rel="preload" href="/${assetFileName}" as="image" type="image/png" fetchpriority="high" />`,
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
