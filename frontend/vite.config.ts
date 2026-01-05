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
            // Split vendor chunks for better caching
            if (id.includes("node_modules")) {
              // Large icon library - split into its own chunk
              if (id.includes("lucide-react")) {
                return "vendor-icons";
              }
              // React and React DOM
              if (id.includes("react") || id.includes("react-dom")) {
                return "vendor-react";
              }
              // React Router
              if (id.includes("react-router")) {
                return "vendor-router";
              }
              // TanStack Query
              if (id.includes("@tanstack/react-query")) {
                return "vendor-query";
              }
              // Radix UI components (large library)
              if (id.includes("@radix-ui")) {
                return "vendor-radix";
              }
              // Other vendor code
              return "vendor";
            }
            // Split large page components
            if (id.includes("/pages/")) {
              const pageName = id.split("/pages/")[1]?.split(".")[0];
              // Keep large pages in separate chunks
              if (["SourcesMethods", "Search", "Account"].includes(pageName)) {
                return `page-${pageName.toLowerCase()}`;
              }
            }
          },
          chunkFileNames: (chunkInfo) => {
            // Organize chunks by type
            const facadeModuleId = chunkInfo.facadeModuleId
              ? chunkInfo.facadeModuleId.split("/").pop()?.replace(/\.[^.]*$/, "")
              : "chunk";
            if (chunkInfo.name?.startsWith("vendor")) {
              return `assets/vendor/${chunkInfo.name}-[hash].js`;
            }
            if (chunkInfo.name?.startsWith("page-")) {
              return `assets/pages/${chunkInfo.name}-[hash].js`;
            }
            return `assets/${facadeModuleId}-[hash].js`;
          },
          assetFileNames: (assetInfo) => {
            // Organize assets
            if (assetInfo.name?.endsWith(".css")) {
              return "assets/css/[name]-[hash][extname]";
            }
            if (assetInfo.name?.match(/\.(png|jpe?g|svg|gif|webp|avif)$/)) {
              return "assets/images/[name]-[hash][extname]";
            }
            return "assets/[name]-[hash][extname]";
          },
        },
      },
      // Increase chunk size warning limit since we're splitting manually
      chunkSizeWarningLimit: 1000,
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

      // Add DNS prefetch for external resources
      const dnsPrefetch = `
  <link rel="dns-prefetch" href="https://www.googletagmanager.com" />
  <link rel="preconnect" href="https://www.googletagmanager.com" crossorigin />`;

      nextHtml = nextHtml.replace("</head>", `${dnsPrefetch}</head>`);

      // Preload critical assets (WebP with PNG fallback)
      const preloadHints = `
  <link rel="preload" href="/assets/logo-128.webp" as="image" type="image/webp" />
  <link rel="preload" href="/assets/logo-128.png" as="image" type="image/png" />`;

      nextHtml = nextHtml.replace("</head>", `${preloadHints}</head>`);

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
