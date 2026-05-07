import express from "express";
import path from "path";
import fs from "fs";
import compression from "compression";
import { handleDemo } from "./routes/demo";
import { getPublicOrigin, getSeoForRequest, injectSeoDocument } from "./seo";
import { FRONTEND_SECURITY_HEADERS } from "./security-headers";
import { PRERENDER_ROUTES } from "../shared/route-manifest.mjs";

const DOCS_SITE_URL = (process.env.PUBLIC_DOCS_URL || "https://docs.pandects.org")
  .trim()
  .replace(/\/+$/, "");

export function createServer() {
  const app = express();

  app.disable("x-powered-by");

  // Enable compression for all responses in production
  if (process.env.NODE_ENV === "production") {
    app.use(
      compression({
        filter: (req, res) => {
          // Compress all text-based responses
          if (req.headers["x-no-compression"]) {
            return false;
          }
          return compression.filter(req, res);
        },
        level: 6, // Balance between compression ratio and CPU usage
      }),
    );
  }

  app.use((_req, res, next) => {
    for (const [name, value] of Object.entries(FRONTEND_SECURITY_HEADERS)) {
      if (name === "Strict-Transport-Security" && process.env.NODE_ENV !== "production") {
        continue;
      }
      if (name === "Content-Security-Policy" && process.env.NODE_ENV !== "production") {
        continue;
      }
      res.setHeader(name, value);
    }
    next();
  });

  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  // Frontend-owned utility endpoints live under /api/.
  app.get("/api/ping", (_req, res) => {
    res.json({ message: "Hello from Express server v2!" });
  });

  app.get("/api/demo", handleDemo);

  if (process.env.NODE_ENV === "production") {
    const staticPath = path.resolve(import.meta.dirname, "../spa");
    const indexHtmlPath = path.join(staticPath, "index.html");
    const indexHtmlTemplate = fs.readFileSync(indexHtmlPath, "utf-8");
    const prerenderedTemplates = loadPrerenderedTemplates(staticPath);

    app.use(express.static(staticPath));

    app.get("*", (req, res) => {
      if (req.path.startsWith("/v1/")) {
        res.status(404).end();
        return;
      }

      if (req.path === "/docs" || req.path.startsWith("/docs/")) {
        const docsSuffix = req.path.slice("/docs".length);
        const queryIndex = req.originalUrl.indexOf("?");
        const query = queryIndex >= 0 ? req.originalUrl.slice(queryIndex) : "";
        res.redirect(301, `${DOCS_SITE_URL}${docsSuffix}${query}`);
        return;
      }

      if (req.path !== "/" && req.path.endsWith("/")) {
        const [pathname, query] = req.originalUrl.split("?");
        const canonicalPath = (pathname ?? req.path).replace(/\/+$/, "");
        const location = query ? `${canonicalPath}?${query}` : canonicalPath;
        res.redirect(301, location);
        return;
      }

      const acceptsHtml = req.accepts(["html"]) === "html";
      if (!acceptsHtml) {
        res.status(404).end();
        return;
      }

      const origin = getPublicOrigin(req);
      const seo = getSeoForRequest(req, origin);
      const template = prerenderedTemplates.get(req.path) ?? indexHtmlTemplate;
      const html = injectSeoDocument(template, seo);

      if (seo.xRobotsTag) res.setHeader("X-Robots-Tag", seo.xRobotsTag);
      res.status(seo.status);

      res.setHeader("Content-Type", "text/html; charset=utf-8");
      res.send(html);
    });
  }

  return app;
}

function loadPrerenderedTemplates(staticPath: string): Map<string, string> {
  const templates = new Map<string, string>();
  const dir = path.join(staticPath, "prerender");
  for (const route of PRERENDER_ROUTES) {
    const filePath = path.join(dir, route.prerenderFilename);
    if (!fs.existsSync(filePath)) continue;
    templates.set(route.pathname, fs.readFileSync(filePath, "utf-8"));
  }

  return templates;
}
