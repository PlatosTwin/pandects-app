import express from "express";
import cors from "cors";
import path from "path";
import fs from "fs";
import { handleDemo } from "./routes/demo";
import { getPublicOrigin, getSeoForPath, injectSeoBlock, isKnownRoute } from "./seo";

export function createServer() {
  const app = express();

  // Middleware
  app.use(cors());
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  // Example API routes
  app.get("/api/ping", (_req, res) => {
    res.json({ message: "Hello from Express server v2!" });
  });

  app.get("/api/demo", handleDemo);

  // Serve static files in production
  if (process.env.NODE_ENV === "production") {
    const staticPath = path.join(__dirname, "../spa");
    const indexHtmlPath = path.join(staticPath, "index.html");
    const indexHtmlTemplate = fs.readFileSync(indexHtmlPath, "utf-8");
    const prerenderedTemplates = loadPrerenderedTemplates(staticPath);

    app.use(express.static(staticPath));

    // SPA fallback - serve index.html for all non-API routes
    app.get("*", (req, res) => {
      if (req.path.startsWith("/api/")) {
        res.status(404).end();
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
      const seo = getSeoForPath(req.path, origin);
      const template = prerenderedTemplates.get(req.path) ?? indexHtmlTemplate;
      const html = injectSeoBlock(template, seo);

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
  const entries = [
    { route: "/about", file: "about.html" },
    { route: "/bulk-data", file: "bulk-data.html" },
  ];

  const dir = path.join(staticPath, "prerender");
  for (const entry of entries) {
    const filePath = path.join(dir, entry.file);
    if (!fs.existsSync(filePath)) continue;
    templates.set(entry.route, fs.readFileSync(filePath, "utf-8"));
  }

  return templates;
}
