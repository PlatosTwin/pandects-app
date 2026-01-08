import fs from "node:fs";
import path from "node:path";

const DEFAULT_ORIGIN = "https://pandects.org";
const origin = (process.env.PUBLIC_ORIGIN || DEFAULT_ORIGIN).replace(/\/+$/, "");

const routeConfig = {
  "/": { priority: "1.0", changefreq: "daily" },
  "/search": { priority: "0.9", changefreq: "weekly" },
  "/docs": { priority: "0.9", changefreq: "weekly" },
  "/bulk-data": { priority: "0.8", changefreq: "weekly" },
  "/agreement-index": { priority: "0.8", changefreq: "weekly" },
  "/sources-methods": { priority: "0.8", changefreq: "monthly" },
  "/xml-schema": { priority: "0.7", changefreq: "monthly" },
  "/about": { priority: "0.7", changefreq: "monthly" },
  "/feedback": { priority: "0.6", changefreq: "monthly" },
  "/donate": { priority: "0.6", changefreq: "monthly" },
  "/privacy-policy": { priority: "0.5", changefreq: "yearly" },
  "/terms": { priority: "0.5", changefreq: "yearly" },
  "/license": { priority: "0.5", changefreq: "yearly" },
};

const routes = Object.keys(routeConfig);

const now = new Date();
const lastmod = now.toISOString().slice(0, 10);

const urls = routes
  .map((route) => {
    const loc = route === "/" ? `${origin}/` : `${origin}${route}`;
    const config = routeConfig[route];
    return [
      "  <url>",
      `    <loc>${escapeXml(loc)}</loc>`,
      `    <lastmod>${lastmod}</lastmod>`,
      `    <changefreq>${config.changefreq}</changefreq>`,
      `    <priority>${config.priority}</priority>`,
      "  </url>",
    ].join("\n");
  })
  .join("\n");

const xml = [
  '<?xml version="1.0" encoding="UTF-8"?>',
  '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
  urls,
  "</urlset>",
  "",
].join("\n");

const outPath = path.resolve("dist/spa/sitemap.xml");
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, xml, "utf-8");

function escapeXml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}
