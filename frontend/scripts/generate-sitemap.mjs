import fs from "node:fs";
import path from "node:path";

const DEFAULT_ORIGIN = "https://pandects.org";
const origin = (process.env.PUBLIC_ORIGIN || DEFAULT_ORIGIN).replace(/\/+$/, "");

const routes = ["/", "/search", "/docs", "/bulk-data", "/about", "/feedback", "/donate"];

const now = new Date();
const lastmod = now.toISOString().slice(0, 10);

const urls = routes
  .map((route) => {
    const loc = route === "/" ? `${origin}/` : `${origin}${route}`;
    return [
      "  <url>",
      `    <loc>${escapeXml(loc)}</loc>`,
      `    <lastmod>${lastmod}</lastmod>`,
      "    <changefreq>weekly</changefreq>",
      "    <priority>0.7</priority>",
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
