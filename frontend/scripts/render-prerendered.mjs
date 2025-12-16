import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const prerenderBundlePath = path.resolve(__dirname, "../dist/prerender/prerender.mjs");
const { renderPage } = await import(pathToFileURL(prerenderBundlePath).href);

const spaDir = path.resolve(__dirname, "../dist/spa");
const indexHtmlPath = path.join(spaDir, "index.html");
const indexHtml = fs.readFileSync(indexHtmlPath, "utf-8");

const outDir = path.join(spaDir, "prerender");
fs.mkdirSync(outDir, { recursive: true });

const routes = [
  { pathname: "/about", filename: "about.html" },
  { pathname: "/bulk-data", filename: "bulk-data.html" },
];

for (const route of routes) {
  const rendered = renderPage(route.pathname);
  const html = injectRootHtml(indexHtml, rendered);
  fs.writeFileSync(path.join(outDir, route.filename), html, "utf-8");
}

function injectRootHtml(html, rendered) {
  const needle = '<div id="root"></div>';
  const replacement = `<div id="root">${rendered}</div>`;
  if (!html.includes(needle)) {
    throw new Error(`Expected ${needle} in dist/spa/index.html`);
  }
  return html.replace(needle, replacement);
}

