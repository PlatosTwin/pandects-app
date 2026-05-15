import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { buildSeoPage, injectSeoDocument } from "../shared/seo-helpers.mjs";
import { PRERENDER_ROUTES } from "../shared/route-manifest.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const prerenderBundlePath = path.resolve(__dirname, "../dist/prerender/prerender.mjs");
const { renderPage } = await import(pathToFileURL(prerenderBundlePath).href);

const DEFAULT_ORIGIN = "https://pandects.org";
const origin = (process.env.PUBLIC_ORIGIN || DEFAULT_ORIGIN).replace(/\/+$/, "");

const spaDir = path.resolve(__dirname, "../dist/spa");
const indexHtmlPath = path.join(spaDir, "index.html");
const indexHtml = fs.readFileSync(indexHtmlPath, "utf-8");

const outDir = path.join(spaDir, "prerender");
fs.mkdirSync(outDir, { recursive: true });

for (const route of PRERENDER_ROUTES) {
  const rendered = renderPage(route.pathname);
  const html = injectSeoDocument(injectRootHtml(indexHtml, rendered), buildSeoPage(route.pathname, "", origin));
  fs.writeFileSync(path.join(outDir, route.prerenderFilename), html, "utf-8");
}

// SSR renders may schedule timers (e.g. React Query GC, retry backoff) that
// keep the Node event loop alive even though the file output is complete.
// Exit explicitly so the build pipeline doesn't stall.
process.exit(0);

function injectRootHtml(html, rendered) {
  const needle = '<div id="root"></div>';
  const replacement = `<div id="root">${rendered}</div>`;
  if (!html.includes(needle)) {
    throw new Error(`Expected ${needle} in dist/spa/index.html`);
  }
  return html.replace(needle, replacement);
}
