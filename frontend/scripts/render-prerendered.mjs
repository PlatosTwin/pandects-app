import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { buildJsonLd, getSeoConfigForPath } from "../shared/seo-helpers.mjs";

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

const routes = [
  { pathname: "/", filename: "index.html" },
  { pathname: "/about", filename: "about.html" },
  { pathname: "/bulk-data", filename: "bulk-data.html" },
  { pathname: "/donate", filename: "donate.html" },
  { pathname: "/feedback", filename: "feedback.html" },
  { pathname: "/sources-methods", filename: "sources-methods.html" },
];

for (const route of routes) {
  const rendered = renderPage(route.pathname);
  const html = injectSeo(
    injectRootHtml(indexHtml, rendered),
    buildSeoBlock(route.pathname),
  );
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

function injectSeo(html, seoBlock) {
  const marker = /<!-- SEO:BEGIN -->[\s\S]*?<!-- SEO:END -->/;
  if (!marker.test(html)) {
    throw new Error("Expected SEO markers in dist/spa/index.html");
  }
  return html.replace(marker, `<!-- SEO:BEGIN -->\n${seoBlock}\n  <!-- SEO:END -->`);
}

function buildSeoBlock(pathname) {
  const seo = getSeoConfigForPath(pathname, origin);
  const jsonLd = JSON.stringify(
    buildJsonLd({
      origin,
      canonical: seo.canonical,
      pageType: seo.pageType,
      pageName: seo.pageName,
      pageDescription: seo.pageDescription,
    }),
    null,
    2,
  );
  const description = escapeHtml(seo.description);
  const title = escapeHtml(seo.title);
  const canonical = escapeHtml(seo.canonical);
  const ogImage = escapeHtml(seo.ogImage);
  const robots = escapeHtml(seo.robots);

  return [
    `  <meta name="description" content="${description}" />`,
    `  <link rel="canonical" href="${canonical}" />`,
    `  <meta name="robots" content="${robots}" />`,
    "",
    `  <meta property="og:type" content="website" />`,
    `  <meta property="og:locale" content="en_US" />`,
    `  <meta property="og:site_name" content="Pandects" />`,
    `  <meta property="og:title" content="${title}" />`,
    `  <meta property="og:description" content="${description}" />`,
    `  <meta property="og:url" content="${canonical}" />`,
    `  <meta property="og:image" content="${ogImage}" />`,
    `  <meta property="og:image:alt" content="Pandects" />`,
    "",
    `  <meta name="twitter:card" content="summary_large_image" />`,
    `  <meta name="twitter:title" content="${title}" />`,
    `  <meta name="twitter:description" content="${description}" />`,
    `  <meta name="twitter:image" content="${ogImage}" />`,
    `  <meta name="twitter:image:alt" content="Pandects" />`,
    "",
    "  <script type=\"application/ld+json\">",
    `${jsonLd.replace(/\n/g, "\n  ")}`,
    "  </script>",
  ].join("\n");
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}
