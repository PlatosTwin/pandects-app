import fs from "node:fs";
import path from "node:path";
import { PRERENDER_ROUTES } from "../shared/route-manifest.mjs";

const spaDir = path.resolve("dist/spa");
const prerenderDir = path.join(spaDir, "prerender");
const shellHtml = fs.readFileSync(path.join(spaDir, "index.html"), "utf-8");
const failures = [];

assert(shellHtml.includes("<!-- SEO:BEGIN -->") && shellHtml.includes("<!-- SEO:END -->"),
  "dist/spa/index.html is missing SEO block markers.");

for (const route of PRERENDER_ROUTES) {
  const filePath = path.join(prerenderDir, route.prerenderFilename);
  if (!fs.existsSync(filePath)) {
    failures.push(`Missing prerendered file for ${route.pathname}: ${route.prerenderFilename}`);
    continue;
  }

  const html = fs.readFileSync(filePath, "utf-8");
  assert(!html.includes('<div id="root"></div>'), `${route.pathname} prerendered output has an empty root.`);
  assert(!html.includes('data-msg="'), `${route.pathname} prerendered output contains a React server-render error marker.`);
  assert(!html.includes("<!--$!-->"), `${route.pathname} prerendered output contains a React error suspense marker.`);
  assert(countMatches(html, /<link rel="canonical" /g) === 1, `${route.pathname} should have exactly one canonical tag.`);
  assert(countMatches(html, /<meta name="description" /g) === 1, `${route.pathname} should have exactly one meta description.`);
  assert(
    countMatches(html, /<script type="application\/ld\+json">/g) === 1,
    `${route.pathname} should have exactly one JSON-LD script.`,
  );
  assert(html.includes(`<title>${escapeRegExpText(route.title)}</title>`) || html.includes(`<title>${route.title}</title>`),
    `${route.pathname} prerendered output is missing the route title.`);
}

if (failures.length > 0) {
  console.error("SEO validation failed:");
  for (const failure of failures) {
    console.error(`- ${failure}`);
  }
  process.exit(1);
}

console.log(`Validated SEO output for ${PRERENDER_ROUTES.length} prerendered routes.`);

function assert(condition, message) {
  if (!condition) {
    failures.push(message);
  }
}

function countMatches(value, pattern) {
  return [...value.matchAll(pattern)].length;
}

function escapeRegExpText(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
