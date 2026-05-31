// Computes SHA-256 hashes of every inline <script> the SPA ever serves,
// so CSP can drop 'unsafe-inline' from script-src.
//
// Inputs (no Vite build required — all inline-script content is deterministic
// from source):
//   1. frontend/index.html                          (gtag bootstrap, static JSON-LD)
//   2. frontend/shared/route-manifest.mjs:PRERENDER_ROUTES   (per-route JSON-LD)
//
// Output:
//   frontend/shared/csp-script-hashes.generated.json — checked in so reviewers
//   see hash changes; consumed by validate-csp.mjs and (by hand) by nginx.conf /
//   netlify.toml at deploy time.

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

import { buildSeoPage, injectSeoDocument } from "../shared/seo-helpers.mjs";
import { PRERENDER_ROUTES } from "../shared/route-manifest.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const repoRoot = path.resolve(__dirname, "..");
const indexHtmlPath = path.join(repoRoot, "index.html");
const outputPath = path.join(repoRoot, "shared", "csp-script-hashes.generated.json");

const PRODUCTION_ORIGIN = "https://pandects.org";

// CSP hashes everything between <script ...> and </script> byte-for-byte. Inline
// <script> elements are those without a `src=` attribute. We deliberately ignore
// the `type` attribute — `<script type="application/ld+json">` is hashed the same
// way as a JS script.
const INLINE_SCRIPT_RE = /<script(?![^>]*\bsrc=)([^>]*)>([\s\S]*?)<\/script>/g;

function sha256Base64(content) {
  return crypto.createHash("sha256").update(content, "utf8").digest("base64");
}

function inlineScriptHashes(html) {
  const hashes = [];
  for (const match of html.matchAll(INLINE_SCRIPT_RE)) {
    const body = match[2];
    hashes.push({
      attrs: match[1].trim(),
      hash: `sha256-${sha256Base64(body)}`,
    });
  }
  return hashes;
}

function describeAttrs(attrs) {
  if (/type=["']application\/ld\+json["']/.test(attrs)) return "jsonld";
  if (/type=["']module["']/.test(attrs)) return "module";
  return "script";
}

const indexHtml = fs.readFileSync(indexHtmlPath, "utf8");

const baseInline = inlineScriptHashes(indexHtml);
const byScript = {};
const hashSet = new Set();

for (const { attrs, hash } of baseInline) {
  const kind = describeAttrs(attrs);
  const key = kind === "jsonld" ? "jsonld:index.html" : "gtag-bootstrap";
  byScript[key] = hash;
  hashSet.add(hash);
}

for (const route of PRERENDER_ROUTES) {
  const seo = buildSeoPage(route.pathname, "", PRODUCTION_ORIGIN);
  const rendered = injectSeoDocument(indexHtml, seo);
  for (const { attrs, hash } of inlineScriptHashes(rendered)) {
    if (describeAttrs(attrs) !== "jsonld") continue;
    byScript[`jsonld:${route.pathname}`] = hash;
    hashSet.add(hash);
  }
}

const manifest = {
  origin: PRODUCTION_ORIGIN,
  hashes: Array.from(hashSet).sort(),
  byScript,
};

const serialized = `${JSON.stringify(manifest, null, 2)}\n`;
const previous = fs.existsSync(outputPath) ? fs.readFileSync(outputPath, "utf8") : null;

if (previous === serialized) {
  process.stdout.write(`CSP script hashes unchanged (${manifest.hashes.length} entries).\n`);
} else {
  fs.writeFileSync(outputPath, serialized, "utf8");
  process.stdout.write(`Wrote ${manifest.hashes.length} CSP script hashes to ${path.relative(repoRoot, outputPath)}\n`);
  for (const hash of manifest.hashes) {
    process.stdout.write(`  '${hash}'\n`);
  }
}
