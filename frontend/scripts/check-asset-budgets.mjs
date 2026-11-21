import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const JS_BUDGET_BYTES = 425_000;
export const CSS_BUDGET_BYTES = 125_000;

// The entry files are the ones dist/spa/index.html loads. Vite also emits other
// chunks named index-*.js (one per lazy route or folder), so a filename pattern
// cannot tell them apart. Read the names from the built HTML instead.
export function findEntryAssets(html) {
  // External URLs (for example Google Fonts) are not build output; skip them.
  const isLocal = (url) => url && !/^[a-z][a-z0-9+.-]*:|^\/\//i.test(url);
  const tags = html.match(/<(?:script|link)\b[^>]*>/gi) ?? [];
  let js = null;
  let css = null;
  for (const tag of tags) {
    if (/^<script\b/i.test(tag) && /\btype\s*=\s*["']module["']/i.test(tag)) {
      const src = attr(tag, "src");
      if (isLocal(src)) js ??= src;
    } else if (
      /^<link\b/i.test(tag) &&
      /\brel\s*=\s*["']stylesheet["']/i.test(tag)
    ) {
      const href = attr(tag, "href");
      if (isLocal(href)) css ??= href;
    }
  }
  return { js, css };
}

function attr(tag, name) {
  const match = tag.match(
    new RegExp(`\\b${name}\\s*=\\s*["']([^"']+)["']`, "i"),
  );
  return match ? match[1] : null;
}

// Turn a URL from index.html (for example "/assets/js/index-abc.js") into a
// path inside the build folder. Refuse anything that points outside it.
export function resolveAsset(spaDir, url, label) {
  if (!url) {
    throw new Error(
      `Could not find the ${label} in ${path.join(spaDir, "index.html")}`,
    );
  }
  const clean = url.split(/[?#]/)[0].replace(/^\/+/, "");
  const filePath = path.resolve(spaDir, clean);
  if (!filePath.startsWith(path.resolve(spaDir) + path.sep)) {
    throw new Error(`The ${label} points outside ${spaDir}: ${url}`);
  }
  if (!fs.existsSync(filePath)) {
    throw new Error(
      `The ${label} named in index.html does not exist: ${filePath}`,
    );
  }
  return filePath;
}

export function checkBudgets(spaDir, budgets = {}) {
  const jsBudget = budgets.js ?? JS_BUDGET_BYTES;
  const cssBudget = budgets.css ?? CSS_BUDGET_BYTES;
  const htmlPath = path.join(spaDir, "index.html");
  if (!fs.existsSync(htmlPath)) {
    throw new Error(`Missing ${htmlPath}. Run the client build first.`);
  }
  const entries = findEntryAssets(fs.readFileSync(htmlPath, "utf8"));
  const entryJs = resolveAsset(
    spaDir,
    entries.js,
    "entry JavaScript (module script src)",
  );
  const entryCss = resolveAsset(
    spaDir,
    entries.css,
    "entry CSS (stylesheet link href)",
  );

  const results = [
    { label: "entry JavaScript", filePath: entryJs, budget: jsBudget },
    { label: "entry CSS", filePath: entryCss, budget: cssBudget },
  ].map((item) => ({ ...item, size: fs.statSync(item.filePath).size }));

  const failures = results
    .filter((item) => item.size > item.budget)
    .map(
      (item) =>
        `${item.label} exceeded budget: ${item.size} bytes > ${item.budget} bytes (${item.filePath})`,
    );
  return { results, failures };
}

function main() {
  const spaDir = path.resolve("dist/spa");
  let report;
  try {
    report = checkBudgets(spaDir);
  } catch (error) {
    console.error(`Performance budget validation failed: ${error.message}`);
    process.exit(1);
  }
  if (report.failures.length > 0) {
    console.error("Performance budget validation failed:");
    for (const failure of report.failures) {
      console.error(`- ${failure}`);
    }
    process.exit(1);
  }
  const summary = report.results
    .map(
      (item) =>
        `${path.basename(item.filePath)} is ${item.size} of ${item.budget} bytes (${item.budget - item.size} under)`,
    )
    .join("; ");
  console.log(`Validated asset budgets: ${summary}.`);
}

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  main();
}
