import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { checkBudgets, findEntryAssets } from "./check-asset-budgets.mjs";

const dirs = [];

function makeBuild({ html, files }) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "budget-spec-"));
  dirs.push(dir);
  fs.writeFileSync(path.join(dir, "index.html"), html);
  for (const [name, size] of Object.entries(files)) {
    const filePath = path.join(dir, name);
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, "x".repeat(size));
  }
  return dir;
}

afterEach(() => {
  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

const HTML = `<!doctype html><html><head>
<script type="module" data-cfasync="false" crossorigin src="/assets/js/index-REAL.js"></script>
<link rel="stylesheet" crossorigin href="/assets/css/index-REAL.css">
<link rel="preload" href="/assets/logo.webp" as="image">
</head><body></body></html>`;

describe("findEntryAssets", () => {
  it("reads the module script and stylesheet from index.html", () => {
    expect(findEntryAssets(HTML)).toEqual({
      js: "/assets/js/index-REAL.js",
      css: "/assets/css/index-REAL.css",
    });
  });

  it("skips external stylesheets and scripts", () => {
    const html = `<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist">
<script type="module" src="https://cdn.example.com/x.js"></script>${HTML}`;
    expect(findEntryAssets(html)).toEqual({
      js: "/assets/js/index-REAL.js",
      css: "/assets/css/index-REAL.css",
    });
  });

  it("returns null when a tag is missing", () => {
    expect(findEntryAssets("<html></html>")).toEqual({ js: null, css: null });
  });
});

describe("checkBudgets", () => {
  it("measures the file index.html loads, not the first index-*.js", () => {
    const dir = makeBuild({
      html: HTML,
      files: {
        "assets/js/index-AAA.js": 10,
        "assets/js/index-REAL.js": 500,
        "assets/css/index-REAL.css": 20,
      },
    });
    const { results, failures } = checkBudgets(dir, { js: 1000, css: 1000 });
    expect(failures).toEqual([]);
    expect(results[0].size).toBe(500);
    expect(path.basename(results[0].filePath)).toBe("index-REAL.js");
  });

  it("fails when the real entry is over budget even if another index-*.js is small", () => {
    const dir = makeBuild({
      html: HTML,
      files: {
        "assets/js/index-AAA.js": 10,
        "assets/js/index-REAL.js": 500,
        "assets/css/index-REAL.css": 20,
      },
    });
    const { failures } = checkBudgets(dir, { js: 400, css: 1000 });
    expect(failures).toHaveLength(1);
    expect(failures[0]).toContain(
      "entry JavaScript exceeded budget: 500 bytes > 400 bytes",
    );
  });

  it("throws when index.html has no module script", () => {
    const dir = makeBuild({
      html: '<link rel="stylesheet" href="/assets/css/index-REAL.css">',
      files: { "assets/css/index-REAL.css": 5 },
    });
    expect(() => checkBudgets(dir)).toThrow(/entry JavaScript/);
  });

  it("throws when the named file does not exist", () => {
    const dir = makeBuild({
      html: HTML,
      files: { "assets/css/index-REAL.css": 5 },
    });
    expect(() => checkBudgets(dir)).toThrow(/does not exist/);
  });

  it("throws when index.html is missing", () => {
    const dir = makeBuild({ html: "", files: {} });
    fs.rmSync(path.join(dir, "index.html"));
    expect(() => checkBudgets(dir)).toThrow(/Run the client build first/);
  });

  it("refuses a script path that leaves the build folder", () => {
    const dir = makeBuild({
      html: '<script type="module" src="/../outside.js"></script><link rel="stylesheet" href="/a.css">',
      files: { "a.css": 1 },
    });
    expect(() => checkBudgets(dir)).toThrow(/outside/);
  });
});
