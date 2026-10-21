import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const clientRoot = path.resolve(process.cwd(), "client");

/**
 * Surfaces where the user is mid-task (searching, reading an agreement,
 * accepting terms). A plain in-app link to a reference page from here throws
 * away their results, filters and scroll position.
 */
const WORKING_SURFACES = [
  "pages/search",
  "pages/Search.tsx",
  "pages/Favorites.tsx",
  "components/AgreementReader.tsx",
  "components/reader",
  "components/SearchResultsTable.tsx",
  "components/taxonomy/TaxonomyNodePanel.tsx",
  "components/auth",
];

/** In-app pages people consult while working, not destinations. */
const REFERENCE_ROUTES = [
  "/taxonomy",
  "/sources-methods",
  "/xml-schema",
  "/support",
  "/about",
  "/bulk-data",
  "/license",
  "/privacy-policy",
  "/terms",
];

function sourceFiles(entry: string): string[] {
  const full = path.join(clientRoot, entry);
  if (fs.statSync(full).isFile()) return [full];
  return fs
    .readdirSync(full, { recursive: true, encoding: "utf8" })
    .filter((name) => /\.tsx?$/.test(name) && !/\.spec\.tsx?$/.test(name))
    .map((name) => path.join(full, name));
}

describe("reference links", () => {
  it("open reference pages from working surfaces in a new tab", () => {
    const routes = REFERENCE_ROUTES.map((route) => route.replace(/[/-]/g, "\\$&")).join("|");
    const plainLink = new RegExp(`<Link\\b[^>]*?\\bto=["'{\`]+(${routes})(?=["'\`/?#])`, "gs");
    const navigation = new RegExp(`navigate\\(\\s*["'\`](${routes})(?=["'\`/?#])`, "g");
    const offenders: string[] = [];
    for (const file of WORKING_SURFACES.flatMap(sourceFiles)) {
      const source = fs.readFileSync(file, "utf8");
      for (const match of [...source.matchAll(plainLink), ...source.matchAll(navigation)]) {
        offenders.push(`${path.relative(clientRoot, file)}: ${match[1]}`);
      }
    }
    expect(offenders, "Use ReferenceLink (opens in a new tab) for these").toEqual([]);
  });
});
