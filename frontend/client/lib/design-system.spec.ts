import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const clientRoot = path.resolve(process.cwd(), "client");

const sources = fs
  .readdirSync(clientRoot, { recursive: true, encoding: "utf8" })
  .filter((name) => name.endsWith(".tsx") && !/\.spec\.tsx?$/.test(name))
  .map((name) => ({ name, source: fs.readFileSync(path.join(clientRoot, name), "utf8") }));

/** Every double-quoted class string in a file. */
function classStrings(source: string): string[] {
  return [...source.matchAll(/"([^"\n]*)"/g)].map((match) => match[1]);
}

describe("design system", () => {
  // Shadows come from one scale (tailwind.config.ts boxShadow, values per theme
  // in global.css). Ad hoc shadow-sm/md/lg drifted until most of the app was
  // flat; the scale keeps depth consistent in both themes.
  it("uses the elevation scale instead of raw shadow sizes", () => {
    const offenders = sources.flatMap(({ name, source }) =>
      classStrings(source).flatMap((classes) =>
        [...classes.matchAll(/(?<![\w-])shadow(-(sm|md|lg|xl|2xl))?(?![\w/[-])/g)].map(
          (match) => `${name}: ${match[0]}`,
        ),
      ),
    );
    expect(offenders).toEqual([]);
  });

  // A bordered, rounded card surface without elevation reads as flat next to
  // the ones that have it.
  it("gives every bordered card surface an elevation", () => {
    const offenders = sources.flatMap(({ name, source }) =>
      classStrings(source)
        .filter(
          (classes) =>
            /(?<![\w-])bg-card(\/\d+)?(?![\w-])/.test(classes) &&
            /(?<![\w-])rounded-(md|lg|xl|2xl)(?![\w-])/.test(classes) &&
            /(?<![\w:-])border(?![\w-])/.test(classes) &&
            !/(?<![\w-])shadow-/.test(classes),
        )
        .map((classes) => `${name}: ${classes}`),
    );
    expect(offenders).toEqual([]);
  });

  // The nav, footer, page shells and the reader share one frame (Search's
  // workspace uses app-frame-bleed). A copied max-width literal is how search
  // and the nav drifted apart. The nav never swaps frames per route: that
  // shifted the logo and links whenever you moved between Search and any
  // other page.
  it("sizes page frames with app-frame only", () => {
    const offenders = sources
      .filter(({ source }) => /(?<![\w-])max-w-7xl(?![\w-])/.test(source))
      .map(({ name }) => name);
    expect(offenders).toEqual([]);
    for (const file of ["components/Navigation.tsx", "components/Footer.tsx", "components/PageShell.tsx"]) {
      expect(fs.readFileSync(path.join(clientRoot, file), "utf8"), file).toContain("app-frame");
    }
    expect(fs.readFileSync(path.join(clientRoot, "pages/search/SearchQueryBar.tsx"), "utf8")).toContain(
      "app-frame-bleed",
    );
    expect(fs.readFileSync(path.join(clientRoot, "components/Navigation.tsx"), "utf8")).not.toContain(
      "app-frame-bleed",
    );
  });

  // Stacked segments touch, so two neighbors of similar lightness (green on
  // amber) merge for color-blind readers. A card-colored edge separates them
  // without leaning on the colors' contrast with each other.
  it("separates stacked chart segments with a card-colored edge", () => {
    const offenders = sources.flatMap(({ name, source }) =>
      [...source.matchAll(/<(Bar|Area)\b[^>]*?\bstackId=[^>]*?\/?>/gs)]
        .filter(([element]) => !/stroke="(hsl\(var\(--card\)\)|transparent)"/.test(element))
        .map(([element]) => `${name}: ${element.replace(/\s+/g, " ").slice(0, 80)}`),
    );
    expect(offenders).toEqual([]);
  });
});
