import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { describe, expect, it } from "vitest";

const frontendRoot = path.resolve(process.cwd());
const globalCss = fs.readFileSync(path.join(frontendRoot, "client/global.css"), "utf8");

/** Body declarations of the app's scroll-lock override. */
function overrideDeclarations(): Map<string, string> {
  const match = globalCss.match(
    /html body\[data-scroll-locked\]:not\(:has\(\[role="dialog"\], \[role="alertdialog"\]\)\)\s*\{([^}]*)\}/,
  );
  expect(match, "Missing the menu-only scroll-lock override in global.css").not.toBeNull();
  return new Map(
    [...(match?.[1] ?? "").matchAll(/([a-z-]+):\s*([^;]+);/g)].map((m) => [m[1], m[2].trim()]),
  );
}

describe("scroll lock", () => {
  // Radix menus and selects lock scroll through react-remove-scroll-bar. Its
  // body rule hides the scrollbar and adds a scrollbar-width margin, which can
  // shift the page sideways under our stable scrollbar gutter. The override is
  // scoped away from dialogs and sheets, which need the full lock.
  it("keeps the page scrollbar and body box unchanged while a menu is open", () => {
    const declarations = overrideDeclarations();
    expect(declarations.get("overflow-x")).toBe("hidden !important");
    expect(declarations.get("overflow-y")).toBe("visible !important");
    expect(declarations.get("margin-right")).toBe("0 !important");
    expect(declarations.get("padding-right")).toBe("0 !important");
  });

  it("matches the attribute and selector weight the library injects", () => {
    const require = createRequire(import.meta.url);
    const entry = require.resolve("react-remove-scroll-bar");
    const source = fs.readFileSync(path.join(path.dirname(entry), "component.js"), "utf8");
    // If the library renames the attribute, the override above silently stops
    // matching; if it raises its selector weight, ours stops winning.
    expect(source).toContain("lockAttribute = 'data-scroll-locked'");
    expect(source).toMatch(/[^.\w]body\["\)\.concat\((exports\.)?lockAttribute, "\] \{/);
  });
});
