import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const frontendRoot = path.resolve(process.cwd());
const globalCss = fs.readFileSync(
  path.join(frontendRoot, "client/global.css"),
  "utf8",
);
const criticalCss = fs.readFileSync(
  path.join(frontendRoot, "client/critical.css"),
  "utf8",
);

function readRule(css: string, selector: string): string {
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = css.match(
    new RegExp(`${escapedSelector}\\s*\\{([\\s\\S]*?)\\n\\s*\\}`, "m"),
  );

  expect(match, `Missing ${selector} rule`).not.toBeNull();
  return match?.[1] ?? "";
}

function readDeclarations(rule: string): Map<string, string> {
  return new Map(
    [...rule.matchAll(/(--[a-z0-9-]+):\s*([^;]+);/g)].map((match) => [
      match[1],
      match[2].trim(),
    ]),
  );
}

function readFontFamily(rule: string): string {
  const match = rule.match(/font-family:\s*([\s\S]*?);/m);

  expect(match, "Missing font-family declaration").not.toBeNull();
  return (match?.[1] ?? "").replace(/\s+/g, " ").trim();
}

/** WCAG relative luminance of an `H S% L%` token value. */
function hslTokenLuminance(value: string): number {
  const [h, sPct, lPct] = value.replace(/%/g, "").split(/\s+/).map(Number);
  const sat = sPct / 100;
  const light = lPct / 100;
  const a = sat * Math.min(light, 1 - light);
  const channel = (n: number) => {
    const k = (n + h / 30) % 12;
    const c = light - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(0) + 0.7152 * channel(8) + 0.0722 * channel(4);
}

function contrastRatio(a: string, b: string): number {
  const [hi, lo] = [hslTokenLuminance(a), hslTokenLuminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

describe("chart palette", () => {
  it("keeps every light-mode chart color at 3:1 or more against card and background", () => {
    const tokens = readDeclarations(readRule(globalCss, ":root"));
    const chartTokens = [...tokens.keys()].filter((token) => token.startsWith("--chart-"));
    expect(chartTokens).toHaveLength(9);
    for (const surface of ["--card", "--background"]) {
      for (const token of chartTokens) {
        const ratio = contrastRatio(tokens.get(token) ?? "", tokens.get(surface) ?? "");
        expect(ratio, `${token} on ${surface}`).toBeGreaterThanOrEqual(3);
      }
    }
  });
});

describe("critical CSS", () => {
  it("keeps duplicated light and dark theme tokens aligned with global CSS", () => {
    for (const selector of [":root", ".dark"]) {
      const globalDeclarations = readDeclarations(readRule(globalCss, selector));
      const criticalDeclarations = readDeclarations(readRule(criticalCss, selector));

      // Bidirectional: a token added to either file must be mirrored in the other.
      expect(
        [...criticalDeclarations.keys()].sort(),
        `${selector} token set`,
      ).toEqual([...globalDeclarations.keys()].sort());

      for (const [token, criticalValue] of criticalDeclarations) {
        expect(globalDeclarations.get(token), `${selector} ${token}`).toBe(
          criticalValue,
        );
      }
    }
  });

  it("keeps the critical html rule aligned with the app html rule", () => {
    const globalHtml = readRule(globalCss, "html");
    const criticalHtml = readRule(criticalCss, "html");

    for (const property of ["font-size", "scrollbar-gutter"]) {
      const read = (rule: string) =>
        rule.match(new RegExp(`${property}:\\s*([^;]+);`))?.[1].trim();

      expect(read(criticalHtml), `html ${property}`).toBe(read(globalHtml));
      expect(read(globalHtml), `html ${property}`).toBeDefined();
    }
  });

  it("keeps the production critical body font stack aligned with the app body font stack", () => {
    expect(readFontFamily(readRule(criticalCss, "body"))).toBe(
      readFontFamily(readRule(globalCss, "body")),
    );
  });
});
