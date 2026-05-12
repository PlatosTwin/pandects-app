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
    [...rule.matchAll(/(--[a-z-]+):\s*([^;]+);/g)].map((match) => [
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

describe("critical CSS", () => {
  it("keeps duplicated light and dark theme tokens aligned with global CSS", () => {
    for (const selector of [":root", ".dark"]) {
      const globalDeclarations = readDeclarations(readRule(globalCss, selector));
      const criticalDeclarations = readDeclarations(readRule(criticalCss, selector));

      for (const [token, criticalValue] of criticalDeclarations) {
        expect(globalDeclarations.get(token), `${selector} ${token}`).toBe(
          criticalValue,
        );
      }
    }
  });

  it("keeps the production critical body font stack aligned with the app body font stack", () => {
    expect(readFontFamily(readRule(criticalCss, "body"))).toBe(
      readFontFamily(readRule(globalCss, "body")),
    );
  });
});
