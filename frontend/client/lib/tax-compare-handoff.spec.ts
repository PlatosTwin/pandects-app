import { describe, expect, it } from "vitest";
import { buildCompareHref, compareOriginSearch } from "./tax-compare-handoff";

describe("tax compare origin", () => {
  it("carries the originating search through the compare link", () => {
    const href = buildCompareHref("/search?mode=tax&standard_id=t1&year=2023");
    const url = new URL(href, "https://example.test");
    expect(url.pathname).toBe("/compare/tax");
    const from = url.searchParams.get("from");
    expect(from).toBe("/search?mode=tax&standard_id=t1&year=2023");
    expect(compareOriginSearch(from)).toBe("/search?mode=tax&standard_id=t1&year=2023");
  });

  it("falls back to the tax search for missing or unsafe origins", () => {
    expect(buildCompareHref(null)).toBe("/compare/tax");
    for (const from of [null, "", "//evil.test", "/\t/evil.test", "/favorites", "https://evil.test/search"]) {
      expect(compareOriginSearch(from)).toBe("/search?mode=tax");
    }
  });
});
