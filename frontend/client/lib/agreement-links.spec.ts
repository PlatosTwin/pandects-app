import { describe, expect, it } from "vitest";
import {
  buildAgreementHref,
  originSearchParams,
  parseAgreementOrigin,
  safeOriginPath,
} from "./agreement-links";

describe("buildAgreementHref", () => {
  it("links to the reader with no params", () => {
    expect(buildAgreementHref({ agreementUuid: "a-1" })).toBe("/agreements/a-1");
  });

  it("carries the focused section and the origin path", () => {
    const href = buildAgreementHref({
      agreementUuid: "a-1",
      sectionUuid: "s-9",
      from: "/search?standard_id=x&year=2023",
    });
    const url = new URL(href, "https://example.test");
    expect(url.pathname).toBe("/agreements/a-1");
    expect(url.searchParams.get("focusSectionUuid")).toBe("s-9");
    expect(url.searchParams.get("from")).toBe("/search?standard_id=x&year=2023");
  });

  it("accepts a router location as the origin", () => {
    const href = buildAgreementHref({
      agreementUuid: "a-1",
      from: { pathname: "/favorites", search: "?project=p1" },
    });
    expect(new URL(href, "https://example.test").searchParams.get("from")).toBe(
      "/favorites?project=p1",
    );
  });

  it("drops unsafe origins", () => {
    for (const from of ["https://evil.test", "//evil.test/x", "javascript:alert(1)", "/agreements/b"]) {
      const href = buildAgreementHref({ agreementUuid: "a-1", from });
      expect(href).toBe("/agreements/a-1");
    }
  });
});

describe("parseAgreementOrigin", () => {
  it.each([
    ["/search?standard_id=x", "search", "Back to Search"],
    ["/search", "search", "Back to Search"],
    ["/favorites", "favorites", "Back to Favorites"],
    ["/favorites?project=p", "favorites", "Back to Favorites"],
    ["/agreement-index?page=2", "index", "Back to Index"],
    ["/compare/tax", "comparison", "Back to comparison"],
    ["/leaderboards", "other", "Back"],
  ])("labels %s as %s", (from, kind, label) => {
    expect(parseAgreementOrigin(from)).toEqual({ kind, label, href: from });
  });

  it("does not treat look-alike paths as search", () => {
    expect(parseAgreementOrigin("/searchable")?.kind).toBe("other");
  });

  it("returns null without a usable origin", () => {
    expect(parseAgreementOrigin(null)).toBeNull();
    expect(parseAgreementOrigin("")).toBeNull();
    expect(parseAgreementOrigin("https://evil.test/search")).toBeNull();
    expect(safeOriginPath("/\\evil.test")).toBeNull();
  });
});

describe("safeOriginPath", () => {
  it.each([
    ["/\t/example.com"],
    ["/\n/example.com"],
    ["/\r/example.com"],
    ["\t//example.com"],
    ["/\u0000/example.com"],
    ["/\u007F/example.com"],
    [decodeURIComponent("%2F%09%2Fexample.com")],
    ["/\\example.com"],
    ["/search\\x"],
    ["/%2F/example.com"],
    ["/%5c/example.com"],
    ["//example.com"],
    ["https://example.com/search"],
    ["javascript:alert(1)"],
    ["search"],
    ["/agreements/abc"],
    ["/agreements/abc?from=/search"],
  ])("rejects %j", (value) => {
    expect(safeOriginPath(value)).toBeNull();
  });

  it.each([
    ["/search"],
    ["/search?standard_id=a&q=Parent#top"],
    ["/favorites?project=p1"],
    ["/compare/tax"],
    ["/leaderboards"],
  ])("accepts %s", (value) => {
    expect(safeOriginPath(value)).toBe(value);
  });

  it("returns a resolved, same-origin path", () => {
    expect(safeOriginPath("  /search?q=a b  ")).toBe("/search?q=a%20b");
    expect(safeOriginPath("/search/../favorites")).toBe("/favorites");
  });
});

describe("originSearchParams", () => {
  it("returns the search query of a /search origin only", () => {
    const params = originSearchParams(parseAgreementOrigin("/search?standard_id=a&standard_id=b#top"));
    expect(params?.getAll("standard_id")).toEqual(["a", "b"]);
    expect(originSearchParams(parseAgreementOrigin("/favorites?x=1"))).toBeNull();
    expect(originSearchParams(null)).toBeNull();
  });
});
