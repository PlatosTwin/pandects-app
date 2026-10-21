import { describe, expect, it } from "vitest";
import { buildFindRegex, splitFindSegments } from "./find-highlight";
import { extractSectionPlainText, sectionShortLabel } from "./section-text";
import { buildReaderMatchesQuery } from "./use-reader-data";
import { buildSectionTypeOptions } from "./ReaderContents";

describe("buildSectionTypeOptions", () => {
  const index = [
    { standard_id: ["a", "b"], standard_id_labels: { a: "Notice", b: "Notice" } },
    { standard_id: ["a", "c"], standard_id_labels: { a: "Notice", c: "Survival" } },
  ];
  const paths = {
    a: ["Termination", "Notice"],
    b: ["Indemnification", "Notice"],
    c: ["Indemnification", "Survival"],
  };

  it("adds the parent to colliding leaf labels and the path to every tooltip", () => {
    expect(buildSectionTypeOptions(index, paths)).toEqual([
      { value: "a", label: "Termination › Notice", count: 2, title: "Termination › Notice" },
      { value: "b", label: "Indemnification › Notice", count: 1, title: "Indemnification › Notice" },
      { value: "c", label: "Survival", count: 1, title: "Indemnification › Survival" },
    ]);
  });

  it("falls back to leaf labels before the taxonomy loads", () => {
    expect(buildSectionTypeOptions(index).map((option) => option.label)).toEqual([
      "Notice",
      "Notice",
      "Survival",
    ]);
  });
});
import { parseAgreementOrigin } from "@/lib/agreement-links";

describe("buildFindRegex / splitFindSegments", () => {
  it("matches case-insensitively and escapes regex syntax", () => {
    const regex = buildFindRegex("Section 1.2(a)");
    expect(splitFindSegments("see section 1.2(a) and SECTION 1.2(A).", regex)).toEqual([
      { text: "see ", hit: false },
      { text: "section 1.2(a)", hit: true },
      { text: " and ", hit: false },
      { text: "SECTION 1.2(A)", hit: true },
      { text: ".", hit: false },
    ]);
  });

  it("lets a space in the query match any whitespace run", () => {
    const regex = buildFindRegex("material  adverse");
    expect(splitFindSegments("a Material\n Adverse Effect", regex).filter((s) => s.hit)).toEqual([
      { text: "Material\n Adverse", hit: true },
    ]);
  });

  it("returns the text untouched for a blank query or no hit", () => {
    expect(buildFindRegex("   ")).toBeNull();
    expect(splitFindSegments("abc", null)).toEqual([{ text: "abc", hit: false }]);
    expect(splitFindSegments("abc", buildFindRegex("zzz"))).toEqual([{ text: "abc", hit: false }]);
  });
});

describe("extractSectionPlainText", () => {
  const xml =
    '<body><article title="ARTICLE I" uuid="a1"><section title="1.1 Defined &amp; Used" uuid="s1">' +
    "<text>First   paragraph here.</text><pageUUID>p</pageUUID><text>Second.</text>" +
    '</section><section title="1.2 Other" uuid="s2"><text>Other.</text></section></article></body>';

  it("returns title and paragraphs of one section", () => {
    expect(extractSectionPlainText(xml, "s1")).toBe(
      "1.1 Defined & Used\n\nFirst paragraph here.\n\nSecond.",
    );
    expect(extractSectionPlainText(xml, "s2")).toBe("1.2 Other\n\nOther.");
  });

  it("returns null for an unknown section", () => {
    expect(extractSectionPlainText(xml, "nope")).toBeNull();
  });
});

describe("sectionShortLabel", () => {
  it.each([
    ["Section 7.08 Indemnification", "§7.08"],
    ["7.08 Indemnification.", "§7.08"],
    ["16.2 Other Public Disclosures.", "§16.2"],
    ["§ 3.1(a) Organization", "§3.1"],
    ["Defined Terms", "Defined Terms"],
    ["", "Section"],
  ])("%s -> %s", (title, label) => {
    expect(sectionShortLabel(title)).toBe(label);
  });
});

describe("buildReaderMatchesQuery", () => {
  it("re-runs a sections search scoped to the agreement", () => {
    const query = buildReaderMatchesQuery(
      parseAgreementOrigin("/search?standard_id=x&year=2023&page=3&sort_by=target"),
      "agr-1",
    );
    expect(query?.endpoint).toBe("sections");
    expect(query?.params.getAll("standard_id")).toEqual(["x"]);
    expect(query?.params.get("year")).toBe("2023");
    expect(query?.params.get("agreement_uuid")).toBe("agr-1");
    expect(query?.params.has("page")).toBe(false);
    expect(query?.params.has("sort_by")).toBe(false);
  });

  it("maps tax-mode taxonomy ids to the tax endpoint's key", () => {
    const query = buildReaderMatchesQuery(
      parseAgreementOrigin("/search?mode=tax&standard_id=t1"),
      "agr-1",
    );
    expect(query?.endpoint).toBe("tax-clauses");
    expect(query?.params.getAll("tax_standard_id")).toEqual(["t1"]);
    expect(query?.params.has("mode")).toBe(false);
  });

  it("sends the keyword as text_query / text_match_mode", () => {
    const query = buildReaderMatchesQuery(
      parseAgreementOrigin("/search?standard_id=x&q=Parent&match=all_terms"),
      "agr-1",
    );
    expect(query?.params.get("text_query")).toBe("Parent");
    expect(query?.params.get("text_match_mode")).toBe("all_terms");
    expect(query?.params.has("q")).toBe(false);
    expect(query?.params.has("match")).toBe(false);
  });

  it("leaves the default phrase match mode out", () => {
    const query = buildReaderMatchesQuery(parseAgreementOrigin("/search?q=Parent"), "agr-1");
    expect(query?.params.get("text_query")).toBe("Parent");
    expect(query?.params.has("text_match_mode")).toBe(false);
  });

  it("treats a keyword-only search as section-level", () => {
    const query = buildReaderMatchesQuery(parseAgreementOrigin("/search?q=indemnification"), "agr-1");
    expect(query?.endpoint).toBe("sections");
    expect(query?.params.get("text_query")).toBe("indemnification");
    expect(buildReaderMatchesQuery(parseAgreementOrigin("/search?q=%20"), "agr-1")).toBeNull();
  });

  it("drops the keyword in tax mode", () => {
    const query = buildReaderMatchesQuery(
      parseAgreementOrigin("/search?mode=tax&standard_id=t1&q=Parent&match=any_terms"),
      "agr-1",
    );
    expect(query?.endpoint).toBe("tax-clauses");
    expect(query?.params.has("text_query")).toBe(false);
    expect(query?.params.has("text_match_mode")).toBe(false);
    expect(query?.params.has("q")).toBe(false);
    expect(query?.params.has("match")).toBe(false);
  });

  it("skips searches without section filters and non-search origins", () => {
    expect(buildReaderMatchesQuery(parseAgreementOrigin("/search?year=2023"), "a")).toBeNull();
    expect(buildReaderMatchesQuery(parseAgreementOrigin("/favorites"), "a")).toBeNull();
    expect(buildReaderMatchesQuery(null, "a")).toBeNull();
  });
});
