// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest";
import type { SearchResult } from "@shared/sections";
import { createEmptySearchFilters } from "@/lib/url-params";
import { keywordHighlightPattern, literalTermsPattern } from "./Highlight";
import { validateKeywordQuery } from "./keyword-query";
import { previewStart } from "./ResultBits";
import {
  FACET_GROUP_FIELDS,
  RECENT_SEARCHES_KEY,
  bucketsBetween,
  cleanSnippet,
  collapseTaxonomyPath,
  countActiveFilters,
  countRailFilters,
  describeSearch,
  filtersForModeSwitch,
  findClauseTypeIds,
  formatBucketRange,
  groupSectionResults,
  hasAnySearchFilter,
  loadRecentSearches,
  resetFieldsPatch,
  saveRecentSearch,
  sectionXmlToPlainText,
  summarizeYearRange,
  yearsBetween,
} from "./search-ui";

function section(overrides: Partial<SearchResult>): SearchResult {
  return {
    id: overrides.section_uuid ?? "s",
    year: "2023",
    target: "Target Co",
    acquirer: "Acquirer Co",
    article_title: "Article VII",
    section_title: "Section 7.1 Efforts",
    standard_id: [],
    xml: "<text>Body text.</text>",
    section_uuid: "s",
    agreement_uuid: "a",
    verified: false,
    ...overrides,
  };
}

describe("facet group counts and reset", () => {
  const filters = {
    ...createEmptySearchFilters(),
    year: ["2021", "2022", "2023"],
    transaction_price_total: ["100M - 250M", "250M - 500M"],
    deal_status: ["pending", "complete"],
    target: ["Acme"],
    target_counsel: ["Firm A"],
    agreement_uuid: "abc",
    clauseType: ["x"],
  };

  it("counts a range once and list values one by one", () => {
    // year range (1) + value range (1) + two statuses (2)
    expect(countActiveFilters(filters, FACET_GROUP_FIELDS.deal)).toBe(4);
    expect(countActiveFilters(filters, FACET_GROUP_FIELDS.parties)).toBe(1);
    expect(countActiveFilters(filters, FACET_GROUP_FIELDS.counsel)).toBe(1);
    expect(countActiveFilters(filters, FACET_GROUP_FIELDS.advanced)).toBe(1);
  });

  it("totals the rail (the query-bar clause type is not a rail filter)", () => {
    expect(countRailFilters(filters, "sections")).toBe(7);
    expect(countRailFilters({ ...filters, include_rep_warranty: true }, "tax")).toBe(8);
    expect(countRailFilters({ ...filters, include_rep_warranty: true }, "sections")).toBe(7);
  });

  it("resets only the group's own fields", () => {
    const next = { ...filters, ...resetFieldsPatch(FACET_GROUP_FIELDS.deal) };
    expect(countActiveFilters(next, FACET_GROUP_FIELDS.deal)).toBe(0);
    expect(next.target).toEqual(["Acme"]);
    expect(next.agreement_uuid).toBe("abc");
    const advanced = { ...filters, ...resetFieldsPatch(FACET_GROUP_FIELDS.advanced) };
    expect(advanced.agreement_uuid).toBeUndefined();
    expect(advanced.deal_status).toEqual(["pending", "complete"]);
  });

  it("knows when anything narrows the search", () => {
    expect(hasAnySearchFilter(createEmptySearchFilters(), "sections")).toBe(false);
    expect(hasAnySearchFilter({ ...createEmptySearchFilters(), text_query: "fee" }, "sections")).toBe(true);
    expect(hasAnySearchFilter({ ...createEmptySearchFilters(), clauseType: ["x"] }, "tax")).toBe(true);
  });

  it("drops the clause type only when crossing into or out of Tax", () => {
    expect(filtersForModeSwitch(filters, "sections", "transactions").clauseType).toEqual(["x"]);
    expect(filtersForModeSwitch(filters, "sections", "tax").clauseType).toEqual([]);
    expect(filtersForModeSwitch(filters, "tax", "transactions").clauseType).toEqual([]);
  });
});

describe("ranges", () => {
  it("expands and summarizes year ranges", () => {
    expect(yearsBetween("2023", "2021")).toEqual(["2023", "2022", "2021"]);
    expect(summarizeYearRange(["2021", "2023", "2022"])).toEqual({ from: "2021", to: "2023", contiguous: true });
    expect(summarizeYearRange(["2019", "2021"]).contiguous).toBe(false);
  });

  it("maps a value range onto the API's buckets", () => {
    expect(bucketsBetween("100M - 250M", "500M - 750M")).toEqual(["100M - 250M", "250M - 500M", "500M - 750M"]);
    expect(formatBucketRange("100M - 250M", "750M - 1B")).toBe("$100M – $1B");
    expect(formatBucketRange("5B - 10B", "20B+")).toBe("$5B+");
  });
});

describe("section text and grouping", () => {
  it("renders section XML as plain paragraphs without page markers", () => {
    const xml =
      "<text>(a) First &amp; part.</text><page>87</page><pageUUID>ba25-uuid</pageUUID><text>(b) Second.</text>";
    expect(sectionXmlToPlainText(xml)).toBe("(a) First & part.\n(b) Second.");
  });

  it("collapses repeated taxonomy segments", () => {
    expect(collapseTaxonomyPath(["Other / Unclassified", "Other / Unclassified", "Other / Unclassified"])).toEqual([
      "Other / Unclassified",
    ]);
    expect(collapseTaxonomyPath(["Termination", "Termination Fee"])).toEqual(["Termination", "Termination Fee"]);
  });

  it("groups consecutive hits from one deal and hides empty sections", () => {
    const groups = groupSectionResults(
      [
        section({ section_uuid: "1", agreement_uuid: "A" }),
        section({ section_uuid: "2", agreement_uuid: "A" }),
        section({ section_uuid: "3", agreement_uuid: "A", xml: "" }),
        section({ section_uuid: "4", agreement_uuid: "B", xml: "<text>[Intentionally Omitted]</text>" }),
        section({ section_uuid: "5", agreement_uuid: "A" }),
      ],
      11,
    );
    expect(groups.map((g) => g.agreementUuid)).toEqual(["A", "B", "A"]);
    expect(groups[0].lead?.result.section_uuid).toBe("1");
    expect(groups[0].lead?.position).toBe(11);
    expect(groups[0].more.map((i) => i.result.section_uuid)).toEqual(["2"]);
    expect(groups[0].empty.map((i) => i.result.section_uuid)).toEqual(["3"]);
    // A run of only empty sections has no lead card.
    expect(groups[1].lead).toBeNull();
    expect(groups[1].empty).toHaveLength(1);
    expect(groups[2].lead?.position).toBe(15);
  });

  it("treats redacted sections as gated, not empty", () => {
    const [group] = groupSectionResults([
      section({ xml: '<text redacted="true">[REDACTED]</text>' }),
    ]);
    expect(group.lead?.redacted).toBe(true);
    expect(group.empty).toHaveLength(0);
  });

  it("removes page markers that leak into API snippets", () => {
    expect(cleanSnippet("...failure. 77 908fdb16-cc24-11f1-9296-be44a5e954c7 (c) Reverse Fee...")).toBe(
      "...failure. (c) Reverse Fee...",
    );
  });
});

describe("keyword helpers", () => {
  it("mirrors the API's keyword rules", () => {
    expect(validateKeywordQuery("reverse termination fee", "phrase")).toBeNull();
    expect(validateKeywordQuery("the", "phrase")).toMatch(/short, common words/);
    expect(validateKeywordQuery("fee of the", "all_terms")).toMatch(/Exact phrase/);
    expect(validateKeywordQuery("in*", "phrase")).toMatch(/3 letters/);
    expect(validateKeywordQuery("a*b", "phrase")).toMatch(/end of a word/);
    expect(validateKeywordQuery("indemnif*", "any_terms")).toBeNull();
    expect(validateKeywordQuery("one two three four five six seven eight nine", "any_terms")).toMatch(/at most 8/);
  });

  it("highlights a phrase across punctuation and word prefixes", () => {
    const phrase = keywordHighlightPattern("material adverse effect", "phrase");
    expect("a Material Adverse-Effect clause".match(phrase!)?.[0]).toBe("Material Adverse-Effect");
    const prefix = keywordHighlightPattern("indemnif*", "any_terms");
    expect("the Indemnification provisions".match(prefix!)?.[0]).toBe("Indemnification");
    expect(literalTermsPattern([])).toBeNull();
  });

  it("starts a collapsed preview at the first match when it sits deep in the text", () => {
    const text = `${"Intro words. ".repeat(40)}\n(c) Reverse Termination Fee applies.`;
    const start = previewStart(text, keywordHighlightPattern("reverse termination fee"));
    expect(text.slice(start)).toBe("(c) Reverse Termination Fee applies.");
    expect(previewStart("Reverse termination fee up top.", keywordHighlightPattern("reverse termination fee"))).toBe(0);
  });
});

describe("taxonomy lookups", () => {
  const tree = {
    Termination: {
      id: "t",
      children: {
        "Termination Economics": {
          id: "te",
          children: { "Termination Fee / Reverse Termination Fee": { id: "fee" } },
        },
      },
    },
    "Legacy group": { "Specific Performance": "sp" },
  };

  it("finds clause types by label, leaves first, in every tree shape", () => {
    expect(findClauseTypeIds(tree, /termination fee/i)).toEqual(["fee"]);
    expect(findClauseTypeIds(tree, /specific performance/i)).toEqual(["sp"]);
    expect(findClauseTypeIds(tree, /^termination/i)).toEqual(["fee", "t", "te"]);
  });

  it("labels a search without raw ids", () => {
    const label = describeSearch(
      "sections",
      { ...createEmptySearchFilters(), text_query: "fee", clauseType: ["fee", "zzz"], deal_status: ["pending"] },
      { fee: "Termination Fee" },
    );
    expect(label).toBe("Sections · “fee” · Termination Fee +1 · 1 filter");
  });
});

describe("recent searches", () => {
  afterEach(() => window.localStorage.clear());

  it("keeps the newest five, deduplicated, and survives bad storage", () => {
    for (let i = 0; i < 7; i += 1) {
      saveRecentSearch({ search: `q=${i}`, label: `#${i}`, mode: "sections" });
    }
    saveRecentSearch({ search: "q=3", label: "#3 again", mode: "sections" });
    const list = loadRecentSearches();
    expect(list.map((entry) => entry.search)).toEqual(["q=3", "q=6", "q=5", "q=4", "q=2"]);
    window.localStorage.setItem(RECENT_SEARCHES_KEY, "{not json");
    expect(loadRecentSearches()).toEqual([]);
  });

  it("replaces the newest entry while a search is being refined", () => {
    saveRecentSearch({ search: "a", label: "A", mode: "sections" });
    saveRecentSearch({ search: "b", label: "B", mode: "sections" });
    saveRecentSearch({ search: "b2", label: "B2", mode: "sections" }, { replaceNewest: true });
    expect(loadRecentSearches().map((entry) => entry.search)).toEqual(["b2", "a"]);
  });
});
