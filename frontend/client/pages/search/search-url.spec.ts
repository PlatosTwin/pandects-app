import { describe, expect, it } from "vitest";
import {
  KEYWORD_SEARCH_SUPPORT,
  buildDealsApiParams,
  buildSectionsApiParams,
  buildTaxClausesApiParams,
} from "@/lib/search-configs";
import { GOLDEN } from "@/lib/search-params-golden";
import {
  buildSearchStateParams,
  createEmptySearchFilters,
  parseSearchFilters,
} from "@/lib/url-params";

describe("keyword (q) and deal paging in the search URL", () => {
  const base = { mode: "sections", sortBy: "year", sortDirection: "desc" } as const;

  it("reads ?q= (the landing page's link) as the keyword", () => {
    const parsed = parseSearchFilters(new URLSearchParams("q=reverse+termination+fee"));
    expect(parsed.text_query).toBe("reverse termination fee");
    expect(parsed.text_match_mode).toBeUndefined();
  });

  it("round-trips q and a non-default match mode, appended after every other param", () => {
    const url = "year=2023&standard_id=s1&q=liquidated+damages&match=all_terms";
    const parsed = parseSearchFilters(new URLSearchParams(url));
    expect(parsed).toMatchObject({ text_query: "liquidated damages", text_match_mode: "all_terms" });
    expect(buildSearchStateParams({ filters: parsed, ...base }).toString()).toBe(url);
  });

  it("leaves phrase (the default) and blank keywords out of the URL", () => {
    const blank = { ...createEmptySearchFilters(), text_query: "  ", text_match_mode: "phrase" as const };
    expect(buildSearchStateParams({ filters: blank, ...base }).toString()).toBe("");
    const phrase = { ...createEmptySearchFilters(), text_query: "earnout", text_match_mode: "phrase" as const };
    expect(buildSearchStateParams({ filters: phrase, ...base }).toString()).toBe("q=earnout");
    expect(parseSearchFilters(new URLSearchParams("q=+&match=any_terms")).text_query).toBeUndefined();
  });

  it("keeps golden links without a keyword unchanged", () => {
    expect(parseSearchFilters(new URLSearchParams(GOLDEN.urlSections)).text_query).toBeUndefined();
  });

  it("deep-links the deals table page, page size and column sort", () => {
    const url = buildSearchStateParams({
      filters: { ...createEmptySearchFilters(), page: 4, page_size: 50 },
      mode: "transactions",
      sortBy: "target",
      sortDirection: "asc",
    }).toString();
    expect(url).toBe("page=4&page_size=50&mode=transactions&sort_by=target&sort_direction=asc");
    const parsed = parseSearchFilters(new URLSearchParams(url));
    expect(parsed.page).toBe(4);
    expect(parsed.page_size).toBe(50);
  });

  it("writes only the mode when browsing every deal", () => {
    expect(
      buildSearchStateParams({
        filters: createEmptySearchFilters(),
        mode: "transactions",
        sortBy: "year",
        sortDirection: "desc",
      }).toString(),
    ).toBe("mode=transactions");
  });
});

describe("keyword API params per mode", () => {
  const snapshot = {
    filters: { ...createEmptySearchFilters(), text_query: " earnout ", text_match_mode: "any_terms" as const },
    sortBy: null,
    sortDirection: "desc" as const,
  };

  it("sends text_query + text_match_mode to sections and deals, after every other param", () => {
    expect(buildSectionsApiParams(snapshot).toString()).toBe(
      "page=1&page_size=25&metadata=transaction_price_total&metadata=deal_status" +
        "&metadata=deal_type&metadata=purpose&text_query=earnout&text_match_mode=any_terms",
    );
    expect(buildDealsApiParams(snapshot).toString()).toBe(
      "page=1&page_size=25&text_query=earnout&text_match_mode=any_terms",
    );
  });

  it("never sends metadata to the deals endpoint (it answers 422)", () => {
    expect(buildDealsApiParams(snapshot).has("metadata")).toBe(false);
  });

  it("leaves phrase mode implicit and sends nothing for tax clauses", () => {
    const phrase = { ...snapshot, filters: { ...snapshot.filters, text_match_mode: "phrase" as const } };
    expect(buildSectionsApiParams(phrase).has("text_match_mode")).toBe(false);
    expect(KEYWORD_SEARCH_SUPPORT.tax).toBe(false);
    expect(buildTaxClausesApiParams(snapshot).has("text_query")).toBe(false);
  });
});
