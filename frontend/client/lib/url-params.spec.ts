import { describe, expect, it } from "vitest";
import {
  buildSearchParams,
  buildSearchStateParams,
  createEmptySearchFilters,
  parseSearchFilters,
} from "./url-params";
import { FULL_FILTERS, GOLDEN } from "./search-params-golden";

describe("buildSearchStateParams", () => {
  it("omits default search state from the public search URL", () => {
    const params = buildSearchStateParams({
      filters: {
        year: [],
        target: [],
        acquirer: [],
        clauseType: [],
        standard_id: [],
        transaction_price_total: [],
        transaction_price_stock: [],
        transaction_price_cash: [],
        transaction_price_assets: [],
        transaction_consideration: [],
        target_type: [],
        acquirer_type: [],
        target_counsel: [],
        acquirer_counsel: [],
        target_industry: [],
        acquirer_industry: [],
        deal_status: [],
        attitude: [],
        deal_type: [],
        purpose: [],
        target_pe: [],
        acquirer_pe: [],
        page: 1,
        page_size: 25,
      },
      mode: "sections",
      sortBy: "year",
      sortDirection: "desc",
    });

    expect(params.toString()).toBe("");
  });

  it("keeps non-default state shareable", () => {
    const params = buildSearchStateParams({
      filters: {
        year: ["2025"],
        target: [],
        acquirer: [],
        clauseType: [],
        standard_id: [],
        transaction_price_total: [],
        transaction_price_stock: [],
        transaction_price_cash: [],
        transaction_price_assets: [],
        transaction_consideration: [],
        target_type: [],
        acquirer_type: [],
        target_counsel: [],
        acquirer_counsel: [],
        target_industry: [],
        acquirer_industry: [],
        deal_status: [],
        attitude: [],
        deal_type: [],
        purpose: [],
        target_pe: [],
        acquirer_pe: [],
        page: 2,
        page_size: 50,
      },
      mode: "transactions",
      sortBy: "target",
      sortDirection: "asc",
    });

    expect(params.toString()).toBe(
      "year=2025&page=2&page_size=50&mode=transactions&sort_by=target&sort_direction=asc",
    );
  });
});

describe("search URL format", () => {
  const cases = [
    {
      name: "sections",
      golden: GOLDEN.urlSections,
      state: { mode: "sections", sortBy: "target", sortDirection: "asc" },
    },
    {
      name: "tax",
      golden: GOLDEN.urlTax,
      state: { mode: "tax", sortBy: "year", sortDirection: "desc", taxIncludeRepWarranty: true },
    },
    {
      name: "transactions",
      golden: GOLDEN.urlDeals,
      state: { mode: "transactions", sortBy: "acquirer", sortDirection: "desc" },
    },
  ] as const;

  it.each(cases)("$name: matches the pre-refactor URL byte for byte", ({ golden, state }) => {
    expect(buildSearchStateParams({ filters: FULL_FILTERS, ...state }).toString()).toBe(golden);
  });

  it.each(cases)("$name: a shared link parses and re-serializes to the same URL", ({ golden, state }) => {
    const parsed = parseSearchFilters(new URLSearchParams(golden));
    const rebuilt = buildSearchStateParams({
      filters: parsed,
      ...state,
      taxIncludeRepWarranty: parsed.include_rep_warranty,
    });
    expect(rebuilt.toString()).toBe(golden);
  });

  it("parses every filter field from a shared link", () => {
    const parsed = parseSearchFilters(new URLSearchParams(GOLDEN.urlTax));
    expect(parsed).toEqual({
      ...FULL_FILTERS,
      agreement_uuid: "a-uuid",
      include_rep_warranty: true,
    });
  });

  it("parses an empty URL to empty array filters and no paging", () => {
    const parsed = parseSearchFilters(new URLSearchParams(""));
    const { page: _page, page_size: _pageSize, ...emptyArrays } = createEmptySearchFilters();
    void _page;
    void _pageSize;
    expect(parsed).toEqual(emptyArrays);
  });

  it("buildSearchParams can leave out paging", () => {
    expect(buildSearchParams({ year: ["2020"], page: 2, page_size: 50 }, false).toString()).toBe(
      "year=2020",
    );
  });
});
