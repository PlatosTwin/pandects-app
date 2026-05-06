import { describe, expect, it } from "vitest";
import { buildSearchStateParams } from "./url-params";

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
