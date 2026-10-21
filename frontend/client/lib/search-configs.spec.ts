import { describe, expect, it } from "vitest";
import {
  SEARCH_CONFIGS,
  buildDealsApiParams,
  buildSectionsApiParams,
  buildTaxClausesApiParams,
} from "./search-configs";
import { FULL_FILTERS, GOLDEN } from "./search-params-golden";
import { createEmptySearchFilters } from "./url-params";

describe("search API params per mode", () => {
  it("sections: filters, standard_id, paging, metadata fields, sort (unchanged format)", () => {
    expect(
      buildSectionsApiParams({
        filters: FULL_FILTERS,
        sortBy: "target",
        sortDirection: "asc",
      }).toString(),
    ).toBe(GOLDEN.sectionsApi);
  });

  it("tax: clauseType goes out as tax_standard_id, plus include_rep_warranty (unchanged format)", () => {
    expect(
      buildTaxClausesApiParams({
        filters: { ...FULL_FILTERS, include_rep_warranty: true },
        sortBy: "target",
        sortDirection: "asc",
      }).toString(),
    ).toBe(GOLDEN.taxApi);
  });

  it("deals: filters + sort, no metadata (unchanged format)", () => {
    expect(
      buildDealsApiParams({
        filters: FULL_FILTERS,
        sortBy: "target",
        sortDirection: "asc",
      }).toString(),
    ).toBe(GOLDEN.dealsApi);
  });

  it("omits sort when sortBy is null and ignores include_rep_warranty outside tax", () => {
    const snapshot = {
      filters: { ...createEmptySearchFilters(), include_rep_warranty: true },
      sortBy: null,
      sortDirection: "desc" as const,
    };
    expect(buildDealsApiParams(snapshot).toString()).toBe("page=1&page_size=25");
    expect(buildSectionsApiParams(snapshot).get("include_rep_warranty")).toBeNull();
    expect(buildTaxClausesApiParams(snapshot).get("include_rep_warranty")).toBe("true");
  });

  it("each config's builder round-trips through its own query string", () => {
    for (const config of Object.values(SEARCH_CONFIGS)) {
      const params = config.buildApiParams({
        filters: FULL_FILTERS,
        sortBy: "year",
        sortDirection: "desc",
      });
      expect(new URLSearchParams(params.toString()).toString()).toBe(params.toString());
    }
  });

  it("uses v1 paths as api_error endpoint labels", () => {
    for (const config of Object.values(SEARCH_CONFIGS)) {
      expect(config.telemetry.endpointLabel).toBe(config.endpoint);
    }
  });
});
