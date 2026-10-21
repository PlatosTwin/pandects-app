// @vitest-environment jsdom
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/components/StarButton", () => ({ StarButton: () => <span data-testid="star" /> }));
vi.mock("@/components/FlagAsInaccurateButton", () => ({
  FlagAsInaccurateButton: () => <span data-testid="flag" />,
}));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: vi.fn() }) }));

import { TooltipProvider } from "@/components/ui/tooltip";
import { SearchResultsTable, type SearchResultsTableProps } from "@/components/SearchResultsTable";
import { SearchSidebar, type SearchSidebarProps } from "@/components/SearchSidebar";
import {
  TransactionResultsList,
  type TransactionResultsListProps,
} from "@/components/TransactionResultsList";
import { createEmptySearchFilters } from "@/lib/url-params";
import type { SearchResult } from "@shared/sections";
import type { TransactionSearchResult } from "@shared/transactions";

afterEach(cleanup);

function wrap(children: ReactNode) {
  return render(
    <MemoryRouter>
      <TooltipProvider>{children}</TooltipProvider>
    </MemoryRouter>,
  );
}

describe("SearchSidebar", () => {
  function renderRail(overrides: Partial<SearchSidebarProps> = {}) {
    const props: SearchSidebarProps = {
      mode: "sections",
      filters: {
        ...createEmptySearchFilters(),
        year: ["2022", "2023"],
        deal_status: ["pending"],
        target_counsel: ["Firm A", "Firm B"],
      },
      target_counsels: [],
      acquirer_counsels: [],
      target_industries: [],
      acquirer_industries: [],
      isLoadingFilterOptions: false,
      onToggleFilterValue: vi.fn(),
      onPatchFilters: vi.fn(),
      onTextFilterChange: vi.fn(),
      variant: "sheet",
      ...overrides,
    };
    wrap(<SearchSidebar {...props} />);
    return props;
  }

  it("shows each group's active count and resets just that group", () => {
    const props = renderRail();
    const deal = screen.getByRole("button", { name: /^Deal 2 active$/ });
    expect(deal).toBeTruthy();
    expect(screen.getByRole("button", { name: /^Counsel 2 active$/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: /^Parties$/ })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Reset Counsel filters" }));
    expect(props.onPatchFilters).toHaveBeenCalledWith({ target_counsel: [], acquirer_counsel: [] });
    expect(screen.queryByRole("button", { name: "Reset Parties filters" })).toBeNull();
  });

  it("opens groups that already hold a filter and adds the tax group in Tax mode", () => {
    renderRail({ mode: "tax" });
    expect(screen.getByRole("button", { name: /^Counsel 2 active$/ }).getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByRole("button", { name: /^Parties$/ }).getAttribute("aria-expanded")).toBe("false");
    expect(screen.getByRole("checkbox", { name: /Include reps & warranties clauses/ })).toBeTruthy();
  });

  it("toggles short enum values with visible checkboxes", () => {
    const props = renderRail();
    fireEvent.click(screen.getByRole("checkbox", { name: "Complete" }));
    expect(props.onToggleFilterValue).toHaveBeenCalledWith("deal_status", "complete");
  });
});

function section(overrides: Partial<SearchResult>): SearchResult {
  return {
    id: overrides.section_uuid ?? "s",
    year: "2023",
    target: "Custom Health, Inc.",
    acquirer: "Berenson Acquisition Corp.",
    article_title: "Article VII",
    section_title: "Section 7.08 Further Action",
    standard_id: ["mae"],
    xml: "<text>(a) Upon the terms.</text>",
    section_uuid: "s",
    agreement_uuid: "A",
    verified: false,
    metadata: { transaction_price_total: 185_000_000, deal_status: "cancelled" },
    ...overrides,
  };
}

describe("SearchResultsTable (sections)", () => {
  function renderSections(results: SearchResult[]) {
    const props: SearchResultsTableProps = {
      searchResults: results,
      startPosition: 1,
      selectedResults: new Set(),
      onToggleResultSelection: vi.fn(),
      clauseTypePathByStandardId: { mae: ["Other / Unclassified", "Other / Unclassified"] },
      getAgreementHref: (r) => `/agreements/${r.agreement_uuid}?focusSectionUuid=${r.section_uuid}`,
      onOpenAgreement: vi.fn(),
      signInHref: "/account",
    };
    wrap(<SearchResultsTable {...props} />);
    return props;
  }

  it("shows the section title as the headline with deal context on one meta line", () => {
    renderSections([section({ section_uuid: "1" })]);
    expect(screen.getByRole("heading", { name: "Section 7.08 Further Action" })).toBeTruthy();
    expect(screen.getByText("$185M")).toBeTruthy();
    expect(screen.getByText("Cancelled")).toBeTruthy();
    // Repeated taxonomy segments appear once; no rank numbers.
    expect(screen.getByText("Other / Unclassified")).toBeTruthy();
    expect(screen.queryByText(/#1/)).toBeNull();
    expect(screen.getByText("(a) Upon the terms.")).toBeTruthy();
  });

  it("folds more hits from the same deal and hides empty sections behind toggles", () => {
    renderSections([
      section({ section_uuid: "1", section_title: "Section 1" }),
      section({ section_uuid: "2", section_title: "Section 2" }),
      section({ section_uuid: "3", section_title: "Section 6.11 Intentionally Omitted", xml: "" }),
    ]);
    expect(screen.queryByRole("heading", { name: "Section 2" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "+1 more section from this deal" }));
    expect(screen.getByRole("heading", { name: "Section 2" })).toBeTruthy();

    expect(screen.queryByRole("heading", { name: /Intentionally Omitted/ })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "1 empty section hidden" }));
    expect(screen.getByRole("heading", { name: /Intentionally Omitted/ })).toBeTruthy();
    expect(screen.getByText("No stored text.")).toBeTruthy();
  });

  it("selects rows with a visible checkbox and opens the reader with telemetry", () => {
    const props = renderSections([section({ section_uuid: "1" })]);
    fireEvent.click(screen.getByRole("checkbox", { name: /Select Section 7.08/ }));
    expect(props.onToggleResultSelection).toHaveBeenCalledWith("1");
    const open = screen.getByRole("link", { name: /Open Section 7.08/ });
    expect(open.getAttribute("href")).toBe("/agreements/A?focusSectionUuid=1");
    fireEvent.click(open);
    expect(props.onOpenAgreement).toHaveBeenCalledWith(expect.objectContaining({ id: "1" }), 1);
  });
});

function deal(overrides: Partial<TransactionSearchResult>): TransactionSearchResult {
  return {
    agreement_uuid: "A",
    year: 2023,
    target: "Target Co",
    acquirer: "Acquirer Co",
    filing_date: null,
    prob_filing: null,
    filing_company_name: null,
    filing_company_cik: null,
    form_type: null,
    exhibit_type: null,
    transaction_price_total: null,
    transaction_price_stock: null,
    transaction_price_cash: null,
    transaction_price_assets: null,
    transaction_consideration: "cash",
    target_type: null,
    acquirer_type: null,
    target_industry: null,
    acquirer_industry: null,
    announce_date: null,
    close_date: null,
    deal_status: "complete",
    attitude: null,
    deal_type: null,
    purpose: null,
    target_pe: null,
    acquirer_pe: null,
    url: null,
    match_count: 4,
    matched_sections: [
      {
        section_uuid: "sec-1",
        article_title: "Article VIII",
        section_title: "Section 8.3 Termination Fee",
        standard_id: [],
        snippet: "...the Reverse Termination Fee shall be paid...",
        matched_terms: ["Reverse Termination Fee"],
      },
    ],
    ...overrides,
  };
}

describe("TransactionResultsList (deals table)", () => {
  function renderDeals(overrides: Partial<TransactionResultsListProps> = {}) {
    const props: TransactionResultsListProps = {
      results: [deal({})],
      getAgreementHref: (r, sectionUuid) => `/agreements/${r.agreement_uuid}${sectionUuid ? `#${sectionUuid}` : ""}`,
      showMatches: true,
      clauseTypeLabelById: {},
      selectedResults: new Set(),
      onToggleResultSelection: vi.fn(),
      onToggleSelectAll: vi.fn(),
      sortBy: "year",
      sortDirection: "desc",
      onSort: vi.fn(),
      ...overrides,
    };
    wrap(<TransactionResultsList {...props} />);
    return props;
  }

  it("sorts from the column headers and reports the active sort", () => {
    const props = renderDeals();
    const yearHeader = screen.getByRole("columnheader", { name: /Year/ });
    expect(yearHeader.getAttribute("aria-sort")).toBe("descending");
    expect(screen.getByRole("columnheader", { name: /Target/ }).getAttribute("aria-sort")).toBe("none");
    fireEvent.click(within(screen.getByRole("columnheader", { name: /Target/ })).getByRole("button"));
    expect(props.onSort).toHaveBeenCalledWith("target");
  });

  it("uses an em dash for a missing value and expands a row to show why it matched", () => {
    renderDeals();
    const row = screen.getAllByRole("row")[1];
    expect(within(row).getAllByText("—").length).toBeGreaterThan(0);
    expect(screen.queryByText("Why this matched")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Show details for Target Co" }));
    expect(screen.getByText("Why this matched")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Section 8.3 Termination Fee" }).getAttribute("href")).toBe(
      "/agreements/A#sec-1",
    );
    // matched_terms are highlighted inside the snippet.
    expect(screen.getByText("Reverse Termination Fee").tagName).toBe("MARK");
    expect(screen.getByText("Showing 1 of 4 matching sections.")).toBeTruthy();
  });

  it("toggles a row from a plain click on the row, but not from its links", () => {
    renderDeals();
    const row = screen.getAllByRole("row")[1];
    fireEvent.click(within(row).getByText("Target Co"));
    expect(screen.getByText("Why this matched")).toBeTruthy();
    fireEvent.click(within(row).getByRole("link", { name: /Open agreement/ }));
    expect(screen.getByText("Why this matched")).toBeTruthy();
  });

  it("hides the Matches column when no clause type or keyword is applied", () => {
    renderDeals({ showMatches: false });
    expect(screen.queryByRole("columnheader", { name: "Matches" })).toBeNull();
  });
});
