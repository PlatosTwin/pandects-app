// @vitest-environment jsdom
import { createElement, type ComponentProps, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth-fetch", () => ({
  authFetch: vi.fn(async () => ({
    ok: true,
    status: 200,
    statusText: "OK",
    json: async () => ({ sectors: [] }),
  })),
}));
vi.mock("@/lib/analytics", () => ({
  trackEvent: vi.fn(),
}));

import { TooltipProvider } from "@/components/ui/tooltip";
import type { SearchFilters } from "@shared/sections";
import type { SearchMode } from "@shared/search";
import { SearchActionsBar } from "./SearchActionsBar";

afterEach(cleanup);

const EMPTY_FILTERS: SearchFilters = {
  year: [],
  target: [],
  acquirer: [],
  clauseType: [],
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
  agreement_uuid: "",
  section_uuid: "",
};

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { refetchOnWindowFocus: false, retry: false },
    },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(
      QueryClientProvider,
      { client: queryClient },
      createElement(TooltipProvider, null, children),
    );
  };
}

function renderBar(overrides: Partial<ComponentProps<typeof SearchActionsBar>> = {}) {
  const props: ComponentProps<typeof SearchActionsBar> = {
    searchMode: "sections",
    isSearching: false,
    selectedSize: 0,
    resultsLength: 0,
    onSearch: vi.fn(),
    onDownloadCSV: vi.fn(),
    onClearFilters: vi.fn(),
    taxIncludeRepWarranty: false,
    onTaxIncludeRepWarrantyChange: vi.fn(),
    taxSelectedCount: 0,
    onTaxCompare: vi.fn(),
    filters: EMPTY_FILTERS,
    clauseTypeLabelById: {},
    onToggleFilterValue: vi.fn(),
    onTextFilterChange: vi.fn(),
    ...overrides,
  };

  return {
    props,
    ...render(<SearchActionsBar {...props} />, { wrapper: createWrapper() }),
  };
}

describe("SearchActionsBar", () => {
  it("describes the disabled download button with a screen-reader hint", () => {
    renderBar();

    const button = screen.getByRole("button", {
      name: "Download CSV (disabled: no results to download. Run a search first.)",
    });
    const hint = screen.getByText("No results to download. Run a search first.");
    expect(button).toHaveProperty("disabled", true);
    expect(button.getAttribute("aria-describedby")).toBe(hint.id);
    expect(hint.id).toMatch(/search-download-hint$/);
  });

  it("keeps download enabled and includes the selected count when rows are selected", () => {
    const { props } = renderBar({ selectedSize: 3 });

    const button = screen.getByRole("button", { name: "Download CSV" });
    expect(button).toHaveProperty("disabled", false);
    expect(button.getAttribute("aria-describedby")).toBeNull();
    expect(screen.getByText(/3/)).toBeTruthy();

    fireEvent.click(button);
    expect(props.onDownloadCSV).toHaveBeenCalledTimes(1);
  });

  it("shows tax compare guidance until enough clauses are selected", () => {
    const mode: SearchMode = "tax";
    renderBar({ searchMode: mode, taxSelectedCount: 1 });

    expect(screen.getByRole("button", { name: "Compare (select 1 more)" })).toHaveProperty(
      "disabled",
      true,
    );
  });
});
