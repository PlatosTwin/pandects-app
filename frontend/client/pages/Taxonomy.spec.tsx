// @vitest-environment jsdom
/**
 * Pins the taxonomy page's default view mode and the wiring of the export
 * controls — both are single-expression decisions in the page body that no
 * component-level test would catch if they regressed.
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import type { ClauseTypeTree } from "@/lib/clause-types";

const useTaxonomyMock = vi.fn();
const useTaxonomyCountsMock = vi.fn();
vi.mock("@/hooks/use-taxonomy", () => ({
  useTaxonomy: (...args: unknown[]) => useTaxonomyMock(...args),
  useTaxonomyCounts: (...args: unknown[]) => useTaxonomyCountsMock(...args),
  useTaxonomyNodeExample: () => ({
    example: {
      agreementUuid: "agr-1",
      sectionUuid: "sec-1",
      target: "Target Co",
      acquirer: "Buyer Co",
      year: "2021",
      heading: "Section 3.2 Capitalization",
      body: "<section><text>The authorized capital stock consists of shares.</text></section>",
      bodyFormat: "xml",
    },
    isLoading: false,
    error: null,
  }),
}));

import Taxonomy from "./Taxonomy";

const tree: ClauseTypeTree = {
  Corporate: {
    id: "L1-CORP",
    children: {
      "Capital Structure": {
        id: "L2-CAP",
        children: { "Authorized Capital": { id: "L3-AUTH" } },
      },
    },
  },
};

beforeAll(() => {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    addListener: () => undefined,
    removeListener: () => undefined,
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
});

afterEach(() => {
  cleanup();
  useTaxonomyMock.mockReset();
});

function renderPage(
  state: { taxonomyTree: ClauseTypeTree | null; isLoading: boolean } = {
    taxonomyTree: tree,
    isLoading: false,
  },
) {
  useTaxonomyMock.mockReturnValue({ ...state, error: null });
  useTaxonomyCountsMock.mockReturnValue({
    counts: { "L1-CORP": 1500, "L2-CAP": 1500, "L3-AUTH": 1234 },
    isLoading: false,
    error: null,
  });
  return render(
    <MemoryRouter initialEntries={["/taxonomy"]}>
      <Taxonomy />
    </MemoryRouter>,
  );
}

describe("Taxonomy page", () => {
  it("opens in tree view rather than tile view", () => {
    renderPage();

    expect(
      screen.getByRole("radio", { name: "Tree" }).getAttribute("data-state"),
    ).toBe("on");
    expect(
      screen.getByRole("radio", { name: "Tile" }).getAttribute("data-state"),
    ).toBe("off");
    expect(screen.getByRole("heading", { name: "Taxonomy Tree View" })).toBeTruthy();
  });

  it("enables the export controls once the taxonomy has loaded", () => {
    renderPage();

    for (const name of [/copy taxonomy tree/i, /download taxonomy tree/i]) {
      expect((screen.getByRole("button", { name }) as HTMLButtonElement).disabled).toBe(
        false,
      );
    }
  });

  it("clears a pending 'Copied' flash when switching to the other taxonomy", async () => {
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
      configurable: true,
      writable: true,
    });

    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /copy taxonomy tree/i }));
    await screen.findByText("Copied");

    // Switching tabs must not leave the previous taxonomy's success state
    // sitting over the one now being fetched.
    fireEvent.click(screen.getByRole("radio", { name: "Tax" }));
    await waitFor(() => expect(screen.queryByText("Copied")).toBeNull());
  });

  it("disables the export controls while the taxonomy is loading or empty", () => {
    renderPage({ taxonomyTree: null, isLoading: true });

    for (const name of [/copy taxonomy tree/i, /download taxonomy tree/i]) {
      expect((screen.getByRole("button", { name }) as HTMLButtonElement).disabled).toBe(
        true,
      );
    }

    cleanup();
    useTaxonomyMock.mockReset();
    renderPage({ taxonomyTree: {}, isLoading: false });

    for (const name of [/copy taxonomy tree/i, /download taxonomy tree/i]) {
      expect((screen.getByRole("button", { name }) as HTMLButtonElement).disabled).toBe(
        true,
      );
    }
  });

  it("links each type to search and shows its corpus count", async () => {
    renderPage();

    const link = await screen.findByRole("link", { name: "Find Authorized Capital in search" });
    expect(link.getAttribute("href")).toBe("/search?standard_id=L3-AUTH");
    expect(screen.getByLabelText("1,234 sections")).toBeTruthy();
  });

  it("opens the node panel with path, count and an example", async () => {
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: /Authorized Capital/ }));

    const panel = await screen.findByRole("heading", { name: "Authorized Capital", level: 2 });
    expect(panel).toBeTruthy();
    const path = screen.getByRole("navigation", { name: "Parent path" });
    expect(path.textContent).toContain("Corporate");
    expect(path.textContent).toContain("Capital Structure");
    expect(screen.getByText("1,234 sections")).toBeTruthy();
    expect(screen.getByText("The authorized capital stock consists of shares.")).toBeTruthy();
    expect(
      screen.getByRole("link", { name: "Open in agreement" }).getAttribute("href"),
    ).toBe("/agreements/agr-1?from=%2Ftaxonomy%3Fnode%3DL3-AUTH&focusSectionUuid=sec-1");
  });
});
