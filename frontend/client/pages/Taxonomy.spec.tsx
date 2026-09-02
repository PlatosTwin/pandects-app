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
vi.mock("@/hooks/use-taxonomy", () => ({
  useTaxonomy: (...args: unknown[]) => useTaxonomyMock(...args),
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
      screen.getByRole("radio", { name: "Show tree view" }).getAttribute("data-state"),
    ).toBe("on");
    expect(
      screen.getByRole("radio", { name: "Show tile view" }).getAttribute("data-state"),
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
    fireEvent.click(screen.getByRole("button", { name: "Tax" }));
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
});
