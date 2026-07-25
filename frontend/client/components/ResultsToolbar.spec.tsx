// @vitest-environment jsdom
/**
 * ARIA/DOM contract for the shared results toolbar. Pins the accessible
 * names, live-region behavior, and callback wiring that the three result
 * lists (sections, deals, tax clauses) relied on before extraction.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ResultsToolbar, type ResultsToolbarProps } from "./ResultsToolbar";

afterEach(cleanup);

function baseProps(overrides: Partial<ResultsToolbarProps> = {}): ResultsToolbarProps {
  return {
    sortBy: "year",
    sortDirection: "desc",
    onSortResults: vi.fn(),
    onToggleSortDirection: vi.fn(),
    sortSelectLabel: "Sort section results by",
    ...overrides,
  };
}

function selection(overrides: Partial<NonNullable<ResultsToolbarProps["selection"]>> = {}) {
  return {
    allSelected: false,
    someSelected: false,
    onToggleSelectAll: vi.fn(),
    selectAllLabel: "Select all results",
    countLabel: "Select all",
    ...overrides,
  };
}

describe("ResultsToolbar (labeled variant)", () => {
  it("renders select-all with accessible name and a polite live-region count", () => {
    const sel = selection({ countLabel: "3 of 25 selected" });
    const { container } = render(<ResultsToolbar {...baseProps({ selection: sel })} />);

    const checkbox = screen.getByRole("checkbox", { name: "Select all results" });
    fireEvent.click(checkbox);
    expect(sel.onToggleSelectAll).toHaveBeenCalledTimes(1);

    const count = container.querySelector('[aria-live="polite"]');
    expect(count?.textContent).toBe("3 of 25 selected");
  });

  it("omits selection controls when no selection is provided", () => {
    render(<ResultsToolbar {...baseProps()} />);
    expect(screen.queryByRole("checkbox")).toBeNull();
  });

  it("exposes the sort select accessible name and toggles direction", () => {
    const props = baseProps({ sortDirection: "desc" });
    render(<ResultsToolbar {...props} />);

    expect(
      screen.getByRole("combobox", { name: "Sort section results by" }),
    ).toBeTruthy();

    // Direction label always describes the direction a click switches to.
    const directionButton = screen.getByRole("button", { name: "Sort ascending" });
    fireEvent.click(directionButton);
    expect(props.onToggleSortDirection).toHaveBeenCalledTimes(1);
  });

  it("reports density changes and guards against deselection", () => {
    const onDensityChange = vi.fn();
    render(<ResultsToolbar {...baseProps({ density: "comfy", onDensityChange })} />);

    fireEvent.click(screen.getByRole("radio", { name: "Compact density" }));
    expect(onDensityChange).toHaveBeenCalledWith("compact");

    // Clicking the active item emits an empty value, which must be ignored.
    onDensityChange.mockClear();
    fireEvent.click(screen.getByRole("radio", { name: "Comfy density" }));
    expect(onDensityChange).not.toHaveBeenCalled();
  });
});

describe("ResultsToolbar (inline variant)", () => {
  it("renders mobile-visible selection without a live region", () => {
    const sel = selection({
      selectAllLabel: "Select all tax clauses on this page",
      countLabel: "2 selected",
    });
    const { container } = render(
      <ResultsToolbar {...baseProps({ variant: "inline", selection: sel })} />,
    );

    const checkbox = screen.getByRole("checkbox", {
      name: "Select all tax clauses on this page",
    });
    fireEvent.click(checkbox);
    expect(sel.onToggleSelectAll).toHaveBeenCalledTimes(1);
    expect(container.querySelector('[aria-live="polite"]')).toBeNull();
    expect(screen.getByText("2 selected")).toBeTruthy();
  });

  it("uses the tax direction-button wording and hides density without a handler", () => {
    const props = baseProps({
      variant: "inline",
      sortDirection: "asc",
      sortSelectLabel: "Sort tax clause results by",
    });
    render(<ResultsToolbar {...props} />);

    expect(
      screen.getByRole("combobox", { name: "Sort tax clause results by" }),
    ).toBeTruthy();

    const directionButton = screen.getByRole("button", {
      name: "Change sort direction. Current direction: ascending",
    });
    fireEvent.click(directionButton);
    expect(props.onToggleSortDirection).toHaveBeenCalledTimes(1);

    expect(screen.queryByRole("radio", { name: "Compact density" })).toBeNull();
  });

  it("renders the density toggle when a handler is provided", () => {
    const onDensityChange = vi.fn();
    render(
      <ResultsToolbar
        {...baseProps({ variant: "inline", density: "comfy", onDensityChange })}
      />,
    );

    fireEvent.click(screen.getByRole("radio", { name: "Compact density" }));
    expect(onDensityChange).toHaveBeenCalledWith("compact");
  });
});
