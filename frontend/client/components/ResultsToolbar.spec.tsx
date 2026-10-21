// @vitest-environment jsdom
/** ARIA/DOM contract for the results-header sort and density controls. */
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
    sortSelectLabel: "Sort sections by",
    ...overrides,
  };
}

describe("ResultsToolbar", () => {
  it("exposes the sort select by name and names the direction a click switches to", () => {
    const props = baseProps({ sortDirection: "desc" });
    render(<ResultsToolbar {...props} />);

    expect(screen.getByRole("combobox", { name: "Sort sections by" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Sort ascending" }));
    expect(props.onToggleSortDirection).toHaveBeenCalledTimes(1);
  });

  it("labels the ascending state as switching to descending", () => {
    render(<ResultsToolbar {...baseProps({ sortDirection: "asc" })} />);
    expect(screen.getByRole("button", { name: "Sort descending" })).toBeTruthy();
  });

  it("reports density changes and ignores deselecting the active option", () => {
    const onDensityChange = vi.fn();
    render(<ResultsToolbar {...baseProps({ density: "comfy", onDensityChange })} />);

    fireEvent.click(screen.getByRole("radio", { name: "Compact density" }));
    expect(onDensityChange).toHaveBeenCalledWith("compact");

    onDensityChange.mockClear();
    fireEvent.click(screen.getByRole("radio", { name: "Comfy density" }));
    expect(onDensityChange).not.toHaveBeenCalled();
  });

  it("hides the density toggle when no handler is given", () => {
    render(<ResultsToolbar {...baseProps()} />);
    expect(screen.queryByRole("radio")).toBeNull();
  });
});
