// @vitest-environment jsdom
import { useRef, useState } from "react";
import { MemoryRouter } from "react-router-dom";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { XMLRenderer } from "@/components/XMLRenderer";
import { FindBar } from "./FindBar";
import { useFindInDocument } from "./use-find-in-document";
import { ReaderBackLink } from "./ReaderBackLink";
import { parseAgreementOrigin } from "@/lib/agreement-links";

afterEach(cleanup);

const XML =
  '<document><body><article title="ARTICLE I Defined Terms" uuid="a1">' +
  '<section title="1.1 Indemnified Party" uuid="s1"><text>The Indemnified Party shall notify.</text></section>' +
  '<section title="1.2 Other" uuid="s2"><text>Nothing to see here.</text></section>' +
  '<section title="1.3 Claims" uuid="s3"><text>Each indemnified party may claim.</text></section>' +
  "</article></body></document>";

function Harness({ onClose = () => {} }: { onClose?: () => void }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState("");
  const find = useFindInDocument({ containerRef, query, topOffset: 0, debounceMs: 0 });
  return (
    <>
      <FindBar query={query} onQueryChange={setQuery} onClose={onClose} find={find} />
      <div ref={containerRef}>
        <XMLRenderer xmlContent={XML} mode="agreement" findQuery={find.appliedQuery} />
      </div>
    </>
  );
}

function currentHitText(container: HTMLElement) {
  return container.querySelector("mark[data-find-current]")?.textContent ?? null;
}

describe("find in agreement", () => {
  it("wraps every hit in <mark>, counts them, and steps with Enter / Shift+Enter", async () => {
    window.scrollTo = vi.fn();
    const { container } = render(<Harness />);
    const input = screen.getByRole("searchbox", { name: "Find in agreement" });

    fireEvent.change(input, { target: { value: "indemnified party" } });

    await waitFor(() => expect(screen.getByText("1 of 3")).toBeTruthy());
    const marks = container.querySelectorAll("mark[data-find-hit]");
    // Heading of 1.1, its body, and the lower-case hit in 1.3.
    expect(Array.from(marks).map((mark) => mark.textContent)).toEqual([
      "Indemnified Party",
      "Indemnified Party",
      "indemnified party",
    ]);
    expect(container.querySelectorAll("mark[data-find-current]")).toHaveLength(1);
    expect(marks[0].hasAttribute("data-find-current")).toBe(true);

    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(screen.getByText("2 of 3")).toBeTruthy());
    expect(marks[1].hasAttribute("data-find-current")).toBe(true);
    expect(marks[0].hasAttribute("data-find-current")).toBe(false);

    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.keyDown(input, { key: "Enter" });
    // Wraps around past the last hit.
    await waitFor(() => expect(screen.getByText("1 of 3")).toBeTruthy());

    fireEvent.keyDown(input, { key: "Enter", shiftKey: true });
    await waitFor(() => expect(screen.getByText("3 of 3")).toBeTruthy());
    expect(currentHitText(container)).toBe("indemnified party");
  });

  it("shows no stale result while a new query is applying, then 'No results'", async () => {
    window.scrollTo = vi.fn();
    render(<Harness />);
    const input = screen.getByRole("searchbox", { name: "Find in agreement" });
    fireEvent.change(input, { target: { value: "claims" } });
    await waitFor(() => expect(screen.getByText("1 of 1")).toBeTruthy());

    act(() => {
      fireEvent.change(input, { target: { value: "zzz-not-here" } });
    });
    // Same tick as the keystroke: the old "1 of 1" must not linger.
    expect(screen.queryByText("1 of 1")).toBeNull();
    expect(screen.queryByText("No results")).toBeNull();
    await waitFor(() => expect(screen.getByText("No results")).toBeTruthy());
  });

  it("closes on Escape and clears highlights when the query is cleared", async () => {
    window.scrollTo = vi.fn();
    const onClose = vi.fn();
    const { container } = render(<Harness onClose={onClose} />);
    const input = screen.getByRole("searchbox", { name: "Find in agreement" });
    fireEvent.change(input, { target: { value: "other" } });
    await waitFor(() => expect(container.querySelectorAll("mark[data-find-hit]")).toHaveLength(1));

    fireEvent.change(input, { target: { value: "" } });
    await waitFor(() => expect(container.querySelectorAll("mark[data-find-hit]")).toHaveLength(0));

    fireEvent.keyDown(input, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("ReaderBackLink", () => {
  const renderLink = (from: string | null, compact = false) =>
    render(
      <MemoryRouter>
        <ReaderBackLink origin={parseAgreementOrigin(from)} compact={compact} />
      </MemoryRouter>,
    );

  it.each([
    ["/search?standard_id=x", "Back to Search"],
    ["/favorites", "Back to Favorites"],
    ["/agreement-index", "Back to Index"],
    ["/compare/tax", "Back to comparison"],
  ])("labels the back link for %s", (from, label) => {
    renderLink(from);
    const link = screen.getByRole("link", { name: label });
    expect(link.getAttribute("href")).toBe(from);
  });

  it("keeps the label as the accessible name when compact", () => {
    renderLink("/favorites", true);
    expect(screen.getByRole("link", { name: "Back to Favorites" })).toBeTruthy();
  });

  it("falls back to a Search breadcrumb without an origin", () => {
    renderLink(null);
    expect(screen.getByRole("navigation", { name: "Breadcrumb" })).toBeTruthy();
    const link = screen.getByRole("link", { name: "Search" });
    expect(link.getAttribute("href")).toBe("/search");
    expect(screen.queryByText(/Back to/)).toBeNull();
  });
});
