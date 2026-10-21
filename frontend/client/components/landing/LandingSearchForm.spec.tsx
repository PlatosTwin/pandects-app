// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { buildLandingSearchHref, LandingSearchForm } from "./LandingSearchForm";

function LocationProbe() {
  const location = useLocation();
  return (
    <p data-testid="location">{`${location.pathname}${location.search}`}</p>
  );
}

function renderForm() {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <Routes>
        <Route path="/" element={<LandingSearchForm />} />
        <Route path="/search" element={<LocationProbe />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("buildLandingSearchHref", () => {
  it("encodes the trimmed query as q and sets no mode", () => {
    expect(buildLandingSearchHref("  material adverse effect  ")).toBe(
      "/search?q=material+adverse+effect",
    );
    expect(buildLandingSearchHref("AT&T / Time Warner?")).toBe(
      "/search?q=AT%26T+%2F+Time+Warner%3F",
    );
  });

  it("goes to plain /search for an empty query", () => {
    expect(buildLandingSearchHref("")).toBe("/search");
    expect(buildLandingSearchHref("   ")).toBe("/search");
  });
});

describe("LandingSearchForm", () => {
  afterEach(() => cleanup());

  it("navigates to /search?q=<text> on submit", () => {
    renderForm();
    fireEvent.change(screen.getByRole("searchbox"), {
      target: { value: "no-shop & fiduciary out" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    const href = screen.getByTestId("location").textContent ?? "";
    expect(href).toBe("/search?q=no-shop+%26+fiduciary+out");
    expect(new URL(href, "https://x.test").searchParams.get("q")).toBe(
      "no-shop & fiduciary out",
    );
  });

  it("navigates to /search on an empty submit", () => {
    renderForm();
    fireEvent.submit(screen.getByRole("search"));
    expect(screen.getByTestId("location").textContent).toBe("/search");
  });

  it("is a plain GET form to /search with a q field before hydration", () => {
    renderForm();
    const form = screen.getByRole("search");
    expect(form.getAttribute("action")).toBe("/search");
    expect(form.getAttribute("method")).toBe("get");
    expect(screen.getByRole("searchbox").getAttribute("name")).toBe("q");
  });
});
