// @vitest-environment jsdom
/** The default fallback must render with or without a router above it. */
import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ErrorBoundary } from "./ErrorBoundary";

function Boom(): never {
  throw new Error("boom");
}

// React rethrows render errors to the window (jsdom prints them) and logs
// them; both are expected here, so keep the test output readable.
const swallow = (event: ErrorEvent) => event.preventDefault();

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => {});
  window.addEventListener("error", swallow);
});

afterEach(() => {
  cleanup();
  window.removeEventListener("error", swallow);
  vi.restoreAllMocks();
});

describe("ErrorBoundary default fallback", () => {
  it("renders with no router (the app-root boundary sits outside <BrowserRouter>)", () => {
    render(
      <ErrorBoundary scope="app-root">
        <Boom />
      </ErrorBoundary>,
    );

    expect(screen.getByRole("alert")).toBeTruthy();
    expect(screen.getByText("Something went wrong")).toBeTruthy();
    const home = screen.getByRole("link", { name: "Go home" });
    expect(home.getAttribute("href")).toBe("/");
    expect(screen.getByRole("button", { name: "Try again" })).toBeTruthy();
  });

  it("renders a router link inside a router", () => {
    render(
      <MemoryRouter initialEntries={["/about"]}>
        <ErrorBoundary scope="route:/about">
          <Boom />
        </ErrorBoundary>
      </MemoryRouter>,
    );

    const home = screen.getByRole("link", { name: "Go home" });
    expect(home.getAttribute("href")).toBe("/");
  });
});
