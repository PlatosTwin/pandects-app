import { createElement, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import NavigationDesktopMenus from "./NavigationDesktopMenus";
import NavigationMobileMenu from "./NavigationMobileMenu";

function renderWithRouter(element: ReactElement) {
  return renderToStaticMarkup(
    createElement(MemoryRouter, { initialEntries: ["/search"] }, element),
  );
}

describe("Navigation menus", () => {
  it("renders desktop docs link and omits deprecated examples links", () => {
    const desktopMarkup = renderWithRouter(createElement(NavigationDesktopMenus));

    expect(desktopMarkup).toContain("/docs/guides/getting-started");
    expect(desktopMarkup).not.toContain('href="/examples"');
    expect(desktopMarkup).not.toContain('to="/examples"');
  });

  it("does not expose deprecated examples links in the mobile menu shell", () => {
    const mobileMarkup = renderWithRouter(createElement(NavigationMobileMenu));

    expect(mobileMarkup).not.toContain('href="/examples"');
    expect(mobileMarkup).not.toContain('to="/examples"');
  });
});
