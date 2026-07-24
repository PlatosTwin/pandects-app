import { createElement, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import NavigationDesktopMenus from "./NavigationDesktopMenus";
import NavigationMobileMenu from "./NavigationMobileMenu";
import { ABOUT_LINKS, DATA_LINKS } from "./navigation-links";

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

  // Desktop and mobile menus both render from the shared navigation-links
  // module, so parity between the two is by construction. (Neither the
  // closed dropdowns nor the closed mobile sheet render their links in
  // static markup, so parity cannot be asserted from markup.) Guard the
  // shared definitions themselves instead.
  it("keeps the shared nav link definitions well-formed", () => {
    const paths = [
      ...ABOUT_LINKS.map((link) => link.to),
      ...DATA_LINKS.filter((item) => item.type === "link").map((item) => item.to),
    ];

    expect(new Set(paths).size).toBe(paths.length);
    for (const path of paths) {
      expect(path).toMatch(/^\/[a-z0-9-]+$/);
    }
  });
});
