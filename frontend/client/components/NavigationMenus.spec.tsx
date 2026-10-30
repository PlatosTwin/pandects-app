import { createElement, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AuthContext } from "@/contexts/AuthContext";
import NavigationDesktopMenus from "./NavigationDesktopMenus";
import NavigationMobileMenu from "./NavigationMobileMenu";
import Footer from "./Footer";
import { AuthMenu } from "./AuthMenu";
import {
  DOCS_LINK,
  FOOTER_COLUMNS,
  LEGAL_LINKS,
  LOCKDOWN_ACTIVE,
  NAV_GROUPS,
  PROJECT_LINKS,
  SEARCH_LINK,
  isGatedLink,
  isGroupActive,
} from "./navigation-links";

type AuthStatus = "loading" | "anonymous" | "authenticated";

function authValue(status: AuthStatus) {
  return {
    status,
    user:
      status === "authenticated"
        ? { id: "u1", email: "member@example.com" }
        : null,
    session_token: null,
    wakePending: false,
    refresh: async () => undefined,
    logout: () => undefined,
  };
}

function render(
  element: ReactElement,
  { path = "/search", auth }: { path?: string; auth?: AuthStatus } = {},
) {
  const routed = createElement(MemoryRouter, { initialEntries: [path] }, element);
  return renderToStaticMarkup(
    auth
      ? createElement(AuthContext.Provider, { value: authValue(auth) }, routed)
      : routed,
  );
}

// Pull the opening tag of the top-level trigger for a dropdown group.
function triggerTag(markup: string, label: string): string {
  const match = markup.match(
    new RegExp(`<button[^>]*>${label}<svg`),
  );
  if (!match) throw new Error(`no trigger for ${label}`);
  return match[0];
}

describe("Navigation menus", () => {
  it("renders Search, the three groups and an external Docs link on desktop", () => {
    const markup = render(createElement(NavigationDesktopMenus));

    for (const group of NAV_GROUPS) {
      expect(markup).toContain(`>${group.label}<svg`);
    }
    expect(markup).not.toContain(">Project<");
    expect(markup).toContain("/docs/guides/getting-started");
    expect(markup).toMatch(
      /href="[^"]*\/docs\/guides\/getting-started" target="_blank" rel="noopener noreferrer"/,
    );
    expect(markup).toContain("(opens in a new tab)");
    expect(markup).not.toContain('href="/examples"');
  });

  it("marks the current top-level link with aria-current and the bottom indicator", () => {
    const markup = render(createElement(NavigationDesktopMenus), { path: "/search" });
    const searchTag = markup.match(/<a[^>]*href="\/search"[^>]*>/)?.[0] ?? "";

    expect(searchTag).toContain('aria-current="page"');
    expect(searchTag).toContain("after:bg-primary");
    expect(searchTag).not.toContain("border-l-2");
  });

  it.each(NAV_GROUPS.map((group) => [group.label, group] as const))(
    "highlights only the %s trigger on one of its pages",
    (_label, group) => {
      const first = group.links.find((link) => link.kind === "internal");
      if (!first || first.kind !== "internal") throw new Error("no internal link");
      const markup = render(createElement(NavigationDesktopMenus), { path: first.to });

      for (const other of NAV_GROUPS) {
        const tag = triggerTag(markup, other.label);
        if (other.id === group.id) {
          expect(tag).toContain('data-active="true"');
          expect(tag).toContain("after:bg-primary");
        } else {
          expect(tag).not.toContain('data-active="true"');
        }
      }
    },
  );

  it("shows a lock glyph on gated links for signed-out visitors only", () => {
    expect(LOCKDOWN_ACTIVE).toBe(true);
    expect(isGatedLink(SEARCH_LINK)).toBe(true);
    expect(isGatedLink(DOCS_LINK)).toBe(false);

    const anonymous = render(createElement(NavigationDesktopMenus), { auth: "anonymous" });
    const member = render(createElement(NavigationDesktopMenus), { auth: "authenticated" });
    const loading = render(createElement(NavigationDesktopMenus), { auth: "loading" });

    expect(anonymous).toContain('data-testid="nav-lock"');
    expect(anonymous).toContain("(sign-in required)");
    expect(member).not.toContain('data-testid="nav-lock"');
    expect(loading).not.toContain('data-testid="nav-lock"');
  });

  it("renders only the avatar initial when signed in (Favorites lives in its menu)", () => {
    const markup = render(createElement(AuthMenu), { auth: "authenticated" });

    expect(markup).not.toContain('href="/favorites"');
    expect(markup).toContain('aria-label="Account menu for member@example.com"');
    expect(markup).toContain(">M</span>");
    expect(markup).not.toContain("member@e...");
  });

  it("renders only a sign-in link when signed out", () => {
    const markup = render(createElement(AuthMenu), { auth: "anonymous" });

    expect(markup).toContain(">Sign in<");
    expect(markup).not.toContain('href="/favorites"');
  });

  it("keeps the mobile bar to the menu button", () => {
    const markup = render(createElement(NavigationMobileMenu));

    expect(markup).toContain('aria-label="Open menu"');
    expect(markup).not.toContain("Theme:");
    expect(markup).not.toContain(">Sign in<");
  });

  it("renders the theme toggle in the desktop bar", () => {
    const markup = render(createElement(NavigationDesktopMenus));
    expect(markup).toContain('aria-label="Theme: System"');
  });

  it("renders footer columns, legal and license links from the shared lists", () => {
    const markup = render(createElement(Footer));

    for (const column of FOOTER_COLUMNS) {
      expect(markup).toContain(`>${column.heading}</h2>`);
    }
    for (const link of LEGAL_LINKS) {
      expect(markup).toContain(`href="${link.to}"`);
    }
    expect(markup).toContain('href="/license/data"');
    expect(markup).toContain('href="/license/software"');
    expect(markup).toContain("/docs/guides/changelog");
  });

  it("keeps the shared nav link definitions well-formed", () => {
    const internal = [
      SEARCH_LINK,
      ...NAV_GROUPS.flatMap((group) => group.links),
      ...PROJECT_LINKS,
    ].filter((link) => link.kind === "internal");
    const paths = internal.map((link) => (link.kind === "internal" ? link.to : ""));

    expect(new Set(paths).size).toBe(paths.length);
    for (const path of paths) {
      expect(path).toMatch(/^\/[a-z0-9-]+$/);
    }
    for (const group of NAV_GROUPS) {
      expect(isGroupActive(group, "/nowhere")).toBe(false);
    }
  });
});
