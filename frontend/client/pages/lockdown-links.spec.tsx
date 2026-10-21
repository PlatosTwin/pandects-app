import { createElement, type ComponentType } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";
import { AuthContext } from "@/contexts/AuthContext";
import { isGatedPath } from "@/components/NavLinkParts";
import About from "./About";
import Landing from "./Landing";
import NotFound from "./NotFound";

type AuthStatus = "loading" | "anonymous" | "authenticated";

function render(page: ComponentType, status: AuthStatus) {
  const value = {
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
  return renderToStaticMarkup(
    createElement(
      QueryClientProvider,
      { client: new QueryClient() },
      createElement(
        AuthContext.Provider,
        { value },
        createElement(MemoryRouter, null, createElement(page)),
      ),
    ),
  );
}

/** Gated pages the markup links to, deduplicated and sorted. */
function gatedHrefs(markup: string): string[] {
  const hrefs = [...markup.matchAll(/href="([^"]+)"/g)]
    .map((match) => match[1].split(/[?#]/)[0])
    .filter((href) => href.startsWith("/") && isGatedPath(href));
  return [...new Set(hrefs)].sort();
}

// TEMPORARY (access lockdown): browse links and calls to action are offered
// only for pages the visitor can open. "loading" is how prerendered pages
// render, so it must hide them too, or they would flash for signed-out users.
describe.each([
  // Landing's member list includes Latest agreements' "all deals" link.
  [
    "Landing",
    Landing,
    ["/agreement-index", "/bulk-data", "/search", "/taxonomy"],
  ],
  ["About", About, ["/agreement-index", "/bulk-data"]],
  ["NotFound", NotFound, ["/search"]],
] as const)("%s during the lockdown", (_name, page, memberLinks) => {
  it.each(["anonymous", "loading"] as const)(
    "offers no gated page when %s",
    (status) => {
      expect(gatedHrefs(render(page, status))).toEqual([]);
    },
  );

  it("offers the gated pages to a signed-in member", () => {
    expect(gatedHrefs(render(page, "authenticated"))).toEqual([...memberLinks]);
  });
});

it("drops the empty Browse box for a signed-out visitor", () => {
  expect(render(Landing, "anonymous")).not.toContain("landing-browse-heading");
  expect(render(Landing, "authenticated")).toContain("landing-browse-heading");
});

it("matches gated route patterns, not just literal paths", () => {
  expect(isGatedPath("/agreements/abc123?from=/")).toBe(true);
  expect(isGatedPath("/search?mode=transactions")).toBe(true);
  expect(isGatedPath("/about")).toBe(false);
  expect(isGatedPath("/agreements")).toBe(false);
});
