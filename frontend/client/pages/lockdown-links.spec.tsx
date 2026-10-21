import { createElement, type ComponentType } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";
import { AuthContext } from "@/contexts/AuthContext";
import { ACCESS_GATED_PATHS } from "@/lib/lockdown";
import About from "./About";
import Landing from "./Landing";

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

/** Internal hrefs in the markup that a signed-out visitor could not open. */
function gatedHrefs(markup: string): string[] {
  return [...markup.matchAll(/href="([^"?#]+)/g)]
    .map((match) => match[1])
    .filter((href) => ACCESS_GATED_PATHS.has(href));
}

// TEMPORARY (access lockdown): browse links and calls to action are offered
// only for pages the visitor can open. "loading" is how prerendered pages
// render, so it must hide them too, or they would flash for signed-out users.
describe.each([
  ["Landing", Landing],
  ["About", About],
] as const)("%s during the lockdown", (_name, page) => {
  it.each(["anonymous", "loading"] as const)(
    "offers no gated page when %s",
    (status) => {
      expect(gatedHrefs(render(page, status))).toEqual([]);
    },
  );

  it("offers the gated pages to a signed-in member", () => {
    expect(gatedHrefs(render(page, "authenticated")).length).toBeGreaterThan(0);
  });
});

it("drops the empty Browse box for a signed-out visitor", () => {
  expect(render(Landing, "anonymous")).not.toContain("landing-browse-heading");
  expect(render(Landing, "authenticated")).toContain("landing-browse-heading");
});
