// @vitest-environment jsdom
/**
 * Hydrates every prerendered page with the real browser app. A mismatch here
 * is what users see as a flash of the page, then a blank screen: React drops
 * the server HTML and re-renders the whole root from scratch. The prerender
 * once lacked the toaster, so every prerendered page failed this way.
 */
import { act } from "react";
import { hydrateRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PRERENDER_ROUTES } from "@shared/route-manifest.mjs";
import { App } from "@/App";
import { renderPage } from "../../prerender/entry";

// The software license page imports the repo-root LICENSE.md, which the test
// server may refuse to read from outside frontend/. It is static markdown,
// rendered identically on both sides, so a stub stands in for it.
vi.mock("@/pages/SoftwareLicense", () => ({
  default: () => <p>License text.</p>,
}));

const MISMATCH = /hydrat|did not match|server html|server rendered/i;

(
  globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;

afterEach(() => {
  act(() => root?.unmount());
  root = null;
  document.body.innerHTML = "";
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("prerendered pages", () => {
  it.each(PRERENDER_ROUTES.map((route) => route.pathname))(
    "%s hydrates without a mismatch",
    async (pathname) => {
      // No network in tests: every request fails fast, as an API outage would.
      vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("offline"));
      // jsdom has no matchMedia; a light, no-preference browser.
      window.matchMedia = (query: string) =>
        ({
          matches: false,
          media: query,
          onchange: null,
          addEventListener: () => undefined,
          removeEventListener: () => undefined,
          addListener: () => undefined,
          removeListener: () => undefined,
          dispatchEvent: () => false,
        }) as MediaQueryList;
      // Built on New Year's Eve, opened on New Year's Day: anything that
      // renders the date must not fail hydration when the two differ.
      vi.useFakeTimers({ toFake: ["Date"] });
      vi.setSystemTime(new Date(2026, 11, 31, 23, 59));
      const html = renderPage(pathname);
      vi.setSystemTime(new Date(2027, 0, 1, 0, 1));
      window.history.replaceState(null, "", pathname);
      const container = document.createElement("div");
      container.id = "root";
      container.innerHTML = html;
      document.body.appendChild(container);

      const problems: string[] = [];
      vi.spyOn(console, "error").mockImplementation((...args: unknown[]) => {
        const message = args.map(String).join(" ");
        if (MISMATCH.test(message)) problems.push(message.slice(0, 300));
      });

      await act(async () => {
        root = hydrateRoot(container, <App />, {
          onRecoverableError: (error) =>
            problems.push(String(error).slice(0, 300)),
        });
      });
      // Let the lazy page chunk resolve so its subtree hydrates too.
      for (let tick = 0; tick < 20; tick += 1) {
        await act(async () => {
          await new Promise((resolve) => setTimeout(resolve, 10));
        });
      }

      expect(problems).toEqual([]);
    },
    20_000,
  );
});
