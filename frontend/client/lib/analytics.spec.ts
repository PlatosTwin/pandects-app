import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

function installBrowserGlobals() {
  const appendedScripts: Array<{ async?: boolean; src?: string }> = [];
  const fakeDocument = {
    title: "Search | Pandects",
    head: {
      appendChild: vi.fn((element: { async?: boolean; src?: string }) => {
        appendedScripts.push(element);
      }),
    },
    createElement: vi.fn(() => ({ async: false, src: "" })),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  };
  const fakeWindow: {
    dataLayer: unknown[];
    gtag?: (...args: unknown[]) => void;
    location: { href: string; origin: string; pathname: string };
    setTimeout: ReturnType<typeof vi.fn>;
    clearTimeout: ReturnType<typeof vi.fn>;
    addEventListener: ReturnType<typeof vi.fn>;
    removeEventListener: ReturnType<typeof vi.fn>;
  } = {
    dataLayer: [] as unknown[],
    location: {
      href: "https://pandects.org/search?q=tax#definition",
      origin: "https://pandects.org",
      pathname: "/search",
    },
    setTimeout: vi.fn(() => 1),
    clearTimeout: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  };

  vi.stubGlobal("document", fakeDocument);
  vi.stubGlobal("window", fakeWindow);

  return { appendedScripts, fakeDocument, fakeWindow };
}

async function importAnalytics() {
  vi.resetModules();
  return await import("./analytics");
}

describe("analytics", () => {
  beforeEach(() => {
    vi.stubEnv("PROD", true);
    vi.stubEnv("VITE_DISABLE_ANALYTICS", "");
    vi.stubEnv("VITE_GA_DEBUG_MODE", "");
    vi.stubEnv("VITE_GA_MEASUREMENT_ID", "");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("does not initialize analytics outside production", async () => {
    const { fakeWindow } = installBrowserGlobals();
    vi.stubEnv("PROD", false);
    const { bootstrapAnalytics, isAnalyticsEnabled } = await importAnalytics();

    expect(isAnalyticsEnabled()).toBe(false);
    bootstrapAnalytics();

    expect(fakeWindow.gtag).toBeUndefined();
    expect(fakeWindow.dataLayer).toEqual([]);
  });

  it("configures GA4 for manual SPA pageviews", async () => {
    const { fakeWindow } = installBrowserGlobals();
    const { bootstrapAnalytics } = await importAnalytics();

    bootstrapAnalytics();

    expect(fakeWindow.dataLayer).toEqual([
      ["js", expect.any(Date)],
      ["config", "G-94X4EVQVHZ", { send_page_view: false }],
    ]);
  });

  it("sends pageviews with canonical path and search, not the current hash", async () => {
    const { fakeWindow } = installBrowserGlobals();
    const { bootstrapAnalytics, trackPageview } = await importAnalytics();

    bootstrapAnalytics();
    trackPageview("/search?q=tax");

    expect(fakeWindow.dataLayer[fakeWindow.dataLayer.length - 1]).toEqual([
      "event",
      "page_view",
      {
        page_path: "/search?q=tax",
        page_location: "https://pandects.org/search?q=tax",
        page_title: "Search | Pandects",
      },
    ]);
  });

  it("sanitizes event parameters before queuing them", async () => {
    const { fakeWindow } = installBrowserGlobals();
    const { bootstrapAnalytics, trackEvent } = await importAnalytics();

    bootstrapAnalytics();
    trackEvent("client_error", {
      message: "x".repeat(600),
      dropped: undefined,
      bad_number: Number.NaN,
      ok: true,
    });

    expect(fakeWindow.dataLayer[fakeWindow.dataLayer.length - 1]).toEqual([
      "event",
      "client_error",
      {
        message: "x".repeat(500),
        ok: true,
      },
    ]);
  });
});
