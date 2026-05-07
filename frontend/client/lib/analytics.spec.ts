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

  it("redacts sensitive account-control routes from pageviews", async () => {
    const { fakeWindow } = installBrowserGlobals();
    const { bootstrapAnalytics, trackPageview } = await importAnalytics();

    bootstrapAnalytics();
    trackPageview("/verify-email?user_id=user-1&code=secret&next=%2Fsearch");
    trackPageview("/reset-password/confirm?userID=user-1&code=secret");
    trackPageview("/auth/zitadel/callback?code=secret&state=state");

    expect(fakeWindow.dataLayer.slice(-3)).toEqual([
      [
        "event",
        "page_view",
        {
          page_path: "/verify-email",
          page_location: "https://pandects.org/verify-email",
          page_title: "Search | Pandects",
        },
      ],
      [
        "event",
        "page_view",
        {
          page_path: "/reset-password/confirm",
          page_location: "https://pandects.org/reset-password/confirm",
          page_title: "Search | Pandects",
        },
      ],
      [
        "event",
        "page_view",
        {
          page_path: "/auth/zitadel/callback",
          page_location: "https://pandects.org/auth/zitadel/callback",
          page_title: "Search | Pandects",
        },
      ],
    ]);
  });

  it("strips sensitive query keys from otherwise normal routes", async () => {
    const { fakeWindow } = installBrowserGlobals();
    const { bootstrapAnalytics, trackPageview } = await importAnalytics();

    bootstrapAnalytics();
    trackPageview(
      "/search?q=tax&code=secret&UserID=user-1&redirect_uri=https%3A%2F%2Fevil.example%2Fcb&page=2",
    );

    expect(fakeWindow.dataLayer[fakeWindow.dataLayer.length - 1]).toEqual([
      "event",
      "page_view",
      {
        page_path: "/search?q=tax&page=2",
        page_location: "https://pandects.org/search?q=tax&page=2",
        page_title: "Search | Pandects",
      },
    ]);
  });

  it("redacts sensitive routes from time-on-page events", async () => {
    const { fakeWindow } = installBrowserGlobals();
    const { bootstrapAnalytics, trackTimeOnPage } = await importAnalytics();

    bootstrapAnalytics();
    trackTimeOnPage("/account?next=%2Fsearch&email=user%40example.com", 1500);

    expect(fakeWindow.dataLayer[fakeWindow.dataLayer.length - 1]).toEqual([
      "event",
      "time_on_page",
      {
        page_path: "/account",
        page_location: "https://pandects.org/account",
        page_title: "Search | Pandects",
        duration_ms: 1500,
        transport_type: "beacon",
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
