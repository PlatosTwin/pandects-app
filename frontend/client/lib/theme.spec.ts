// @vitest-environment jsdom
import fs from "node:fs";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type MediaListener = (event: { matches: boolean }) => void;

function installMatchMedia(initialMatches: boolean) {
  let matches = initialMatches;
  const changeListeners = new Set<MediaListener>();

  const mediaQueryList = {
    get matches() {
      return matches;
    },
    media: "(prefers-color-scheme: dark)",
    addEventListener: (_event: string, listener: MediaListener) => {
      changeListeners.add(listener);
    },
    removeEventListener: (_event: string, listener: MediaListener) => {
      changeListeners.delete(listener);
    },
  };

  vi.stubGlobal(
    "matchMedia",
    vi.fn(() => mediaQueryList),
  );

  return {
    setSystemDark(next: boolean) {
      matches = next;
      for (const listener of changeListeners) {
        listener({ matches: next });
      }
    },
  };
}

// theme.ts caches the MediaQueryList across tests; import a fresh module copy
// per test so each test's matchMedia stub is the one that gets cached.
async function importTheme() {
  vi.resetModules();
  return import("./theme");
}

describe("theme", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.classList.remove("dark");
    document.documentElement.style.colorScheme = "";
    document.head.innerHTML = '<meta name="theme-color" content="#ffffff" />';
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("defaults to system and resolves via prefers-color-scheme", async () => {
    installMatchMedia(true);
    const theme = await importTheme();

    expect(theme.getThemePreference()).toBe("system");
    expect(theme.resolveTheme()).toBe("dark");
  });

  it("persists explicit choices and clears storage for system", async () => {
    installMatchMedia(false);
    const theme = await importTheme();

    theme.setThemePreference("dark");
    expect(localStorage.getItem(theme.THEME_STORAGE_KEY)).toBe("dark");
    expect(theme.getThemePreference()).toBe("dark");

    theme.setThemePreference("system");
    expect(localStorage.getItem(theme.THEME_STORAGE_KEY)).toBeNull();
    expect(theme.getThemePreference()).toBe("system");
  });

  it("ignores unknown stored values", async () => {
    installMatchMedia(false);
    const theme = await importTheme();

    localStorage.setItem(theme.THEME_STORAGE_KEY, "solarized");
    expect(theme.getThemePreference()).toBe("system");
  });

  it("applyTheme toggles the dark class, color-scheme, and theme-color", async () => {
    installMatchMedia(false);
    const theme = await importTheme();
    const root = document.documentElement;
    const themeColor = () =>
      document
        .querySelector('meta[name="theme-color"]')
        ?.getAttribute("content");

    theme.setThemePreference("dark");
    expect(root.classList.contains("dark")).toBe(true);
    expect(root.style.colorScheme).toBe("dark");
    expect(themeColor()).toBe("#0f0e10");

    theme.setThemePreference("light");
    expect(root.classList.contains("dark")).toBe(false);
    expect(root.style.colorScheme).toBe("light");
    expect(themeColor()).toBe("#ffffff");
  });

  it("follows live system changes while preference is system", async () => {
    const media = installMatchMedia(false);
    const theme = await importTheme();
    const notified = vi.fn();
    theme.subscribeTheme(notified);

    theme.setThemePreference("system");
    expect(document.documentElement.classList.contains("dark")).toBe(false);

    media.setSystemDark(true);
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(notified).toHaveBeenCalled();

    // An explicit choice pins the theme regardless of system changes.
    theme.setThemePreference("light");
    media.setSystemDark(false);
    media.setSystemDark(true);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
  });

  it("stays in sync with the pre-paint script in index.html", () => {
    const indexHtml = fs.readFileSync(
      path.resolve(process.cwd(), "index.html"),
      "utf8",
    );

    // The inline script must read the same storage key and apply the same
    // class/color-scheme outputs as client/lib/theme.ts.
    expect(indexHtml).toContain('localStorage.getItem("pandects-theme")');
    expect(indexHtml).toContain('classList.toggle("dark", dark)');
    expect(indexHtml).toContain("style.colorScheme");
  });
});
