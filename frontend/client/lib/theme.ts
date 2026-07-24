/**
 * Theme (light / dark / system) state.
 *
 * The stored preference and the class-toggling logic MUST stay in sync with
 * the pre-paint inline script in index.html, which applies the theme before
 * first paint (and is CSP-hashed — editing it requires regenerating
 * shared/csp-script-hashes.generated.json and the nginx/netlify CSP headers).
 */

export type ThemePreference = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

/** localStorage key; absence means "system". Mirrored by the inline script. */
export const THEME_STORAGE_KEY = "pandects-theme";

/** Browser-chrome color per theme; dark is hsl(225 6% 6%) — the dark --cream. */
const THEME_COLOR: Record<ResolvedTheme, string> = {
  light: "#ffffff",
  dark: "#0f0e10",
};

const listeners = new Set<() => void>();
let systemMedia: MediaQueryList | null = null;

function media(): MediaQueryList {
  if (!systemMedia) {
    systemMedia = window.matchMedia("(prefers-color-scheme: dark)");
    systemMedia.addEventListener("change", () => {
      if (getThemePreference() === "system") {
        applyTheme();
      }
      notifyListeners();
    });
  }
  return systemMedia;
}

function notifyListeners(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function getThemePreference(): ThemePreference {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "light" || stored === "dark") {
      return stored;
    }
  } catch {
    // localStorage unavailable (private mode / SSR) — fall through to system.
  }
  return "system";
}

export function resolveTheme(
  preference: ThemePreference = getThemePreference(),
): ResolvedTheme {
  if (preference === "system") {
    return media().matches ? "dark" : "light";
  }
  return preference;
}

/** Applies the current preference to the document (class, color-scheme, theme-color). */
export function applyTheme(): void {
  const resolved = resolveTheme();
  const root = document.documentElement;
  root.classList.toggle("dark", resolved === "dark");
  // Inline style rather than a CSS rule: the pre-paint script already sets the
  // style attribute, and a class-based rule would lose to it on later toggles.
  root.style.colorScheme = resolved;
  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute("content", THEME_COLOR[resolved]);
}

export function setThemePreference(preference: ThemePreference): void {
  try {
    if (preference === "system") {
      localStorage.removeItem(THEME_STORAGE_KEY);
    } else {
      localStorage.setItem(THEME_STORAGE_KEY, preference);
    }
  } catch {
    // Persistence failed; still apply for this session.
  }
  applyTheme();
  notifyListeners();
}

/**
 * Subscribes to preference/system changes. Installing the subscription also
 * arms the matchMedia listener that keeps "system" live.
 */
export function subscribeTheme(listener: () => void): () => void {
  media();
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
