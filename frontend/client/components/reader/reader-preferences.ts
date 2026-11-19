const CONTENTS_OPEN_KEY = "pandects.reader.contentsOpen";

/** Desktop Contents panel open/closed, remembered per browser. Defaults to open. */
export function readContentsOpen(): boolean {
  try {
    const stored = window.localStorage.getItem(CONTENTS_OPEN_KEY);
    return stored === null ? true : stored === "1";
  } catch {
    return true;
  }
}

export function writeContentsOpen(open: boolean): void {
  try {
    window.localStorage.setItem(CONTENTS_OPEN_KEY, open ? "1" : "0");
  } catch {
    // Storage can be unavailable (private mode, blocked site data); the
    // panel still works, it just isn't remembered.
  }
}

const SMART_READER_KEY = "pandects.reader.smart";

/**
 * The smart reader (defined terms, cross-reference links, subsection contents)
 * is on in development and when VITE_SMART_READER=1 is set at build time. The
 * production build sets it in fly.toml since 2026-11-18. A build without it
 * keeps the reader off, except for one browser after
 * `localStorage.setItem("pandects.reader.smart", "1")`.
 */
export function isSmartReaderEnabled(): boolean {
  if (!import.meta.env.PROD || import.meta.env.VITE_SMART_READER === "1") return true;
  try {
    return window.localStorage.getItem(SMART_READER_KEY) === "1";
  } catch {
    return false;
  }
}

/** Defined-term underlines: off, on for distinctive terms, or on for all (incl. "Law"). */
export type TermHighlightMode = "off" | "distinctive" | "all";

const TERM_HIGHLIGHT_KEY = "pandects.reader.terms";

export function readTermHighlightMode(): TermHighlightMode {
  try {
    const stored = window.localStorage.getItem(TERM_HIGHLIGHT_KEY);
    return stored === "off" || stored === "all" ? stored : "distinctive";
  } catch {
    return "distinctive";
  }
}

export function writeTermHighlightMode(mode: TermHighlightMode): void {
  try {
    window.localStorage.setItem(TERM_HIGHLIGHT_KEY, mode);
  } catch {
    // Not remembered when storage is unavailable; the choice still applies.
  }
}
