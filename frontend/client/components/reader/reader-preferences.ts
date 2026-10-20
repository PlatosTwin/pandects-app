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
