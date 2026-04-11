export function readSessionCache<T>(key: string, maxAgeMs: number): T | null {
  try {
    const raw = sessionStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { ts?: unknown; data?: T };
    if (typeof parsed?.ts !== "number") {
      sessionStorage.removeItem(key);
      return null;
    }
    if (Date.now() - parsed.ts > maxAgeMs) {
      sessionStorage.removeItem(key);
      return null;
    }
    return parsed.data ?? null;
  } catch {
    sessionStorage.removeItem(key);
    return null;
  }
}

export function writeSessionCache<T>(key: string, data: T): void {
  try {
    sessionStorage.setItem(
      key,
      JSON.stringify({
        ts: Date.now(),
        data,
      }),
    );
  } catch {
    // Ignore storage failures.
  }
}
