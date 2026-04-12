function getSessionStorage(): Storage | null {
  if (typeof window === "undefined" || typeof window.sessionStorage === "undefined") {
    return null;
  }
  return window.sessionStorage;
}

export function readSessionCache<T>(key: string, maxAgeMs: number): T | null {
  const storage = getSessionStorage();
  if (!storage) return null;
  try {
    const raw = storage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { ts?: unknown; data?: T };
    if (typeof parsed?.ts !== "number") {
      storage.removeItem(key);
      return null;
    }
    if (Date.now() - parsed.ts > maxAgeMs) {
      storage.removeItem(key);
      return null;
    }
    return parsed.data ?? null;
  } catch {
    storage.removeItem(key);
    return null;
  }
}

export function writeSessionCache<T>(key: string, data: T): void {
  const storage = getSessionStorage();
  if (!storage) return;
  try {
    storage.setItem(
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
