const STORAGE_KEY = "pandects.sessionToken";

function canUseLocalStorage(): boolean {
  try {
    return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
  } catch {
    return false;
  }
}

export function getSessionToken(): string | null {
  if (!canUseLocalStorage()) return null;
  const token = localStorage.getItem(STORAGE_KEY);
  return token && token.trim().length > 0 ? token : null;
}

export function setSessionToken(token: string) {
  if (!canUseLocalStorage()) return;
  localStorage.setItem(STORAGE_KEY, token);
}

export function clearSessionToken() {
  if (!canUseLocalStorage()) return;
  localStorage.removeItem(STORAGE_KEY);
}
