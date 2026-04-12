let sessionToken: string | null = null;

function canUseBrowserMemory(): boolean {
  return typeof window !== "undefined";
}

export function getSessionToken(): string | null {
  if (!canUseBrowserMemory()) return null;
  return sessionToken && sessionToken.trim().length > 0 ? sessionToken : null;
}

export function setSessionToken(token: string) {
  if (!canUseBrowserMemory()) return;
  sessionToken = token;
}

export function clearSessionToken() {
  if (!canUseBrowserMemory()) return;
  sessionToken = null;
}
