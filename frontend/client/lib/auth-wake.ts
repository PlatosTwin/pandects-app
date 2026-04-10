import { AuthApiError } from "@/lib/auth-fetch";
import { apiUrl } from "@/lib/api-config";

const AUTH_WAKEUP_RETRY_DELAYS_MS = [300, 700, 1400] as const;
const AUTH_PREWARM_TTL_MS = 30_000;

export const AUTH_WAKEUP_MESSAGE = "Waiting for auth DB to wake up…";

let authPrewarmPromise: Promise<void> | null = null;
let authPrewarmAtMs = 0;

function sleep(ms: number) {
  return new Promise<void>((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

export function isAuthWakeupError(error: unknown): error is AuthApiError {
  if (!(error instanceof AuthApiError)) return false;
  if (error.status !== 503) return false;

  const combined = `${error.message}\n${error.bodyText}`.toLowerCase();
  return (
    combined.includes("auth backend is unavailable") ||
    combined.includes("remote auth provider is unavailable")
  );
}

export function shouldHandleAuthWakeupMessage(message: string): boolean {
  const normalized = message.toLowerCase();
  return (
    normalized.includes("auth backend is unavailable") ||
    normalized.includes("remote auth provider is unavailable") ||
    normalized.includes("external identity provider is unavailable")
  );
}

export async function prewarmAuthBackend(): Promise<void> {
  const now = Date.now();
  if (authPrewarmPromise && now - authPrewarmAtMs < AUTH_PREWARM_TTL_MS) {
    return authPrewarmPromise;
  }

  authPrewarmAtMs = now;
  authPrewarmPromise = fetch(apiUrl("v1/auth/health"), {
    credentials: "include",
  })
    .then(() => undefined)
    .catch(() => undefined)
    .finally(() => {
      authPrewarmPromise = null;
    });

  return authPrewarmPromise;
}

export async function withAuthWakeRetry<T>(run: () => Promise<T>): Promise<T> {
  try {
    return await run();
  } catch (error) {
    if (!isAuthWakeupError(error)) {
      throw error;
    }
  }

  for (const delayMs of AUTH_WAKEUP_RETRY_DELAYS_MS) {
    await sleep(delayMs);
    try {
      return await run();
    } catch (error) {
      if (!isAuthWakeupError(error)) {
        throw error;
      }
    }
  }

  return run();
}
