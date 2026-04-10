import { AuthApiError } from "@/lib/auth-fetch";

const AUTH_WAKEUP_RETRY_DELAYS_MS = [300, 700, 1400] as const;

export const AUTH_WAKEUP_MESSAGE = "Waiting for auth DB to wake up…";

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
