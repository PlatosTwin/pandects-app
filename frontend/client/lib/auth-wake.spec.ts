import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api/client";
import { isAuthWakeupError, withAuthWakeRetry } from "@/lib/auth-wake";

function wakeupError(
  message = "Auth backend is unavailable",
  bodyText = "remote auth provider is unavailable",
): ApiError {
  return new ApiError({
    message,
    status: 503,
    statusText: "Service Unavailable",
    bodyText,
  });
}

describe("isAuthWakeupError", () => {
  it("returns true for known 503 wake-up signatures", () => {
    const error = wakeupError("AUTH BACKEND IS UNAVAILABLE", "");
    expect(isAuthWakeupError(error)).toBe(true);
  });

  it("returns false for non-503 auth errors", () => {
    const error = new ApiError({
      message: "Auth backend is unavailable",
      status: 500,
      statusText: "Internal Server Error",
      bodyText: "",
    });

    expect(isAuthWakeupError(error)).toBe(false);
  });

  it("returns false for non-auth errors", () => {
    expect(isAuthWakeupError(new Error("boom"))).toBe(false);
  });
});

describe("withAuthWakeRetry", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("returns without retries when first attempt succeeds", async () => {
    const run = vi.fn().mockResolvedValue("ok");

    await expect(withAuthWakeRetry(run)).resolves.toBe("ok");
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("retries wake-up errors and succeeds on a later attempt", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("window", globalThis);
    const run = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(wakeupError())
      .mockRejectedValueOnce(wakeupError("", "remote auth provider is unavailable"))
      .mockResolvedValue("ok");

    const resultPromise = withAuthWakeRetry(run);
    await Promise.resolve();
    expect(run).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(300);
    expect(run).toHaveBeenCalledTimes(2);

    await vi.advanceTimersByTimeAsync(700);
    expect(run).toHaveBeenCalledTimes(3);

    await expect(resultPromise).resolves.toBe("ok");
  });

  it("does not retry non-wake-up errors", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("window", globalThis);
    const run = vi.fn<() => Promise<never>>().mockRejectedValue(new Error("boom"));

    const resultPromise = withAuthWakeRetry(run);
    const rejection = expect(resultPromise).rejects.toThrow("boom");
    await Promise.resolve();

    expect(run).toHaveBeenCalledTimes(1);
    await rejection;
  });

  it("rethrows after exhausting wake-up retries", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("window", globalThis);
    const run = vi.fn<() => Promise<never>>().mockRejectedValue(wakeupError());

    const resultPromise = withAuthWakeRetry(run);
    const rejection = expect(resultPromise).rejects.toThrow("Auth backend is unavailable");
    await Promise.resolve();
    expect(run).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(300 + 700 + 1400);
    await Promise.resolve();
    expect(run).toHaveBeenCalledTimes(5);

    await rejection;
  });
});
