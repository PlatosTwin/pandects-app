import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clearSessionToken, getSessionToken, setSessionToken } from "./auth-session";

describe("auth-session", () => {
  beforeEach(() => {
    vi.stubGlobal("window", {});
  });

  afterEach(() => {
    clearSessionToken();
    vi.unstubAllGlobals();
  });

  it("stores the session token only in memory", () => {
    expect(getSessionToken()).toBeNull();

    setSessionToken("test-session-token");
    expect(getSessionToken()).toBe("test-session-token");

    clearSessionToken();
    expect(getSessionToken()).toBeNull();
  });

  it("ignores empty bearer tokens", () => {
    setSessionToken("   ");
    expect(getSessionToken()).toBeNull();
  });
});
