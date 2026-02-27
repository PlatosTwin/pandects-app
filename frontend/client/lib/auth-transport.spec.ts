import { afterEach, describe, expect, it, vi } from "vitest";
import { authSessionTransport } from "./auth-transport";

describe("authSessionTransport", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("returns cookie when explicitly configured", () => {
    vi.stubEnv("VITE_AUTH_SESSION_TRANSPORT", "cookie");
    expect(authSessionTransport()).toBe("cookie");
  });

  it("returns bearer when explicitly configured", () => {
    vi.stubEnv("VITE_AUTH_SESSION_TRANSPORT", "bearer");
    expect(authSessionTransport()).toBe("bearer");
  });

  it("falls back to cookie when not configured", () => {
    vi.stubEnv("VITE_AUTH_SESSION_TRANSPORT", "");
    expect(authSessionTransport()).toBe("cookie");
  });

  it("ignores invalid values and falls back", () => {
    vi.stubEnv("VITE_AUTH_SESSION_TRANSPORT", "invalid-value");
    expect(authSessionTransport()).toBe("cookie");
  });
});
