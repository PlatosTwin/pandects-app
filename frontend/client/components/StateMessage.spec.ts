import { describe, expect, it } from "vitest";
import { describeError } from "./StateMessage";

describe("describeError", () => {
  it("treats an ApiError network failure (status 0) as unreachable", () => {
    const described = describeError({
      status: 0,
      code: "network",
      name: "ApiError",
      message: "Unable to reach the server.",
    });
    expect(described.title).toBe("Couldn't reach the server");
  });

  it("shows the server's reason for a 403 that isn't about signing in", () => {
    const described = describeError({
      status: 403,
      serverMessage: "Email address not verified.",
    });
    expect(described.body).toBe("Email address not verified.");
    expect(described.needsSignIn).toBe(false);
  });

  it("still asks to sign in for a sign-in 403", () => {
    const described = describeError({
      status: 403,
      serverMessage: "Favorites require a signed-in user session.",
    });
    expect(described.needsSignIn).toBe(true);
  });
});
