import { describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api-config", () => ({
  apiUrl: (endpoint: string) => `https://api.pandects.org/${endpoint.replace(/^\/+/, "")}`,
}));

import {
  buildAccountPathWithNext,
  nextPathHref,
  nextPathRequiresDocumentNavigation,
  safeNextPath,
} from "./auth-next";

describe("safeNextPath", () => {
  it("defaults to /account when value is missing", () => {
    expect(safeNextPath(null)).toBe("/account");
    expect(safeNextPath(undefined)).toBe("/account");
  });

  it("rejects non-relative and protocol-relative values", () => {
    expect(safeNextPath("https://example.com")).toBe("/account");
    expect(safeNextPath("//evil.example/path")).toBe("/account");
  });

  it("accepts safe relative paths", () => {
    expect(safeNextPath("/search?query=test")).toBe("/search?query=test");
  });
});

describe("buildAccountPathWithNext", () => {
  it("omits next when destination is /account", () => {
    expect(buildAccountPathWithNext("/account")).toBe("/account");
  });

  it("encodes next path for account route", () => {
    expect(buildAccountPathWithNext("/search?query=test")).toBe(
      "/account?next=%2Fsearch%3Fquery%3Dtest",
    );
  });
});

describe("nextPathRequiresDocumentNavigation", () => {
  it("requires a document navigation for backend auth routes", () => {
    expect(nextPathRequiresDocumentNavigation("/v1/auth/oauth/authorize?state=test")).toBe(true);
  });

  it("rejects backend auth paths that are not approved browser navigations", () => {
    expect(nextPathRequiresDocumentNavigation("/v1/auth/logout")).toBe(false);
  });

  it("keeps normal app routes inside the SPA", () => {
    expect(nextPathRequiresDocumentNavigation("/search?query=test")).toBe(false);
  });
});

describe("nextPathHref", () => {
  it("rewrites backend auth routes to the API origin", () => {
    expect(nextPathHref("/v1/auth/oauth/authorize?state=test")).toBe(
      "https://api.pandects.org/v1/auth/oauth/authorize?state=test",
    );
  });

  it("keeps app routes relative", () => {
    expect(nextPathHref("/search?query=test")).toBe("/search?query=test");
  });

  it("sanitizes nested next parameters on backend auth routes", () => {
    expect(nextPathHref("/v1/auth/zitadel/start?next=https://evil.example&provider=email")).toBe(
      "https://api.pandects.org/v1/auth/zitadel/start?next=%2Faccount&provider=email",
    );
  });

  it("drops unsupported provider and prompt values for zitadel starts", () => {
    expect(nextPathHref("/v1/auth/zitadel/start?next=/search&provider=github&prompt=consent")).toBe(
      "https://api.pandects.org/v1/auth/zitadel/start?next=%2Fsearch",
    );
  });

  it("keeps only supported oauth authorize query fields", () => {
    expect(
      nextPathHref(
        "/v1/auth/oauth/authorize?client_id=client&redirect_uri=https%3A%2F%2Fexample.com%2Fcb&response_type=code&state=abc&scope=openid&code_challenge=xyz&code_challenge_method=S256&next=%2Fsearch",
      ),
    ).toBe(
      "https://api.pandects.org/v1/auth/oauth/authorize?client_id=client&redirect_uri=https%3A%2F%2Fexample.com%2Fcb&response_type=code&state=abc&scope=openid&code_challenge=xyz&code_challenge_method=S256",
    );
  });

  it("falls back to /account for unapproved backend auth routes", () => {
    expect(nextPathHref("/v1/auth/logout")).toBe("/account");
  });
});
