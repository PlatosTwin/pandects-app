import { buildZitadelAuthorizeUrl, type ZitadelLinkConfig } from "@/lib/zitadel-link";
import { describe, expect, it } from "vitest";

describe("buildZitadelAuthorizeUrl", () => {
  it("includes PKCE and ZITADEL resource parameters", () => {
    const config: ZitadelLinkConfig = {
      authority: "https://zitadel.example.com",
      clientId: "pandects-web",
      redirectUri: "https://app.pandects.org/auth/zitadel/callback",
      authorizationEndpoint: "https://zitadel.example.com/oauth/v2/authorize",
      tokenEndpoint: "https://zitadel.example.com/oauth/v2/token",
      scopes: ["openid", "profile", "agreements:read"],
      resource: "https://api.pandects.org/mcp",
      audience: "https://api.pandects.org/mcp",
    };

    const url = new URL(
      buildZitadelAuthorizeUrl(config, {
        state: "state-123",
        codeChallenge: "challenge-456",
      }),
    );

    expect(url.origin + url.pathname).toBe(config.authorizationEndpoint);
    expect(url.searchParams.get("client_id")).toBe("pandects-web");
    expect(url.searchParams.get("redirect_uri")).toBe(config.redirectUri);
    expect(url.searchParams.get("scope")).toBe("openid profile agreements:read");
    expect(url.searchParams.get("state")).toBe("state-123");
    expect(url.searchParams.get("code_challenge")).toBe("challenge-456");
    expect(url.searchParams.get("code_challenge_method")).toBe("S256");
    expect(url.searchParams.get("resource")).toBe("https://api.pandects.org/mcp");
    expect(url.searchParams.get("audience")).toBe("https://api.pandects.org/mcp");
  });

  it("omits optional audience and resource when unset", () => {
    const config: ZitadelLinkConfig = {
      authority: "https://zitadel.example.com",
      clientId: "pandects-web",
      redirectUri: "https://app.pandects.org/auth/zitadel/callback",
      authorizationEndpoint: "https://zitadel.example.com/oauth/v2/authorize",
      tokenEndpoint: "https://zitadel.example.com/oauth/v2/token",
      scopes: ["openid"],
      resource: null,
      audience: null,
    };

    const url = new URL(
      buildZitadelAuthorizeUrl(config, {
        state: "state-123",
        codeChallenge: "challenge-456",
      }),
    );

    expect(url.searchParams.has("resource")).toBe(false);
    expect(url.searchParams.has("audience")).toBe(false);
  });
});
