import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import OAuthConsent from "./OAuthConsent";

function renderAt(url: string): string {
  return renderToStaticMarkup(
    createElement(
      MemoryRouter,
      { initialEntries: [url] },
      createElement(OAuthConsent),
    ),
  );
}

const validQuery = new URLSearchParams({
  client_id: "abc123",
  client_name: "Codex MCP",
  redirect_uri: "https://codex.example.com/callback",
  response_type: "code",
  scope: "agreements:read sections:search",
  state: "test-state",
  code_challenge: "challenge",
  code_challenge_method: "S256",
}).toString();

describe("OAuthConsent", () => {
  it("renders the client name, host, scope descriptions, and consent buttons", () => {
    const markup = renderAt(`/oauth/consent?${validQuery}`);
    expect(markup).toContain("Codex MCP");
    expect(markup).toContain("codex.example.com");
    // Friendly + raw scope identifiers both render.
    expect(markup).toContain("Read agreement metadata.");
    expect(markup).toContain("(agreements:read)");
    expect(markup).toContain("Search sections in the Pandects database.");
    expect(markup).toContain("(sections:search)");
    expect(markup).toContain(">Allow<");
    expect(markup).toContain(">Deny<");
  });

  it("shows an invalid-request alert when required OAuth params are missing", () => {
    const markup = renderAt("/oauth/consent?client_id=abc123");
    expect(markup).toContain("Invalid authorization request");
    expect(markup).not.toContain(">Allow<");
  });

  it("falls back to a generic subtitle when client_name is absent", () => {
    const params = new URLSearchParams({
      client_id: "abc123",
      redirect_uri: "https://codex.example.com/callback",
      response_type: "code",
      scope: "agreements:read",
      code_challenge: "challenge",
      code_challenge_method: "S256",
    }).toString();
    const markup = renderAt(`/oauth/consent?${params}`);
    expect(markup).toContain("An external application wants to access your Pandects account.");
  });
});
