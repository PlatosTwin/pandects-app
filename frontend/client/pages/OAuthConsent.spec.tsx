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
    // The self-reported name never becomes the headline for an unknown host.
    expect(markup).toContain(
      "An external application wants to access your Pandects account.",
    );
    expect(markup).toContain("Unrecognized application");
  });

  function consentQuery(overrides: Record<string, string>): string {
    const params = new URLSearchParams(validQuery);
    for (const [k, v] of Object.entries(overrides)) params.set(k, v);
    return params.toString();
  }

  it("names a known client from its redirect host, with no warning", () => {
    const markup = renderAt(
      `/oauth/consent?${consentQuery({
        client_name: "Claude",
        redirect_uri: "https://claude.ai/api/mcp/auth_callback",
      })}`,
    );
    expect(markup).toContain("Claude wants to access your Pandects account.");
    expect(markup).not.toContain("Unrecognized application");
    expect(markup).not.toContain("This may not be");
  });

  it("warns when a client claims a known name but redirects elsewhere", () => {
    const markup = renderAt(
      `/oauth/consent?${consentQuery({
        client_name: "Claude",
        redirect_uri: "https://claude-login.evil.example/cb",
      })}`,
    );
    expect(markup).toContain("This may not be Claude");
    expect(markup).toContain("claude-login.evil.example");
    expect(markup).not.toContain("Claude wants to access");
  });

  it("labels loopback redirects as a local app without a warning", () => {
    const markup = renderAt(
      `/oauth/consent?${consentQuery({
        client_name: "Claude Code",
        redirect_uri: "http://127.0.0.1:53682/callback",
      })}`,
    );
    expect(markup).toContain("127.0.0.1:53682");
    expect(markup).toContain("(an app on this computer)");
    expect(markup).not.toContain("Unrecognized application");
    expect(markup).not.toContain("This may not be");
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
    expect(markup).toContain(
      "An external application wants to access your Pandects account.",
    );
  });
});
