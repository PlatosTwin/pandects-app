// @vitest-environment jsdom
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { createElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const authApi = vi.hoisted(() => ({
  fetchOAuthConsentClient: vi.fn(),
  grantOAuthConsent: vi.fn(),
}));
vi.mock("@/lib/auth-api", () => authApi);

import { ApiError } from "@/lib/api/client";
import OAuthConsent from "./OAuthConsent";

const baseParams = {
  client_id: "abc123",
  redirect_uri: "https://codex.example.com/callback",
  response_type: "code",
  scope: "agreements:read sections:search",
  state: "test-state",
  code_challenge: "challenge",
  code_challenge_method: "S256",
};

function renderConsent(overrides: Record<string, string> = {}) {
  const query = new URLSearchParams({ ...baseParams, ...overrides }).toString();
  return render(
    createElement(
      MemoryRouter,
      { initialEntries: [`/oauth/consent?${query}`] },
      createElement(OAuthConsent),
    ),
  );
}

function registration(client_name: string | null, redirect_uris: string[]) {
  authApi.fetchOAuthConsentClient.mockResolvedValue({
    client_id: "abc123",
    client_name,
    redirect_uris,
  });
}

let assigned: string | null;

beforeEach(() => {
  assigned = null;
  vi.stubGlobal("location", {
    ...window.location,
    set href(value: string) {
      assigned = value;
    },
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("OAuthConsent", () => {
  it("renders the registered name, host, scope descriptions, and consent buttons", async () => {
    registration("Codex MCP", ["https://codex.example.com/callback"]);
    renderConsent({ client_name: "Spoofed name in the URL" });
    expect(await screen.findByText("Codex MCP")).toBeTruthy();
    expect(authApi.fetchOAuthConsentClient).toHaveBeenCalledWith({
      client_id: "abc123",
      redirect_uri: "https://codex.example.com/callback",
    });
    expect(screen.queryByText("Spoofed name in the URL")).toBeNull();
    expect(screen.getByText("codex.example.com")).toBeTruthy();
    expect(screen.getByText("Read agreement metadata.")).toBeTruthy();
    expect(
      screen.getByText("Search sections in the Pandects database."),
    ).toBeTruthy();
    expect(screen.getByText("Unrecognized application")).toBeTruthy();
    expect(document.body.textContent).toContain(
      "An external application wants to access your Pandects account.",
    );
  });

  it("names a known client with no warning", async () => {
    const uri = "https://claude.ai/api/mcp/auth_callback";
    registration("Claude", [uri]);
    renderConsent({ redirect_uri: uri });
    expect(await screen.findByText("claude.ai")).toBeTruthy();
    expect(document.body.textContent).toContain(
      "Claude wants to access your Pandects account.",
    );
    expect(screen.queryByText("Unrecognized application")).toBeNull();
    expect(screen.queryByText(/This may not be/)).toBeNull();
  });

  it("warns when a known host is registered alongside another host", async () => {
    const uri = "https://claude.ai/api/mcp/auth_callback";
    registration("Claude", [uri, "https://evil.example/cb"]);
    renderConsent({ redirect_uri: uri });
    expect(await screen.findByText("This may not be Claude")).toBeTruthy();
    expect(document.body.textContent).toContain(
      "Claude doesn't use evil.example",
    );
    expect(document.body.textContent).toContain(
      "Allowing also lets this application sign in through",
    );
    expect(document.body.textContent).not.toContain("Claude wants to access");
  });

  it("labels loopback redirects as a local app without a warning", async () => {
    const uri = "http://127.0.0.1:53682/callback";
    registration("Claude Code", ["http://127.0.0.1/callback"]);
    renderConsent({ redirect_uri: uri });
    expect(await screen.findByText("127.0.0.1:53682")).toBeTruthy();
    expect(screen.getByText("(an app on this computer)")).toBeTruthy();
    expect(document.body.textContent).not.toContain("Allowing also lets");
    expect(screen.queryByText("Unrecognized application")).toBeNull();
  });

  it("shows the invalid-request alert when the server rejects the client or redirect", async () => {
    authApi.fetchOAuthConsentClient.mockRejectedValue(
      new ApiError({
        status: 400,
        message: "Invalid OAuth client or redirect URI.",
      }),
    );
    renderConsent();
    expect(
      await screen.findByText("Invalid authorization request"),
    ).toBeTruthy();
    expect(screen.queryByText("Allow")).toBeNull();
  });

  it("offers a retry and Deny when the lookup fails for another reason", async () => {
    authApi.fetchOAuthConsentClient.mockRejectedValueOnce(
      new ApiError({ status: 503, message: "down" }),
    );
    registration("Codex MCP", ["https://codex.example.com/callback"]);
    renderConsent();
    expect(await screen.findByText("Could not load this request")).toBeTruthy();
    expect(screen.queryByText("Invalid authorization request")).toBeNull();
    expect(screen.getByText("Deny")).toBeTruthy();
    fireEvent.click(screen.getByText("Try again"));
    expect(screen.getByRole("status").textContent).toBe("Loading…");
    expect(await screen.findByText("Codex MCP")).toBeTruthy();
    expect(authApi.fetchOAuthConsentClient).toHaveBeenCalledTimes(2);
  });

  it("shows an invalid-request alert when required OAuth params are missing", () => {
    render(
      createElement(
        MemoryRouter,
        { initialEntries: ["/oauth/consent?client_id=abc123"] },
        createElement(OAuthConsent),
      ),
    );
    expect(screen.getByText("Invalid authorization request")).toBeTruthy();
    expect(authApi.fetchOAuthConsentClient).not.toHaveBeenCalled();
  });

  it("Deny goes through /authorize with deny=1, never to the redirect URI directly", async () => {
    registration("Codex MCP", ["https://codex.example.com/callback"]);
    renderConsent();
    fireEvent.click(await screen.findByText("Deny"));
    const url = new URL(assigned ?? "");
    expect(url.pathname).toMatch(/\/v1\/auth\/oauth\/authorize$/);
    expect(url.searchParams.get("deny")).toBe("1");
    expect(url.searchParams.get("redirect_uri")).toBe(
      "https://codex.example.com/callback",
    );
    expect(url.searchParams.get("state")).toBe("test-state");
  });

  it("Allow records consent for the redirect URI, then replays /authorize", async () => {
    registration("Codex MCP", ["https://codex.example.com/callback"]);
    authApi.grantOAuthConsent.mockResolvedValue({ status: "granted" });
    renderConsent();
    fireEvent.click(await screen.findByText("Allow"));
    await waitFor(() => expect(assigned).not.toBeNull());
    expect(authApi.grantOAuthConsent).toHaveBeenCalledWith({
      client_id: "abc123",
      redirect_uri: "https://codex.example.com/callback",
      scope: "agreements:read sections:search",
    });
    const url = new URL(assigned ?? "");
    expect(url.pathname).toMatch(/\/v1\/auth\/oauth\/authorize$/);
    expect(url.searchParams.get("deny")).toBeNull();
    expect(url.searchParams.get("code_challenge")).toBe("challenge");
  });
});
