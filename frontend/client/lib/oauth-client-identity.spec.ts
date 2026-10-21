import { describe, expect, it } from "vitest";
import { classifyOAuthClient } from "./oauth-client-identity";

describe("classifyOAuthClient", () => {
  it("recognizes known clients by their redirect host", () => {
    expect(
      classifyOAuthClient("Claude", "https://claude.ai/api/mcp/auth_callback"),
    ).toEqual({
      kind: "known",
      host: "claude.ai",
      knownLabel: "Claude",
    });
    expect(
      classifyOAuthClient(
        null,
        "https://chatgpt.com/connector_platform_oauth_redirect",
      ).kind,
    ).toBe("known");
  });

  it("flags a known name on someone else's host", () => {
    expect(
      classifyOAuthClient(
        "Claude Desktop",
        "https://claude.ai.evil.example/cb",
      ),
    ).toEqual({
      kind: "impersonation",
      host: "claude.ai.evil.example",
      knownLabel: "Claude",
    });
    expect(
      classifyOAuthClient("OpenAI ChatGPT", "https://evil.example/cb").kind,
    ).toBe("impersonation");
  });

  it("treats loopback redirects as local apps, whatever they're called", () => {
    expect(
      classifyOAuthClient("Claude Code", "http://127.0.0.1:53682/callback"),
    ).toEqual({
      kind: "local",
      host: "127.0.0.1:53682",
      knownLabel: null,
    });
    expect(classifyOAuthClient(null, "http://localhost:3000/cb").kind).toBe(
      "local",
    );
  });

  it("marks everything else unknown", () => {
    expect(
      classifyOAuthClient("Codex MCP", "https://codex.example.com/callback")
        .kind,
    ).toBe("unknown");
    expect(classifyOAuthClient(null, "not a url").kind).toBe("unknown");
    // A known host over plain http is not the real client.
    expect(classifyOAuthClient(null, "http://claude.ai/cb").kind).toBe(
      "unknown",
    );
  });
});
