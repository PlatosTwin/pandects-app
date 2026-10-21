import { describe, expect, it } from "vitest";
import { classifyOAuthClient } from "./oauth-client-identity";

const CLAUDE_CALLBACK = "https://claude.ai/api/mcp/auth_callback";

describe("classifyOAuthClient", () => {
  it("recognizes a known client when every redirect is on its host", () => {
    expect(
      classifyOAuthClient("Claude", CLAUDE_CALLBACK, [CLAUDE_CALLBACK]),
    ).toEqual({
      kind: "known",
      host: "claude.ai",
      otherHosts: [],
      knownLabel: "Claude",
      foreignHosts: [],
    });
    const chatgpt = "https://chatgpt.com/connector_platform_oauth_redirect";
    expect(classifyOAuthClient(null, chatgpt, [chatgpt]).kind).toBe("known");
    // A trailing dot names the same host.
    expect(
      classifyOAuthClient("Claude", "https://claude.ai./cb", [
        "https://claude.ai./cb",
      ]).kind,
    ).toBe("known");
  });

  it("does not trust a known host when another registered redirect goes elsewhere", () => {
    const identity = classifyOAuthClient("Claude", CLAUDE_CALLBACK, [
      CLAUDE_CALLBACK,
      "https://evil.example/cb",
    ]);
    expect(identity).toEqual({
      kind: "impersonation",
      host: "claude.ai",
      otherHosts: ["evil.example"],
      knownLabel: "Claude",
      foreignHosts: ["evil.example"],
    });
    expect(
      classifyOAuthClient("Helper", CLAUDE_CALLBACK, [
        CLAUDE_CALLBACK,
        "https://evil.example/cb",
      ]).kind,
    ).toBe("unknown");
  });

  it("matches known hosts exactly", () => {
    for (const uri of [
      "https://anything.claude.ai/cb",
      "https://claude.ai.evil.example/cb",
      "https://user@evil.example/cb",
    ]) {
      expect(classifyOAuthClient("Evil", uri, [uri]).kind).toBe("unknown");
    }
    expect(
      classifyOAuthClient("Claude", "https://anything.claude.ai/cb", [
        "https://anything.claude.ai/cb",
      ]),
    ).toMatchObject({
      kind: "impersonation",
      foreignHosts: ["anything.claude.ai"],
    });
    // Plain http to a known host is not the known client.
    expect(
      classifyOAuthClient(null, "http://claude.ai/cb", ["http://claude.ai/cb"])
        .kind,
    ).toBe("unknown");
  });

  it("only treats whole-word names as a claim", () => {
    const uri = "https://claudette.example/cb";
    expect(classifyOAuthClient("Claudette's tool", uri, [uri]).kind).toBe(
      "unknown",
    );
    expect(classifyOAuthClient("OpenAI Helper", uri, [uri]).kind).toBe(
      "impersonation",
    );
  });

  it("labels loopback-only redirects as a local app, whatever the name", () => {
    for (const uri of [
      "http://127.0.0.1:53682/callback",
      "http://localhost:3000/cb",
      "http://[::1]:8080/cb",
    ]) {
      expect(classifyOAuthClient("Claude Code", uri, [uri]).kind).toBe("local");
    }
    expect(
      classifyOAuthClient("Claude Code", "http://localhost:1/cb", [
        "http://localhost:1/cb",
        "https://evil.example/cb",
      ]).kind,
    ).toBe("impersonation");
  });

  it("treats unparseable or missing registrations as unknown", () => {
    expect(classifyOAuthClient("Claude", CLAUDE_CALLBACK, []).kind).toBe(
      "unknown",
    );
    expect(
      classifyOAuthClient("Claude", "not a url", ["not a url"]),
    ).toMatchObject({
      kind: "unknown",
      host: "not a url",
    });
  });
});
