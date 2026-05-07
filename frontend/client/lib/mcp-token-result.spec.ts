import { afterEach, describe, expect, it, vi } from "vitest";
import type { NavigateFunction } from "react-router-dom";

vi.mock("@/lib/api-config", () => ({
  apiUrl: (endpoint: string) => `https://api.pandects.org/${endpoint.replace(/^\/+/, "")}`,
}));

import { navigateToMcpTokenResult } from "./mcp-token-result";

describe("navigateToMcpTokenResult", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("navigates to the MCP next path without writing token data to sessionStorage", () => {
    const navigateMock = vi.fn();
    const navigate = navigateMock as unknown as NavigateFunction;
    const setItem = vi.fn();
    vi.stubGlobal("sessionStorage", { setItem });

    navigateToMcpTokenResult(navigate, {
      status: "mcp_token",
      next_path: "/account",
      access_token: "secret-access-token",
      token_type: "Bearer",
    });

    expect(setItem).not.toHaveBeenCalled();
    expect(navigateMock).toHaveBeenCalledWith("/account", { replace: true });
  });
});
