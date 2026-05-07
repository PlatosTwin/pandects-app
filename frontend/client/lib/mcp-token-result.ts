import type { NavigateFunction } from "react-router-dom";

import type { McpTokenResult } from "@/lib/auth-api";
import { navigateToNextPath } from "@/lib/auth-next";

export function navigateToMcpTokenResult(
  navigate: NavigateFunction,
  payload: McpTokenResult,
): void {
  navigateToNextPath(navigate, payload.next_path, { replace: true });
}
