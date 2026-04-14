import type { NavigateFunction } from "react-router-dom";

import { apiUrl } from "@/lib/api-config";

export function safeNextPath(value: string | null | undefined): string {
  if (!value) return "/account";
  const trimmed = value.trim();
  if (!trimmed.startsWith("/")) return "/account";
  if (trimmed.startsWith("//")) return "/account";
  return trimmed;
}

export function nextPathRequiresDocumentNavigation(value: string | null | undefined): boolean {
  return safeNextPath(value).startsWith("/v1/auth/");
}

export function nextPathHref(value: string | null | undefined): string {
  const nextPath = safeNextPath(value);
  if (nextPathRequiresDocumentNavigation(nextPath)) {
    return apiUrl(nextPath);
  }
  return nextPath;
}

export function navigateToNextPath(
  navigate: NavigateFunction,
  value: string | null | undefined,
  options?: { replace?: boolean },
): void {
  const href = nextPathHref(value);
  if (nextPathRequiresDocumentNavigation(value)) {
    if (options?.replace) {
      window.location.replace(href);
      return;
    }
    window.location.assign(href);
    return;
  }
  navigate(href, options);
}

export function buildAccountPathWithNext(nextPath: string): string {
  const safeNext = safeNextPath(nextPath);
  if (safeNext === "/account") return "/account";
  return `/account?next=${encodeURIComponent(safeNext)}`;
}
