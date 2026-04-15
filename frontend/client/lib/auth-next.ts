import type { NavigateFunction } from "react-router-dom";

import { apiUrl } from "@/lib/api-config";

const DOCUMENT_NAVIGATION_PATHS = new Set([
  "/v1/auth/oauth/authorize",
  "/v1/auth/external-subjects/zitadel/start",
  "/v1/auth/zitadel/start",
  "/v1/auth/mcp-token/start",
  "/v1/auth/zitadel/google/start",
]);

export function safeNextPath(value: string | null | undefined): string {
  if (!value) return "/account";
  const trimmed = value.trim();
  if (!trimmed.startsWith("/")) return "/account";
  if (trimmed.startsWith("//")) return "/account";
  return trimmed;
}

function safeDocumentNavigationHref(value: string | null | undefined): string | null {
  const nextPath = safeNextPath(value);
  const nextUrl = new URL(nextPath, "https://pandects.local");
  if (nextUrl.origin !== "https://pandects.local") return null;
  if (!DOCUMENT_NAVIGATION_PATHS.has(nextUrl.pathname)) return null;

  const nestedNext = nextUrl.searchParams.get("next");
  if (nestedNext !== null) {
    nextUrl.searchParams.set("next", safeNextPath(nestedNext));
  }

  return apiUrl(`${nextUrl.pathname}${nextUrl.search}${nextUrl.hash}`);
}

export function nextPathRequiresDocumentNavigation(value: string | null | undefined): boolean {
  return safeDocumentNavigationHref(value) !== null;
}

export function nextPathHref(value: string | null | undefined): string {
  const documentNavigationHref = safeDocumentNavigationHref(value);
  if (documentNavigationHref !== null) {
    return documentNavigationHref;
  }
  const nextPath = safeNextPath(value);
  if (nextPath.startsWith("/v1/auth/")) {
    return "/account";
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
