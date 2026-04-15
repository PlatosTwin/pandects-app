import type { NavigateFunction } from "react-router-dom";

import { apiUrl } from "@/lib/api-config";

const OAUTH_AUTHORIZE_QUERY_KEYS = [
  "client_id",
  "redirect_uri",
  "response_type",
  "state",
  "scope",
  "code_challenge",
  "code_challenge_method",
] as const;
const ALLOWED_ZITADEL_PROVIDERS = new Set(["email", "google"]);
const ALLOWED_ZITADEL_PROMPTS = new Set(["login", "create", "select_account"]);

type DocumentNavigationTarget = {
  pathname: string;
  searchParams: URLSearchParams;
};

export function safeNextPath(value: string | null | undefined): string {
  if (!value) return "/account";
  const trimmed = value.trim();
  if (!trimmed.startsWith("/")) return "/account";
  if (trimmed.startsWith("//")) return "/account";
  return trimmed;
}

function buildDocumentNavigationTarget(value: string | null | undefined): DocumentNavigationTarget | null {
  const nextPath = safeNextPath(value);
  const nextUrl = new URL(nextPath, "https://pandects.local");
  if (nextUrl.origin !== "https://pandects.local") return null;

  if (nextUrl.pathname === "/v1/auth/oauth/authorize") {
    const searchParams = new URLSearchParams();
    for (const key of OAUTH_AUTHORIZE_QUERY_KEYS) {
      const param = nextUrl.searchParams.get(key);
      if (param) {
        searchParams.set(key, param);
      }
    }
    return { pathname: nextUrl.pathname, searchParams };
  }

  if (nextUrl.pathname === "/v1/auth/external-subjects/zitadel/start") {
    return {
      pathname: nextUrl.pathname,
      searchParams: new URLSearchParams({ next: safeNextPath(nextUrl.searchParams.get("next")) }),
    };
  }

  if (nextUrl.pathname === "/v1/auth/zitadel/start") {
    const searchParams = new URLSearchParams({ next: safeNextPath(nextUrl.searchParams.get("next")) });
    const provider = nextUrl.searchParams.get("provider");
    if (provider && ALLOWED_ZITADEL_PROVIDERS.has(provider)) {
      searchParams.set("provider", provider);
    }
    const prompt = nextUrl.searchParams.get("prompt");
    if (prompt && ALLOWED_ZITADEL_PROMPTS.has(prompt)) {
      searchParams.set("prompt", prompt);
    }
    return { pathname: nextUrl.pathname, searchParams };
  }

  if (nextUrl.pathname === "/v1/auth/mcp-token/start" || nextUrl.pathname === "/v1/auth/zitadel/google/start") {
    return {
      pathname: nextUrl.pathname,
      searchParams: new URLSearchParams({ next: safeNextPath(nextUrl.searchParams.get("next")) }),
    };
  }

  return null;
}

function safeDocumentNavigationHref(value: string | null | undefined): string | null {
  const target = buildDocumentNavigationTarget(value);
  if (target === null) return null;
  const search = target.searchParams.toString();
  return apiUrl(`${target.pathname}${search ? `?${search}` : ""}`);
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
  const documentNavigationHref = safeDocumentNavigationHref(value);
  if (documentNavigationHref !== null) {
    if (options?.replace) {
      window.location.replace(documentNavigationHref);
      return;
    }
    window.location.assign(documentNavigationHref);
    return;
  }
  navigate(nextPathHref(value), options);
}

export function buildAccountPathWithNext(nextPath: string): string {
  const safeNext = safeNextPath(nextPath);
  if (safeNext === "/account") return "/account";
  return `/account?next=${encodeURIComponent(safeNext)}`;
}
