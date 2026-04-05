import type { Request } from "express";
import {
  buildSeoPage,
  getRouteDefinition,
  injectSeoDocument,
  normalizePathname,
} from "../shared/seo-helpers.mjs";

export type SeoPage = ReturnType<typeof buildSeoPage>;

const DEFAULT_ORIGIN = "https://pandects.org";
const CONFIGURED_ORIGIN = (process.env.PUBLIC_ORIGIN || "").trim().replace(/\/+$/, "");

export function isKnownRoute(pathname: string): boolean {
  return getRouteDefinition(normalizePathname(pathname)) !== null;
}

export function getPublicOrigin(req: Request): string {
  const fallback = CONFIGURED_ORIGIN || DEFAULT_ORIGIN;

  const forwardedProto = req.get("x-forwarded-proto");
  const forwardedHost = req.get("x-forwarded-host");
  const hostHeader = forwardedHost ?? req.get("host");
  if (!hostHeader) return fallback;

  const host = hostHeader.split(",", 1)[0]?.trim();
  if (!host) return fallback;

  const protoHeader = (forwardedProto ?? req.protocol).split(",", 1)[0]?.trim().toLowerCase();
  const protocol = protoHeader === "http" || protoHeader === "https" ? protoHeader : "https";

  try {
    const url = new URL(`${protocol}://${host}`);
    if (url.username || url.password) return fallback;
    if (url.protocol !== "http:" && url.protocol !== "https:") return fallback;
    return url.origin.replace(/\/+$/, "");
  } catch {
    return fallback;
  }
}

export function getSeoForRequest(req: Request, origin: string): SeoPage {
  const requestUrl = new URL(req.originalUrl, `${origin}/`);
  return buildSeoPage(requestUrl.pathname, requestUrl.search, origin);
}

export { injectSeoDocument };
