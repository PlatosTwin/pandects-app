/**
 * Links into the agreement reader (`/agreements/:uuid`).
 *
 * The `from` contract: `from` is the app-relative path (pathname + search,
 * optionally a hash) of the page the reader was opened from, e.g.
 * `/search?standard_id=abc&year=2023` or `/favorites`. The reader uses it to
 * label and target its back button, and, for `/search` origins, to re-run the
 * originating search scoped to the agreement so the reader can step through
 * every match. Anything that is not a same-origin path (`//host`, `https://`,
 * `javascript:`) is ignored.
 */

export type AgreementOriginKind =
  | "search"
  | "favorites"
  | "index"
  | "comparison"
  | "other";

export interface AgreementOrigin {
  kind: AgreementOriginKind;
  /** Safe, app-relative path to navigate back to. */
  href: string;
  /** Visible back-button label, e.g. "Back to Search". */
  label: string;
}

export interface BuildAgreementHrefOptions {
  agreementUuid: string;
  /** Section to focus (scroll to and mark as matched). */
  sectionUuid?: string | null;
  /**
   * Page the link is rendered on. Pass `location.pathname + location.search`
   * (or a router `Location`) so the reader can offer "Back to …".
   */
  from?: string | { pathname: string; search?: string; hash?: string } | null;
}

/** Path to the agreement reader, carrying focus and origin. */
export function buildAgreementHref({
  agreementUuid,
  sectionUuid,
  from,
}: BuildAgreementHrefOptions): string {
  const params = new URLSearchParams();
  const fromPath =
    typeof from === "string"
      ? from
      : from
        ? `${from.pathname}${from.search ?? ""}${from.hash ?? ""}`
        : null;
  const safeFrom = safeOriginPath(fromPath);
  if (safeFrom) params.set("from", safeFrom);
  if (sectionUuid) params.set("focusSectionUuid", sectionUuid);
  const query = params.toString();
  return `/agreements/${encodeURIComponent(agreementUuid)}${query ? `?${query}` : ""}`;
}

/** Path-only link to one section, for "copy link to section". */
export function buildSectionHref(agreementUuid: string, sectionUuid: string): string {
  return buildAgreementHref({ agreementUuid, sectionUuid });
}

/** Base used to resolve `from` when there is no window (prerender, tests). */
const FALLBACK_ORIGIN = "https://pandects.invalid";
/**
 * Control characters (U+0000-U+001F, U+007F) and backslashes. Browsers strip tab/CR/LF from URLs and
 * treat `\` as `/`, so `/\t/evil.test` would navigate off-site.
 */
function hasUnsafePathChar(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code <= 0x1f || code === 0x7f || code === 0x5c) return true;
  }
  return false;
}

function currentOrigin(): string {
  if (typeof window !== "undefined" && window.location?.origin && window.location.origin !== "null") {
    return window.location.origin;
  }
  return FALLBACK_ORIGIN;
}

/** Returns `value` when it is a same-origin app path, otherwise null. */
export function safeOriginPath(value: string | null | undefined): string | null {
  if (!value) return null;
  const trimmed = value.trim();
  if (hasUnsafePathChar(trimmed)) return null;
  if (!trimmed.startsWith("/") || trimmed.startsWith("//")) return null;
  const origin = currentOrigin();
  let resolved: URL;
  try {
    resolved = new URL(trimmed, origin);
  } catch {
    return null;
  }
  if (resolved.origin !== origin) return null;
  // `/%2F/host` and `/%5C/host` are same-origin paths, but a router or server
  // that decodes them could see `//host`; refuse them outright.
  if (/^\/(?:%2f|%5c)/i.test(resolved.pathname)) return null;
  const path = `${resolved.pathname}${resolved.search}${resolved.hash}`;
  // Never point "back" at the reader itself.
  if (/^\/agreements\//.test(path)) return null;
  return path;
}

const ORIGIN_RULES: Array<{ test: RegExp; kind: AgreementOriginKind; label: string }> = [
  { test: /^\/search(?:[/?#]|$)/, kind: "search", label: "Back to Search" },
  { test: /^\/favorites(?:[/?#]|$)/, kind: "favorites", label: "Back to Favorites" },
  { test: /^\/agreement-index(?:[/?#]|$)/, kind: "index", label: "Back to Index" },
  { test: /^\/compare(?:[/?#]|$)/, kind: "comparison", label: "Back to comparison" },
];

/** Resolves `from` into a back target, or null when there is no usable origin. */
export function parseAgreementOrigin(from: string | null | undefined): AgreementOrigin | null {
  const href = safeOriginPath(from);
  if (!href) return null;
  const rule = ORIGIN_RULES.find((candidate) => candidate.test.test(href));
  if (rule) return { kind: rule.kind, href, label: rule.label };
  return { kind: "other", href, label: "Back" };
}

/**
 * Search params of a `/search` origin, or null for any other origin. The
 * search page encodes its filters with the same keys the API accepts.
 */
export function originSearchParams(origin: AgreementOrigin | null): URLSearchParams | null {
  if (!origin || origin.kind !== "search") return null;
  const queryIndex = origin.href.indexOf("?");
  if (queryIndex === -1) return new URLSearchParams();
  const hashIndex = origin.href.indexOf("#", queryIndex);
  return new URLSearchParams(
    origin.href.slice(queryIndex + 1, hashIndex === -1 ? undefined : hashIndex),
  );
}
