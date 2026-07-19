/**
 * First-party telemetry: page views recorded to the Pandects API (the
 * `page_views` table read by the usage-analytics dashboard) and a
 * first-touch attribution cookie the backend snapshots at signup.
 *
 * Independent of Google Analytics: no third parties, no identifiers beyond
 * the existing session cookie, paths sanitized with the same rules as GA.
 */

import { sanitizeAnalyticsPath } from "@/lib/analytics";
import { apiUrl } from "@/lib/api-config";

const ATTRIBUTION_COOKIE_NAME = "pdcts_attr";
const ATTRIBUTION_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 90;
const MAX_COOKIE_FIELD_LENGTH = 300;

let lastRecordedPath: string | null = null;
let lastPageViewWasFirst = true;

function telemetryDisabled(): boolean {
  return (
    typeof window === "undefined" || import.meta.env.VITE_DISABLE_ANALYTICS === "1"
  );
}

/** POST the route change to /v1/page-views. Fire-and-forget. */
export function recordFirstPartyPageView(pagePath: string) {
  if (telemetryDisabled()) return;
  const safePath = sanitizeAnalyticsPath(pagePath);
  if (safePath === lastRecordedPath) return;
  lastRecordedPath = safePath;

  const payload: { path: string; referrer?: string } = { path: safePath };
  // Only meaningful on the landing view; after that document.referrer is
  // stale (SPA navigations don't update it), so send it once.
  if (document.referrer && lastPageViewWasFirst) {
    payload.referrer = document.referrer.slice(0, 512);
  }
  lastPageViewWasFirst = false;

  try {
    void fetch(apiUrl("v1/page-views"), {
      method: "POST",
      credentials: "include",
      keepalive: true,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).catch(() => undefined);
  } catch {
    // Never let telemetry break navigation.
  }
}

function cookieDomainAttribute(): string {
  const host = window.location.hostname;
  // Share the cookie across www./api. so the backend sees it at signup.
  if (host === "pandects.org" || host.endsWith(".pandects.org")) {
    return "; domain=.pandects.org";
  }
  return "";
}

function readCookie(name: string): string | null {
  const prefix = `${name}=`;
  for (const part of document.cookie.split("; ")) {
    if (part.startsWith(prefix)) return part.slice(prefix.length);
  }
  return null;
}

/**
 * On first landing, persist the acquisition context (external referrer +
 * UTM parameters + landing path) in a first-touch cookie. The backend reads
 * it when an account is registered and stores it in
 * `auth_signup_attributions`. No-op when the cookie already exists or the
 * visit has no attribution signal (direct navigation).
 */
export function captureAttributionOnce() {
  if (telemetryDisabled()) return;
  try {
    if (readCookie(ATTRIBUTION_COOKIE_NAME)) return;

    const params = new URLSearchParams(window.location.search);
    const value: Record<string, string> = {};
    const utmKeys: Array<[string, string]> = [
      ["utm_source", "s"],
      ["utm_medium", "m"],
      ["utm_campaign", "c"],
      ["utm_term", "t"],
      ["utm_content", "n"],
    ];
    for (const [param, short] of utmKeys) {
      const raw = params.get(param);
      if (raw && raw.trim()) value[short] = raw.trim().slice(0, MAX_COOKIE_FIELD_LENGTH);
    }

    if (document.referrer) {
      try {
        const referrerUrl = new URL(document.referrer);
        if (referrerUrl.origin !== window.location.origin) {
          value.r = document.referrer.slice(0, MAX_COOKIE_FIELD_LENGTH);
        }
      } catch {
        // Ignore unparseable referrers.
      }
    }

    // Direct visit with no campaign markers: nothing to attribute; leave the
    // cookie unset so a later tagged visit can still claim first touch.
    if (Object.keys(value).length === 0) return;

    value.l = sanitizeAnalyticsPath(
      `${window.location.pathname}${window.location.search}`,
    ).slice(0, MAX_COOKIE_FIELD_LENGTH);

    const encoded = encodeURIComponent(JSON.stringify(value));
    if (encoded.length > 1800) return;
    const secure = window.location.protocol === "https:" ? "; Secure" : "";
    document.cookie =
      `${ATTRIBUTION_COOKIE_NAME}=${encoded}; max-age=${ATTRIBUTION_COOKIE_MAX_AGE_SECONDS}` +
      `; path=/${cookieDomainAttribute()}; SameSite=Lax${secure}`;
  } catch {
    // Attribution is best-effort only.
  }
}
