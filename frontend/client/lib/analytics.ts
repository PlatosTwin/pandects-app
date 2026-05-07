type AnalyticsParamValue = string | number | boolean | null | undefined;
type AnalyticsParams = Record<string, AnalyticsParamValue>;

const DEFAULT_GA_MEASUREMENT_ID = "G-94X4EVQVHZ";
const MAX_PARAM_STRING_LENGTH = 500;
const GA_MEASUREMENT_ID =
  import.meta.env.VITE_GA_MEASUREMENT_ID?.trim() || DEFAULT_GA_MEASUREMENT_ID;
const SENSITIVE_ANALYTICS_PATHS = new Set([
  "/account",
  "/auth/zitadel/callback",
  "/login",
  "/reset-password",
  "/reset-password/confirm",
  "/signup",
  "/verify-email",
]);
const SENSITIVE_ANALYTICS_QUERY_KEYS = new Set([
  "access_token",
  "code",
  "email",
  "id_token",
  "intent_token",
  "next",
  "redirect_uri",
  "refresh_token",
  "session_token",
  "state",
  "token",
  "user_id",
  "userid",
]);
let analyticsBootstrapped = false;
let analyticsScriptLoaded = false;
let analyticsScriptScheduled = false;

export function isAnalyticsEnabled() {
  return (
    typeof window !== "undefined" &&
    import.meta.env.PROD &&
    import.meta.env.VITE_DISABLE_ANALYTICS !== "1" &&
    GA_MEASUREMENT_ID.length > 0
  );
}

function sanitizeAnalyticsParams(params?: AnalyticsParams): AnalyticsParams | undefined {
  if (!params) return undefined;

  const sanitized: AnalyticsParams = {};
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) continue;
    if (typeof value === "number" && !Number.isFinite(value)) continue;
    sanitized[key] =
      typeof value === "string" ? value.slice(0, MAX_PARAM_STRING_LENGTH) : value;
  }

  if (import.meta.env.VITE_GA_DEBUG_MODE === "1") {
    sanitized.debug_mode = true;
  }

  return Object.keys(sanitized).length > 0 ? sanitized : undefined;
}

export function sanitizeAnalyticsPath(pagePath: string): string {
  let parsed: URL;
  try {
    parsed = new URL(pagePath, "https://pandects.local");
  } catch {
    return "/";
  }

  const pathname = parsed.pathname || "/";
  if (SENSITIVE_ANALYTICS_PATHS.has(pathname)) {
    return pathname;
  }

  for (const key of Array.from(parsed.searchParams.keys())) {
    if (SENSITIVE_ANALYTICS_QUERY_KEYS.has(key.toLowerCase())) {
      parsed.searchParams.delete(key);
    }
  }

  const search = parsed.searchParams.toString();
  return `${pathname}${search ? `?${search}` : ""}`;
}

export function scheduleWhenBrowserIdle(
  callback: () => void,
  timeout = 1500,
) {
  if (typeof window === "undefined") {
    return () => undefined;
  }

  if ("requestIdleCallback" in window) {
    const idleId = window.requestIdleCallback(callback, { timeout });
    return () => {
      window.cancelIdleCallback(idleId);
    };
  }

  const timeoutId = globalThis.setTimeout(callback, Math.min(timeout, 400));
  return () => {
    globalThis.clearTimeout(timeoutId);
  };
}

export function bootstrapAnalytics() {
  if (!isAnalyticsEnabled() || analyticsBootstrapped) return;
  analyticsBootstrapped = true;

  window.dataLayer = window.dataLayer || [];
  window.gtag =
    window.gtag ||
    function gtag(...args: unknown[]) {
      window.dataLayer?.push(args);
    };
  window.gtag("js", new Date());
  window.gtag("config", GA_MEASUREMENT_ID, { send_page_view: false });
}

export function loadAnalyticsScript() {
  if (!isAnalyticsEnabled() || analyticsScriptLoaded) return;
  analyticsScriptLoaded = true;
  bootstrapAnalytics();

  const script = document.createElement("script");
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`;
  document.head.appendChild(script);
}

export function scheduleAnalyticsScriptLoad() {
  if (!isAnalyticsEnabled() || analyticsScriptLoaded || analyticsScriptScheduled) {
    return () => undefined;
  }

  analyticsScriptScheduled = true;
  bootstrapAnalytics();

  const listenerOptions: AddEventListenerOptions = {
    once: true,
    passive: true,
  };

  let timeoutId: number | null = window.setTimeout(() => {
    load();
  }, 3_000);

  const cleanup = () => {
    window.removeEventListener("pointerdown", load, listenerOptions);
    window.removeEventListener("keydown", load, listenerOptions);
    window.removeEventListener("touchstart", load, listenerOptions);
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId);
      timeoutId = null;
    }
  };

  const load = () => {
    cleanup();
    loadAnalyticsScript();
  };

  window.addEventListener("pointerdown", load, listenerOptions);
  window.addEventListener("keydown", load, listenerOptions);
  window.addEventListener("touchstart", load, listenerOptions);

  return cleanup;
}

export function trackEvent(eventName: string, params?: AnalyticsParams) {
  if (!isAnalyticsEnabled()) return;
  if (typeof window.gtag !== "function") return;
  window.gtag("event", eventName, sanitizeAnalyticsParams(params));
}

export function trackPageview(pagePath: string) {
  if (!isAnalyticsEnabled()) return;
  if (typeof window.gtag !== "function") return;
  const safePagePath = sanitizeAnalyticsPath(pagePath);
  window.gtag(
    "event",
    "page_view",
    sanitizeAnalyticsParams({
      page_path: safePagePath,
      page_location: `${window.location.origin}${safePagePath}`,
      page_title: document.title,
    }),
  );
}

export function trackTimeOnPage(pagePath: string, durationMs: number) {
  if (!isAnalyticsEnabled()) return;
  if (typeof window.gtag !== "function") return;
  if (durationMs <= 1000) return;
  const safePagePath = sanitizeAnalyticsPath(pagePath);
  window.gtag(
    "event",
    "time_on_page",
    sanitizeAnalyticsParams({
      page_path: safePagePath,
      page_location: `${window.location.origin}${safePagePath}`,
      page_title: document.title,
      duration_ms: Math.max(0, Math.round(durationMs)),
      transport_type: "beacon",
    }),
  );
}

export function installOutboundLinkTracking() {
  if (!isAnalyticsEnabled()) return () => undefined;

  const onClick = (event: MouseEvent) => {
    if (event.defaultPrevented) return;
    if (event.button !== 0) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;

    const target = event.target;
    if (!(target instanceof Element)) return;

    const anchor = target.closest("a");
    if (!anchor) return;

    const href = anchor.getAttribute("href");
    if (!href || href.startsWith("#")) return;
    if (href.startsWith("mailto:") || href.startsWith("tel:")) return;
    if (href.startsWith("javascript:")) return;

    let url: URL;
    try {
      url = new URL(href, window.location.href);
    } catch {
      return;
    }

    if (url.origin === window.location.origin) return;

    const linkText = anchor.textContent?.trim();

    trackEvent("outbound_link_click", {
      href: url.href,
      from_path: window.location.pathname,
      link_text: linkText ? linkText.slice(0, 120) : undefined,
      target: anchor.target || undefined,
      rel: anchor.rel || undefined,
    });
  };

  document.addEventListener("click", onClick, true);

  return () => {
    document.removeEventListener("click", onClick, true);
  };
}

export function installGlobalErrorTracking() {
  if (!isAnalyticsEnabled()) return () => undefined;

  const onError = (event: ErrorEvent) => {
    trackEvent("client_error", {
      kind: "error",
      message: event.message,
      filename: event.filename
        ? new URL(event.filename, window.location.href).pathname
        : "",
      lineno: event.lineno,
      colno: event.colno,
    });
  };

  const onUnhandledRejection = (event: PromiseRejectionEvent) => {
    const reason = event.reason;
    const message =
      reason instanceof Error
        ? reason.message
        : typeof reason === "string"
          ? reason
          : "Unhandled promise rejection";

    trackEvent("client_error", {
      kind: "unhandledrejection",
      message,
    });
  };

  window.addEventListener("error", onError);
  window.addEventListener("unhandledrejection", onUnhandledRejection);

  return () => {
    window.removeEventListener("error", onError);
    window.removeEventListener("unhandledrejection", onUnhandledRejection);
  };
}
