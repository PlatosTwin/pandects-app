type AnalyticsParams = Record<string, string | number | boolean | undefined>;

const GA_MEASUREMENT_ID = "G-94X4EVQVHZ";
let analyticsBootstrapped = false;
let analyticsScriptLoaded = false;

export function bootstrapAnalytics() {
  if (typeof window === "undefined" || analyticsBootstrapped) return;
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
  if (typeof window === "undefined" || analyticsScriptLoaded) return;
  analyticsScriptLoaded = true;
  bootstrapAnalytics();

  const script = document.createElement("script");
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`;
  document.head.appendChild(script);
}

export function trackEvent(eventName: string, params?: AnalyticsParams) {
  if (typeof window === "undefined") return;
  if (typeof window.gtag !== "function") return;
  window.gtag("event", eventName, params);
}

export function trackPageview(pagePath: string) {
  if (typeof window === "undefined") return;
  if (typeof window.gtag !== "function") return;
  window.gtag("event", "page_view", {
    page_path: pagePath,
    page_location: window.location.href,
    page_title: document.title,
  });
}

export function trackTimeOnPage(pagePath: string, durationMs: number) {
  if (typeof window === "undefined") return;
  if (typeof window.gtag !== "function") return;
  window.gtag("event", "time_on_page", {
    page_path: pagePath,
    page_location: window.location.href,
    page_title: document.title,
    duration_ms: Math.max(0, Math.round(durationMs)),
    transport_type: "beacon",
  });
}

export function installOutboundLinkTracking() {
  if (typeof window === "undefined") return () => undefined;

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
  if (typeof window === "undefined") return () => undefined;

  const onError = (event: ErrorEvent) => {
    trackEvent("client_error", {
      kind: "error",
      message: event.message,
      filename: event.filename,
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
