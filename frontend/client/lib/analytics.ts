type AnalyticsParams = Record<string, string | number | boolean | undefined>;

export function trackEvent(eventName: string, params?: AnalyticsParams) {
  if (typeof window === "undefined") return;
  if (typeof window.gtag !== "function") return;
  window.gtag("event", eventName, params);
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
