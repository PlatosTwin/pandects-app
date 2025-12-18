import { useEffect, useRef, useState } from "react";

type TurnstileStatus = "idle" | "loading" | "ready" | "error";

declare global {
  interface Window {
    turnstile?: {
      render: (
        container: string | HTMLElement,
        params: {
          sitekey: string;
          theme?: "light" | "dark" | "auto";
          callback?: (token: string) => void;
          "expired-callback"?: () => void;
          "error-callback"?: () => void;
        },
      ) => string;
      remove: (widgetId: string) => void;
      reset: (widgetId: string) => void;
    };
  }
}

function ensureTurnstileScript(): Promise<void> {
  if (typeof window === "undefined") return Promise.resolve();
  if (typeof window.turnstile?.render === "function") return Promise.resolve();

  const existing = document.querySelector<HTMLScriptElement>(
    'script[data-turnstile="true"]',
  );
  if (existing) {
    return new Promise((resolve, reject) => {
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener("error", () => reject(new Error("Turnstile load failed")), {
        once: true,
      });
    });
  }

  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
    script.async = true;
    script.defer = true;
    script.dataset.turnstile = "true";
    script.addEventListener("load", () => resolve(), { once: true });
    script.addEventListener(
      "error",
      () => reject(new Error("Turnstile load failed")),
      { once: true },
    );
    document.head.appendChild(script);
  });
}

export function TurnstileWidget({
  siteKey,
  onToken,
  onError,
}: {
  siteKey: string;
  onToken: (token: string | null) => void;
  onError: (message: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const widgetIdRef = useRef<string | null>(null);
  const onTokenRef = useRef(onToken);
  const onErrorRef = useRef(onError);
  const [status, setStatus] = useState<TurnstileStatus>("idle");

  useEffect(() => {
    onTokenRef.current = onToken;
    onErrorRef.current = onError;
  }, [onError, onToken]);

  useEffect(() => {
    if (!siteKey) return;
    setStatus("loading");
    onTokenRef.current(null);

    let canceled = false;
    void ensureTurnstileScript()
      .then(() => {
        if (canceled) return;
        if (!containerRef.current) throw new Error("Missing captcha container.");
        const api = window.turnstile;
        if (!api?.render) throw new Error("Captcha is unavailable.");
        if (widgetIdRef.current) api.remove(widgetIdRef.current);

        widgetIdRef.current = api.render(containerRef.current, {
          sitekey: siteKey,
          theme: "auto",
          callback: (token) => {
            onTokenRef.current(token);
          },
          "expired-callback": () => {
            onTokenRef.current(null);
          },
          "error-callback": () => {
            onTokenRef.current(null);
            onErrorRef.current("Captcha failed to load. Please retry.");
          },
        });
        setStatus("ready");
      })
      .catch((err) => {
        if (canceled) return;
        setStatus("error");
        onTokenRef.current(null);
        onErrorRef.current(String(err));
      });

    return () => {
      canceled = true;
      const api = window.turnstile;
      const widgetId = widgetIdRef.current;
      if (api?.remove && widgetId) api.remove(widgetId);
      widgetIdRef.current = null;
    };
  }, [siteKey]);

  return (
    <div className="grid gap-2">
      <div ref={containerRef} />
      {status === "loading" ? (
        <div className="text-xs text-muted-foreground">Loading captcha…</div>
      ) : status === "error" ? (
        <div className="text-xs text-destructive">Captcha unavailable.</div>
      ) : null}
    </div>
  );
}
