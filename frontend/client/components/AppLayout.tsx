import { Outlet, matchPath, useLocation } from "react-router-dom";
import { Suspense, useEffect, useRef } from "react";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { STATUS_FRAME } from "@/components/StatusPage";
import { ROUTES, type RouteShell } from "@/lib/routes";
import { applySeoForLocation } from "@/lib/seo";
import {
  installOutboundLinkTracking,
  sanitizeAnalyticsPath,
  scheduleWhenBrowserIdle,
  trackPageview,
  trackTimeOnPage,
} from "@/lib/analytics";
import {
  captureAttributionOnce,
  recordFirstPartyPageView,
} from "@/lib/first-party-telemetry";

// Neutral route-level placeholder: plain blocks with no card chrome or blur,
// laid out in the same frame as the page it stands in for (see `shell` in
// routes.tsx), so nothing moves sideways when the page replaces it.
const FALLBACK_FRAME = "app-frame py-8";

const fallbackBar = "animate-pulse rounded bg-muted/70";

function FallbackTitle({ description = true }: { description?: boolean }) {
  return (
    <div className="space-y-3">
      <div className="h-8 w-48 animate-pulse rounded-md bg-muted sm:h-9" />
      {description ? (
        <div className={`h-4 w-full max-w-xl ${fallbackBar}`} />
      ) : null}
    </div>
  );
}

function FallbackLines() {
  return (
    <div className="mt-8 space-y-3">
      <div className={`h-4 w-full ${fallbackBar}`} />
      <div className={`h-4 w-11/12 ${fallbackBar}`} />
      <div className={`h-4 w-4/5 ${fallbackBar}`} />
    </div>
  );
}

function FallbackBody({ shell }: { shell: Exclude<RouteShell, "none"> }) {
  switch (shell) {
    case "landing":
      return (
        <div className="px-4 py-8 sm:py-12 lg:py-16">
          <div className="mx-auto h-[36rem] w-full max-w-[960px] animate-pulse rounded-xl bg-muted/40" />
        </div>
      );
    case "status":
      return (
        <div className={STATUS_FRAME}>
          <div className="mx-auto flex w-full max-w-xl flex-col items-center gap-4">
            <div className={`h-4 w-10 ${fallbackBar}`} />
            <div className="h-8 w-64 animate-pulse rounded-md bg-muted sm:h-9" />
            <div className="mt-4 flex gap-3">
              <div className="h-10 w-32 animate-pulse rounded-md bg-muted/70" />
              <div className="h-10 w-32 animate-pulse rounded-md bg-muted/50" />
            </div>
          </div>
        </div>
      );
    case "form":
    case "settings":
      return (
        <div className={FALLBACK_FRAME}>
          <div
            className={`mx-auto ${shell === "form" ? "max-w-md" : "max-w-4xl"}`}
          >
            <FallbackTitle />
            <div className="mt-8 h-80 animate-pulse rounded-lg bg-muted/50" />
          </div>
        </div>
      );
    case "docs":
      return (
        <div className={FALLBACK_FRAME}>
          <div className="mx-auto max-w-[720px]">
            <FallbackTitle description={false} />
            <FallbackLines />
            <div className="mt-8 h-64 animate-pulse rounded-lg bg-muted/50" />
          </div>
        </div>
      );
    case "docs-toc":
      return (
        <div className={FALLBACK_FRAME}>
          <div className="mx-auto max-w-[1008px]">
            <FallbackTitle description={false} />
            <div className="mt-8 lg:grid lg:grid-cols-[240px_minmax(0,720px)] lg:gap-12">
              <div className="mb-6 h-11 animate-pulse rounded-lg bg-muted/50 lg:mb-0 lg:h-56" />
              <div>
                <FallbackLines />
                <div className="mt-8 h-64 animate-pulse rounded-lg bg-muted/50" />
              </div>
            </div>
          </div>
        </div>
      );
    default:
      return (
        <div className={FALLBACK_FRAME}>
          <FallbackTitle />
          <FallbackLines />
          <div className="mt-8 h-64 animate-pulse rounded-lg bg-muted/50" />
        </div>
      );
  }
}

function routeShellFor(pathname: string): RouteShell {
  const match = ROUTES.find(
    (route) => !route.catchAll && matchPath(route.path, pathname),
  );
  if (match) return match.shell ?? "data";
  return ROUTES.find((route) => route.catchAll)?.shell ?? "data";
}

function RouteFallback({ pathname }: { pathname: string }) {
  const shell = routeShellFor(pathname);
  return (
    <div role="status" aria-live="polite">
      <span className="sr-only">Loading page</span>
      {shell === "none" ? null : (
        <div aria-hidden="true">
          <FallbackBody shell={shell} />
        </div>
      )}
    </div>
  );
}

export function AppLayout() {
  const location = useLocation();
  const routeTimerRef = useRef<{ path: string; start: number } | null>(null);

  useEffect(() => {
    applySeoForLocation(location.pathname, location.search);
    trackPageview(`${location.pathname}${location.search}`);
    recordFirstPartyPageView(`${location.pathname}${location.search}`);
  }, [location.pathname, location.search]);

  useEffect(() => {
    captureAttributionOnce();
  }, []);

  useEffect(() => {
    let cleanup = () => undefined;
    const cancelScheduledInstall = scheduleWhenBrowserIdle(() => {
      cleanup = installOutboundLinkTracking();
    });

    return () => {
      cancelScheduledInstall();
      cleanup();
    };
  }, []);

  useEffect(() => {
    const path = sanitizeAnalyticsPath(`${location.pathname}${location.search}`);

    if (routeTimerRef.current) {
      trackTimeOnPage(
        routeTimerRef.current.path,
        performance.now() - routeTimerRef.current.start,
      );
    }

    routeTimerRef.current = {
      path,
      start: performance.now(),
    };
  }, [location.pathname, location.search]);

  useEffect(() => {
    const handlePageHide = () => {
      if (!routeTimerRef.current) return;

      trackTimeOnPage(
        routeTimerRef.current.path,
        performance.now() - routeTimerRef.current.start,
      );
    };

    window.addEventListener("pagehide", handlePageHide);
    return () => {
      window.removeEventListener("pagehide", handlePageHide);
    };
  }, []);

  return (
    <div className="min-h-screen bg-cream flex flex-col">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 z-[60] rounded-md bg-background px-3 py-2 text-sm font-medium text-foreground shadow focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
      >
        Skip to content
      </a>
      <Navigation />
      <main
        id="main-content"
        role="main"
        tabIndex={-1}
        // At least a viewport tall (minus the h-16 nav), so the footer starts
        // below the fold whether the route fallback or the page is showing and
        // never jumps when a lazy page arrives (it was most of each route's CLS).
        className="min-h-[calc(100svh-4rem)] flex-1 outline-none print:min-h-0"
      >
        <ErrorBoundary
          key={location.pathname}
          scope={`route:${location.pathname}`}
        >
          <Suspense fallback={<RouteFallback pathname={location.pathname} />}>
            <Outlet />
          </Suspense>
        </ErrorBoundary>
      </main>
      <Footer />
    </div>
  );
}
