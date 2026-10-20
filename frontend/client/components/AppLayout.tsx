import { Outlet, useLocation } from "react-router-dom";
import { Suspense, useEffect, useRef } from "react";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { ErrorBoundary } from "@/components/ErrorBoundary";
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

// Neutral route-level placeholder. It uses the header's frame (max-w-7xl and
// the same gutters) so the title block lines up with the logo, and plain
// blocks with no card chrome or blur so no page appears to "jump" when it
// replaces the fallback.
function RouteFallback() {
  return (
    <div
      className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8"
      role="status"
      aria-live="polite"
    >
      <span className="sr-only">Loading page</span>
      <div aria-hidden="true">
        <div className="space-y-3">
          <div className="h-8 w-48 animate-pulse rounded-md bg-muted sm:h-9" />
          <div className="h-4 w-full max-w-xl animate-pulse rounded bg-muted/70" />
        </div>
        <div className="mt-8 space-y-3">
          <div className="h-4 w-full animate-pulse rounded bg-muted/70" />
          <div className="h-4 w-11/12 animate-pulse rounded bg-muted/70" />
          <div className="h-4 w-4/5 animate-pulse rounded bg-muted/70" />
        </div>
        <div className="mt-8 h-64 animate-pulse rounded-lg bg-muted/50" />
      </div>
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
        className="flex-1 outline-none"
      >
        <ErrorBoundary
          key={location.pathname}
          scope={`route:${location.pathname}`}
        >
          <Suspense fallback={<RouteFallback />}>
            <Outlet />
          </Suspense>
        </ErrorBoundary>
      </main>
      <Footer />
    </div>
  );
}
