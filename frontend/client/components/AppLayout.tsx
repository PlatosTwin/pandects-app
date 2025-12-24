import { Outlet, useLocation } from "react-router-dom";
import { Suspense, useEffect, useRef } from "react";
import SiteBanner from "@/components/SiteBanner";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { applySeoForPath } from "@/lib/seo";
import {
  installOutboundLinkTracking,
  trackPageview,
  trackTimeOnPage,
} from "@/lib/analytics";

function RouteFallback() {
  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
      <div className="rounded-xl border border-border bg-background/70 p-6 backdrop-blur">
        <div className="h-4 w-36 animate-pulse rounded bg-muted" />
        <div className="mt-4 h-3 w-full animate-pulse rounded bg-muted" />
        <div className="mt-2 h-3 w-5/6 animate-pulse rounded bg-muted" />
      </div>
    </div>
  );
}

export function AppLayout() {
  const location = useLocation();
  const routeTimerRef = useRef<{ path: string; start: number } | null>(null);

  useEffect(() => {
    applySeoForPath(location.pathname);
    trackPageview(
      `${location.pathname}${location.search}${location.hash}`,
    );
  }, [location.hash, location.pathname, location.search]);

  useEffect(() => installOutboundLinkTracking(), []);

  useEffect(() => {
    const path = `${location.pathname}${location.search}${location.hash}`;

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
  }, [location.hash, location.pathname, location.search]);

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
      <SiteBanner />
      <Navigation />
      <main
        id="main-content"
        tabIndex={-1}
        className="flex-1 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
      >
        <Suspense fallback={<RouteFallback />}>
          <Outlet />
        </Suspense>
      </main>
      <Footer />
    </div>
  );
}
