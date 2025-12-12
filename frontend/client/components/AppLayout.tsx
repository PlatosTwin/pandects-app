import { Outlet } from "react-router-dom";
import { Suspense } from "react";
import SiteBanner from "@/components/SiteBanner";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";

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
  return (
    <div className="min-h-screen bg-cream flex flex-col">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 z-[60] rounded-md bg-background px-3 py-2 text-sm font-medium text-foreground shadow"
      >
        Skip to content
      </a>
      <SiteBanner />
      <Navigation />
      <main id="main-content" className="flex-1">
        <Suspense fallback={<RouteFallback />}>
          <Outlet />
        </Suspense>
      </main>
      <Footer />
    </div>
  );
}

