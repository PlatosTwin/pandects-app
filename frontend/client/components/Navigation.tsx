import { Link, useLocation } from "react-router-dom";
import { Suspense, lazy, memo, useEffect, useRef, useState } from "react";
import { LazyPandaEasterEgg } from "@/components/LazyPandaEasterEgg";
import { NavBetaBadge } from "@/components/NavBetaBadge";
import { NavLogo } from "@/components/NavLogo";
import { trackEvent } from "@/lib/analytics";

const NavigationDesktopMenus = lazy(
  () => import("@/components/NavigationDesktopMenus"),
);
const NavigationMobileMenu = lazy(
  () => import("@/components/NavigationMobileMenu"),
);

function DesktopNavigationFallback() {
  return (
    <div className="hidden flex-1 items-center justify-between md:flex" aria-hidden="true">
      <div className="ml-3 flex items-center gap-1 lg:ml-6">
        <div className="h-9 w-16 animate-pulse rounded-md bg-muted/70" />
        <div className="h-9 w-20 animate-pulse rounded-md bg-muted/70" />
        <div className="h-9 w-20 animate-pulse rounded-md bg-muted/70" />
        <div className="h-9 w-16 animate-pulse rounded-md bg-muted/70" />
        <div className="h-9 w-16 animate-pulse rounded-md bg-muted/70" />
      </div>
      <div className="flex items-center gap-1">
        <div className="h-9 w-9 animate-pulse rounded-md bg-muted/70" />
        <div className="h-9 w-20 animate-pulse rounded-md bg-muted/70" />
      </div>
    </div>
  );
}

function MobileNavigationFallback() {
  return <div className="h-10 w-10 rounded-md bg-muted/70 md:hidden" aria-hidden="true" />;
}

function NavigationComponent() {
  const location = useLocation();
  const navRef = useRef<HTMLElement | null>(null);
  const [hasHydrated, setHasHydrated] = useState(false);

  useEffect(() => {
    setHasHydrated(true);
  }, []);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/90">
      <nav
        ref={navRef}
        aria-label="Primary"
        className="relative z-0 mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8"
      >
        {/* Brand */}
        <div className="flex items-center gap-2">
          <Link
            to="/"
            className="flex items-center gap-3 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
            onClick={() =>
              trackEvent("logo_click", {
                from_path: location.pathname,
                to_path: "/",
              })
            }
          >
            <NavLogo pandaTarget="logo" />
            <span
              data-panda-target="brand"
              className="text-base font-semibold tracking-tight text-foreground"
            >
              Pandects
            </span>
          </Link>
          {/* On phones the badge lives in the menu sheet header instead. */}
          <NavBetaBadge className="hidden md:inline-block" />
        </div>

        {hasHydrated ? (
          <>
            <Suspense fallback={<DesktopNavigationFallback />}>
              <NavigationDesktopMenus />
            </Suspense>
            <Suspense fallback={<MobileNavigationFallback />}>
              <NavigationMobileMenu />
            </Suspense>
          </>
        ) : (
          <>
            <DesktopNavigationFallback />
            <MobileNavigationFallback />
          </>
        )}

        <LazyPandaEasterEgg containerRef={navRef} />
      </nav>
    </header>
  );
}

export default memo(NavigationComponent);
