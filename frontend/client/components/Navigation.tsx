import { Link, useLocation } from "react-router-dom";
import { Suspense, lazy, memo, useEffect, useRef, useState } from "react";
import logo128Webp from "../../assets/logo-128.webp";
import logo256Webp from "../../assets/logo-256.webp";
import logo128Png from "../../assets/logo-128.png";
import logo256Png from "../../assets/logo-256.png";
import { LazyPandaEasterEgg } from "@/components/LazyPandaEasterEgg";
import { trackEvent } from "@/lib/analytics";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

const NavigationDesktopMenus = lazy(
  () => import("@/components/NavigationDesktopMenus"),
);
const NavigationMobileMenu = lazy(
  () => import("@/components/NavigationMobileMenu"),
);

function DesktopNavigationFallback() {
  return (
    <div className="hidden items-center gap-1 md:flex">
      <div className="flex items-center gap-1">
        <div className="h-9 w-20 animate-pulse rounded-md bg-muted/70" />
        <div className="h-9 w-16 animate-pulse rounded-md bg-muted/70" />
        <div className="h-9 w-16 animate-pulse rounded-md bg-muted/70" />
      </div>
      <div className="ml-2 h-9 w-20 animate-pulse rounded-md bg-muted/70" />
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
  const [isBetaDialogOpen, setIsBetaDialogOpen] = useState(false);
  const betaDisclaimer =
    "Pandects is in early development. Layout, API schema, and data organization may change.";

  useEffect(() => {
    setHasHydrated(true);
  }, []);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/70">
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
            <picture>
              <source
                srcSet={`${logo128Webp} 128w, ${logo256Webp} 256w`}
                sizes="36px"
                type="image/webp"
              />
              <img
                src={logo128Png}
                alt="Pandects Logo"
                data-panda-target="logo"
                width={36}
                height={36}
                srcSet={`${logo128Png} 128w, ${logo256Png} 256w`}
                sizes="36px"
                decoding="async"
                className="relative z-10 h-9 w-9 rounded-md object-cover ring-1 ring-border"
              />
            </picture>
            <span
              data-panda-target="brand"
              className="hidden text-base font-semibold tracking-tight text-foreground sm:block"
            >
              Pandects
            </span>
          </Link>
          <AlertDialog open={isBetaDialogOpen} onOpenChange={setIsBetaDialogOpen}>
            <button
              type="button"
              onClick={() => setIsBetaDialogOpen(true)}
              className="rounded-full bg-muted px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground/80 ring-1 ring-inset ring-border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
              aria-label="Read beta notice"
            >
              BETA
            </button>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Beta Notice</AlertDialogTitle>
                <AlertDialogDescription className="text-left">
                  {betaDisclaimer}
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogAction onClick={() => setIsBetaDialogOpen(false)}>
                Got it
              </AlertDialogAction>
            </AlertDialogContent>
          </AlertDialog>
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
