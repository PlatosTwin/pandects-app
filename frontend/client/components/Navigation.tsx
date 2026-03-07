import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { ChevronDown, Menu } from "lucide-react";
import { useMemo, useRef, useState, memo } from "react";
import logo128Webp from "../../assets/logo-128.webp";
import logo256Webp from "../../assets/logo-256.webp";
import logo128Png from "../../assets/logo-128.png";
import logo256Png from "../../assets/logo-256.png";
import { Button } from "@/components/ui/button";
import { LazyPandaEasterEgg } from "@/components/LazyPandaEasterEgg";
import { trackEvent } from "@/lib/analytics";
import { AuthMenu } from "@/components/AuthMenu";
import brandLinks from "@branding/links.json";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetDescription,
  SheetTitle,
  SheetTrigger,
  SheetClose,
} from "@/components/ui/sheet";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

function NavigationComponent() {
  const location = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isBetaDialogOpen, setIsBetaDialogOpen] = useState(false);
  const navRef = useRef<HTMLElement | null>(null);
  const docsUrl = import.meta.env.DEV ? "http://localhost:3001" : brandLinks.docsSiteUrl;
  const betaDisclaimer = `Pandects is in early development. Layout, API schema, and data organization may change.`;

  const isActive = (path: string) => location.pathname === path;
  const navLinkBase =
    "rounded-md px-3 py-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background";

  const primaryLinks = useMemo(
    () => [
      { to: "/sections", label: "Search", pandaTarget: "nav-search" },
      {
        to: docsUrl,
        label: "Docs",
        pandaTarget: "nav-docs",
        external: true,
      },
    ],
    [],
  );
  const aboutLinks = useMemo(
    () => [
      { to: "/about", label: "About", pandaTarget: "nav-about" },
      { to: "/feedback", label: "Feedback" },
      { to: "/contribute", label: "Contribute", pandaTarget: "nav-contribute" },
    ],
    [],
  );
  const dataLinks = useMemo(
    () => [
      { to: "/bulk-data", label: "Bulk Data", pandaTarget: "nav-bulk-data" },
      { to: "/agreement-index", label: "Agreement Index" },
      { to: "/sources-methods", label: "Sources & Methods" },
      { to: "/xml-schema", label: "XML Schema" },
      { to: "/taxonomy", label: "Taxonomy" },
    ],
    [],
  );
  const isDataActive = dataLinks.some((link) => isActive(link.to));
  const isAboutActive = aboutLinks.some((link) => isActive(link.to));

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
                className="relative z-10 h-9 w-9 rounded-md object-cover ring-1 ring-border/60"
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
              className="rounded-full bg-yellow-100 px-2 py-0.5 text-xs font-semibold text-yellow-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
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

        {/* Desktop navigation */}
        <div className="hidden items-center gap-1 md:flex">
          <div className="flex items-center gap-1">
            {primaryLinks.map((link) => (
              link.external ? (
                <a
                  key={link.to}
                  href={link.to}
                  data-panda-target={link.pandaTarget}
                  onClick={() =>
                    trackEvent("nav_primary_click", {
                      nav_item: link.label,
                      from_path: location.pathname,
                      to_path: link.to,
                    })
                  }
                  className={cn(
                    navLinkBase,
                    "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                  )}
                >
                  {link.label}
                </a>
              ) : (
                <Link
                  key={link.to}
                  to={link.to}
                  data-panda-target={link.pandaTarget}
                  onClick={() =>
                    trackEvent("nav_primary_click", {
                      nav_item: link.label,
                      from_path: location.pathname,
                      to_path: link.to,
                    })
                  }
                  aria-current={isActive(link.to) ? "page" : undefined}
                  className={cn(
                    navLinkBase,
                    isActive(link.to)
                      ? "bg-primary/10 text-primary font-medium border-l-2 border-primary"
                      : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                  )}
                >
                  {link.label}
                </Link>
              )
            ))}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  data-panda-target="nav-bulk-data"
                  className={cn(
                    navLinkBase,
                    "inline-flex items-center gap-1",
                    isDataActive
                      ? "bg-primary/10 text-primary font-medium border-l-2 border-primary"
                      : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                  )}
                >
                  Data
                  <ChevronDown className="h-4 w-4 opacity-70" aria-hidden="true" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-56">
                {dataLinks.map((link) => (
                  <DropdownMenuItem key={link.to} asChild>
                    <Link
                      to={link.to}
                      onClick={() =>
                        trackEvent("nav_data_click", {
                          nav_item: link.label,
                          from_path: location.pathname,
                          to_path: link.to,
                        })
                      }
                      aria-current={isActive(link.to) ? "page" : undefined}
                      className="text-foreground"
                    >
                      {link.label}
                    </Link>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  data-panda-target="nav-about"
                  className={cn(
                    navLinkBase,
                    "inline-flex items-center gap-1",
                    isAboutActive
                      ? "bg-primary/10 text-primary font-medium border-l-2 border-primary"
                      : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                  )}
                >
                  Project
                  <ChevronDown className="h-4 w-4 opacity-70" aria-hidden="true" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-56">
                {aboutLinks.map((link) => (
                  <DropdownMenuItem key={link.to} asChild>
                    <Link
                      to={link.to}
                      data-panda-target={link.pandaTarget}
                      onClick={() =>
                        trackEvent("nav_secondary_click", {
                          nav_item: link.label,
                          from_path: location.pathname,
                          to_path: link.to,
                        })
                      }
                      aria-current={isActive(link.to) ? "page" : undefined}
                      className="text-foreground"
                    >
                      {link.label}
                    </Link>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          <div className="ml-2">
            <AuthMenu />
          </div>
        </div>

        {/* Mobile navigation */}
        <div className="flex items-center md:hidden">
          <div className="mr-2">
            <AuthMenu />
          </div>
          <Sheet open={isMobileMenuOpen} onOpenChange={setIsMobileMenuOpen}>
            <SheetTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-10 w-10"
                aria-label="Open menu"
              >
                <Menu className="h-5 w-5" aria-hidden="true" />
              </Button>
            </SheetTrigger>
            <SheetContent
              side="right"
              className="w-[min(320px,100vw)] max-w-full p-0"
            >
              <div className="flex h-full flex-col">
                <SheetHeader className="border-b p-4 text-left">
                  <SheetTitle className="flex items-center gap-3">
                    <picture>
                      <source
                        srcSet={`${logo128Webp} 128w, ${logo256Webp} 256w`}
                        sizes="36px"
                        type="image/webp"
                      />
                      <img
                        src={logo128Png}
                        alt="Pandects Logo"
                        width={36}
                        height={36}
                        srcSet={`${logo128Png} 128w, ${logo256Png} 256w`}
                        sizes="36px"
                        decoding="async"
                        className="h-9 w-9 rounded-md object-cover ring-1 ring-border/60"
                      />
                    </picture>
                    <span className="text-base font-semibold tracking-tight">
                      Pandects
                    </span>
                  </SheetTitle>
                  <SheetDescription className="sr-only">
                    Primary navigation links and account access.
                  </SheetDescription>
                </SheetHeader>

                <div className="flex-1 overflow-auto p-2">
                  <div className="grid gap-1">
                    {primaryLinks.map((link) => (
                      <SheetClose asChild key={link.to}>
                        {link.external ? (
                          <a
                            href={link.to}
                            onClick={() =>
                              trackEvent("nav_primary_click", {
                                nav_item: link.label,
                                from_path: location.pathname,
                                to_path: link.to,
                              })
                            }
                            className={cn(
                              navLinkBase,
                              "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                            )}
                          >
                            {link.label}
                          </a>
                        ) : (
                          <Link
                            to={link.to}
                            onClick={() =>
                              trackEvent("nav_primary_click", {
                                nav_item: link.label,
                                from_path: location.pathname,
                                to_path: link.to,
                              })
                            }
                            aria-current={isActive(link.to) ? "page" : undefined}
                            className={cn(
                              navLinkBase,
                              isActive(link.to)
                                ? "bg-primary/10 text-primary font-medium border-l-2 border-primary"
                                : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                            )}
                          >
                            {link.label}
                          </Link>
                        )}
                      </SheetClose>
                    ))}
                  </div>

                  <div className="mt-3 rounded-lg border border-border/60 bg-muted/20 p-2">
                    <div className="px-2 pb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      Data
                    </div>
                    <div className="grid gap-1">
                      {dataLinks.map((link) => (
                        <SheetClose asChild key={link.to}>
                          <Link
                            to={link.to}
                            onClick={() =>
                              trackEvent("nav_data_click", {
                                nav_item: link.label,
                                from_path: location.pathname,
                                to_path: link.to,
                              })
                            }
                            aria-current={isActive(link.to) ? "page" : undefined}
                            className={cn(
                              navLinkBase,
                              "pl-4",
                              isActive(link.to)
                                ? "bg-primary/10 text-primary font-medium border-l-2 border-primary"
                                : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                            )}
                          >
                            {link.label}
                          </Link>
                        </SheetClose>
                      ))}
                    </div>
                  </div>

                  <div className="mt-3 rounded-lg border border-border/60 bg-muted/20 p-2">
                    <div className="px-2 pb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      Project
                    </div>
                    <div className="grid gap-1">
                      {aboutLinks.map((link) => (
                      <SheetClose asChild key={link.to}>
                        <Link
                          to={link.to}
                          data-panda-target={link.pandaTarget}
                          onClick={() =>
                            trackEvent("nav_secondary_click", {
                              nav_item: link.label,
                              from_path: location.pathname,
                              to_path: link.to,
                            })
                          }
                          aria-current={isActive(link.to) ? "page" : undefined}
                          className={cn(
                            navLinkBase,
                            "pl-4",
                            isActive(link.to)
                              ? "bg-primary/10 text-primary font-medium border-l-2 border-primary"
                              : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                          )}
                        >
                          {link.label}
                        </Link>
                      </SheetClose>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>

        <LazyPandaEasterEgg containerRef={navRef} />
      </nav>
    </header>
  );
}

export default memo(NavigationComponent);
