import { Link, useLocation } from "react-router-dom";
import { Menu } from "lucide-react";
import { Suspense, lazy, memo, useMemo, useState } from "react";
import logo128Webp from "../../assets/logo-128.webp";
import logo256Webp from "../../assets/logo-256.webp";
import logo128Png from "../../assets/logo-128.png";
import logo256Png from "../../assets/logo-256.png";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { trackEvent } from "@/lib/analytics";
import brandLinks from "@branding/links.json";
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";

const AuthMenu = lazy(() =>
  import("@/components/AuthMenu").then((mod) => ({ default: mod.AuthMenu })),
);

function AuthMenuFallback() {
  return (
    <div className="h-9 w-20 animate-pulse rounded-md bg-muted/70" aria-hidden="true" />
  );
}

function NavigationMobileMenuComponent() {
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);
  const docsUrl = import.meta.env.DEV ? "http://localhost:3001" : brandLinks.docsSiteUrl;
  const docsHomeUrl = `${docsUrl}/docs/guides/getting-started`;
  const isActive = (path: string) => location.pathname === path;
  const navLinkBase =
    "rounded-md px-3 py-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background";
  const dataSeparatorClass = "bg-border";

  const primaryLinks = useMemo(
    () => [
      { to: "/search", label: "Search" },
      { to: docsHomeUrl, label: "Docs", external: true },
    ],
    [docsHomeUrl],
  );
  const aboutLinks = useMemo(
    () => [
      { to: "/about", label: "About" },
      { to: "/feedback", label: "Feedback" },
      { to: "/contribute", label: "Contribute" },
    ],
    [],
  );
  const dataLinks = useMemo(
    () => [
      { type: "link", to: "/bulk-data", label: "Bulk Data" },
      { type: "link", to: "/agreement-index", label: "Agreement Index" },
      { type: "link", to: "/sources-methods", label: "Sources & Methods" },
      { type: "link", to: "/xml-schema", label: "XML Schema" },
      { type: "link", to: "/taxonomy", label: "Taxonomy" },
      { type: "separator", key: "data-divider-1" },
      { type: "link", to: "/examples", label: "Examples" },
      { type: "separator", key: "data-divider-2" },
      { type: "link", to: "/leaderboards", label: "Leaderboards" },
      { type: "link", to: "/trends-analyses", label: "Trends & Analyses" },
    ] as const,
    [],
  );

  return (
    <div className="flex items-center md:hidden">
      <div className="mr-2">
        <Suspense fallback={<AuthMenuFallback />}>
          <AuthMenu />
        </Suspense>
      </div>
      <Sheet open={isOpen} onOpenChange={setIsOpen}>
        <SheetTrigger asChild>
          <Button variant="ghost" size="icon" className="h-10 w-10" aria-label="Open menu">
            <Menu className="h-5 w-5" aria-hidden="true" />
          </Button>
        </SheetTrigger>
        <SheetContent side="right" className="w-[min(320px,100vw)] max-w-full p-0">
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
                    className="h-9 w-9 rounded-md object-cover ring-1 ring-border"
                  />
                </picture>
                <span className="text-base font-semibold tracking-tight">Pandects</span>
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
                        target="_blank"
                        rel="noopener noreferrer"
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
                            ? "border-l-2 border-primary bg-primary/10 font-medium text-primary"
                            : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                        )}
                      >
                        {link.label}
                      </Link>
                    )}
                  </SheetClose>
                ))}
              </div>

              <div className="mt-3 rounded-lg border border-border bg-muted/20 p-2">
                <div className="px-2 pb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Data
                </div>
                <div className="grid gap-1">
                  {dataLinks.map((item) =>
                    item.type === "separator" ? (
                      <div
                        key={item.key}
                        aria-hidden="true"
                        className={cn("mx-2 my-1 h-px", dataSeparatorClass)}
                      />
                    ) : (
                      <SheetClose asChild key={item.to}>
                        <Link
                          to={item.to}
                          onClick={() =>
                            trackEvent("nav_data_click", {
                              nav_item: item.label,
                              from_path: location.pathname,
                              to_path: item.to,
                            })
                          }
                          aria-current={isActive(item.to) ? "page" : undefined}
                          className={cn(
                            navLinkBase,
                            "pl-4",
                            isActive(item.to)
                              ? "border-l-2 border-primary bg-primary/10 font-medium text-primary"
                              : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                          )}
                        >
                          {item.label}
                        </Link>
                      </SheetClose>
                    ),
                  )}
                </div>
              </div>

              <div className="mt-3 rounded-lg border border-border bg-muted/20 p-2">
                <div className="px-2 pb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Project
                </div>
                <div className="grid gap-1">
                  {aboutLinks.map((link) => (
                    <SheetClose asChild key={link.to}>
                      <Link
                        to={link.to}
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
                            ? "border-l-2 border-primary bg-primary/10 font-medium text-primary"
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
  );
}

export default memo(NavigationMobileMenuComponent);
