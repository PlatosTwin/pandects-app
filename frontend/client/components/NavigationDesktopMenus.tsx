import { Link, useLocation } from "react-router-dom";
import { ChevronDown } from "lucide-react";
import { Suspense, lazy, memo, useMemo } from "react";
import { cn } from "@/lib/utils";
import { trackEvent } from "@/lib/analytics";
import brandLinks from "@branding/links.json";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const AuthMenu = lazy(() =>
  import("@/components/AuthMenu").then((mod) => ({ default: mod.AuthMenu })),
);

function AuthMenuFallback() {
  return (
    <div className="h-9 w-20 animate-pulse rounded-md bg-muted/70" aria-hidden="true" />
  );
}

function NavigationDesktopMenusComponent() {
  const location = useLocation();
  const docsUrl = import.meta.env.DEV ? "http://localhost:3001" : brandLinks.docsSiteUrl;
  const docsHomeUrl = `${docsUrl}/docs/guides/getting-started`;
  const isActive = (path: string) => location.pathname === path;
  const navLinkBase =
    "rounded-md px-3 py-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background";
  const dataSeparatorClass = "bg-border";

  const primaryLinks = useMemo(
    () => [
      { to: "/search", label: "Search", pandaTarget: "nav-search" },
      {
        to: docsHomeUrl,
        label: "Docs",
        pandaTarget: "nav-docs",
        external: true,
      },
    ],
    [docsHomeUrl],
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
      { type: "link", to: "/bulk-data", label: "Bulk Data", pandaTarget: "nav-bulk-data" },
      { type: "link", to: "/agreement-index", label: "Agreement Index" },
      { type: "link", to: "/sources-methods", label: "Sources & Methods" },
      { type: "link", to: "/xml-schema", label: "XML Schema" },
      { type: "link", to: "/taxonomy", label: "Taxonomy" },
      { type: "separator", key: "data-divider" },
      { type: "link", to: "/leaderboards", label: "Leaderboards" },
      { type: "link", to: "/trends-analyses", label: "Trends & Analyses" },
    ] as const,
    [],
  );
  const dataNavLinks = dataLinks.filter((item) => item.type === "link");
  const isDataActive = dataNavLinks.some((link) => isActive(link.to));
  const isAboutActive = aboutLinks.some((link) => isActive(link.to));

  return (
    <div className="hidden items-center gap-1 md:flex">
      <div className="flex items-center gap-1">
        {primaryLinks.map((link) =>
          link.external ? (
            <a
              key={link.to}
              href={link.to}
              target="_blank"
              rel="noopener noreferrer"
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
                  ? "border-l-2 border-primary bg-primary/10 font-medium text-primary"
                  : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
              )}
            >
              {link.label}
            </Link>
          ),
        )}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              data-panda-target="nav-bulk-data"
              className={cn(
                navLinkBase,
                "inline-flex items-center gap-1",
                isDataActive
                  ? "border-l-2 border-primary bg-primary/10 font-medium text-primary"
                  : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
              )}
            >
              Data
              <ChevronDown className="h-4 w-4 opacity-70" aria-hidden="true" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-56">
            {dataLinks.map((item) =>
              item.type === "separator" ? (
                <DropdownMenuSeparator key={item.key} className={dataSeparatorClass} />
              ) : (
                <DropdownMenuItem key={item.to} asChild>
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
                    className="text-foreground"
                  >
                    {item.label}
                  </Link>
                </DropdownMenuItem>
              ),
            )}
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
                  ? "border-l-2 border-primary bg-primary/10 font-medium text-primary"
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
        <Suspense fallback={<AuthMenuFallback />}>
          <AuthMenu />
        </Suspense>
      </div>
    </div>
  );
}

export default memo(NavigationDesktopMenusComponent);
