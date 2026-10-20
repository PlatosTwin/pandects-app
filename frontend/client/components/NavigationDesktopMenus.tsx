import { useLocation } from "react-router-dom";
import { ChevronDown } from "lucide-react";
import { Suspense, lazy, memo } from "react";
import { cn } from "@/lib/utils";
import { trackEvent } from "@/lib/analytics";
import {
  DOCS_LINK,
  NAV_GROUPS,
  SEARCH_LINK,
  isGroupActive,
  linkKey,
  type NavLinkItem,
} from "@/components/navigation-links";
import { NavItemLink, useShowLock } from "@/components/NavLinkParts";
import { ThemeToggle } from "@/components/ThemeToggle";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
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

// Horizontal bar: the active item gets a 2px primary indicator flush with the
// header's bottom edge ((h-16 bar - h-9 item) / 2 = 0.875rem below the item).
const topLevelBase =
  "relative inline-flex h-9 items-center gap-1 rounded-md px-2.5 text-sm font-medium transition-colors after:pointer-events-none after:absolute after:inset-x-2.5 lg:px-3 lg:after:inset-x-3 after:-bottom-[0.875rem] after:h-0.5 after:rounded-full after:content-[''] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background";
const topLevelActive = "text-foreground after:bg-primary";
const topLevelIdle =
  "text-muted-foreground after:bg-transparent hover:bg-accent/60 hover:text-foreground";

function NavigationDesktopMenusComponent() {
  const location = useLocation();
  const showLock = useShowLock();
  const isActive = (link: NavLinkItem) =>
    link.kind === "internal" && location.pathname === link.to;

  const track = (event: string, link: NavLinkItem) => () =>
    trackEvent(event, {
      nav_item: link.label,
      from_path: location.pathname,
      to_path: linkKey(link),
    });

  const renderTopLink = (link: NavLinkItem) => (
    <NavItemLink
      key={linkKey(link)}
      link={link}
      active={isActive(link)}
      locked={showLock(link)}
      onClick={track("nav_primary_click", link)}
      className={cn(topLevelBase, isActive(link) ? topLevelActive : topLevelIdle)}
    />
  );

  return (
    <div className="ml-3 hidden flex-1 items-center justify-between gap-2 md:flex lg:ml-6">
      <div className="flex items-center gap-0.5 lg:gap-1">
        {renderTopLink(SEARCH_LINK)}
        {NAV_GROUPS.map((group) => {
          const groupActive = isGroupActive(group, location.pathname);
          return (
            <DropdownMenu key={group.id} modal={false}>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  data-panda-target={group.pandaTarget}
                  data-active={groupActive ? "true" : undefined}
                  className={cn(topLevelBase, groupActive ? topLevelActive : topLevelIdle)}
                >
                  {group.label}
                  <ChevronDown className="h-4 w-4 opacity-70" aria-hidden="true" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-56">
                {group.links.map((link) => (
                  <DropdownMenuItem key={linkKey(link)} asChild>
                    <NavItemLink
                      link={link}
                      active={isActive(link)}
                      locked={showLock(link)}
                      onClick={track("nav_data_click", link)}
                      className={cn(
                        "flex items-center gap-2",
                        isActive(link)
                          ? "bg-primary/10 font-medium text-primary"
                          : "text-foreground",
                      )}
                    />
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          );
        })}
        {renderTopLink(DOCS_LINK)}
      </div>

      <div className="flex items-center gap-1">
        <ThemeToggle />
        <Suspense fallback={<AuthMenuFallback />}>
          <AuthMenu />
        </Suspense>
      </div>
    </div>
  );
}

export default memo(NavigationDesktopMenusComponent);
