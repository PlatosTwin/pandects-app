import { Link, useLocation } from "react-router-dom";
import { Menu } from "lucide-react";
import { memo, useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { trackEvent } from "@/lib/analytics";
import {
  DOCS_LINK,
  LEGAL_LINKS,
  NAV_GROUPS,
  PROJECT_LINKS,
  SEARCH_LINK,
  linkKey,
  type InternalNavLink,
  type NavLinkItem,
} from "@/components/navigation-links";
import {
  AvatarInitial,
  NavItemLink,
  useNavAuth,
  useShowLock,
} from "@/components/NavLinkParts";
import { NavBetaBadge } from "@/components/NavBetaBadge";
import { NavLogo } from "@/components/NavLogo";
import { ThemeToggleInline } from "@/components/ThemeToggle";
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";

// Every sheet item carries the same 2px left border so labels line up; the
// active item colors it.
const itemBase =
  "flex items-center gap-2 rounded-md border-l-2 px-3 py-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background";
const itemActive = "border-primary bg-primary/10 text-primary";
const itemIdle =
  "border-transparent text-muted-foreground hover:bg-accent/60 hover:text-foreground";

function SheetSection({ heading, children }: { heading: string; children: ReactNode }) {
  return (
    <div className="mt-4">
      <div className="pb-1 pl-[calc(0.75rem+2px)] pr-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {heading}
      </div>
      <div className="grid gap-0.5">{children}</div>
    </div>
  );
}

function NavigationMobileMenuComponent() {
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);
  const { status, user, logout } = useNavAuth();
  const showLock = useShowLock();
  const isActive = (link: NavLinkItem) =>
    link.kind === "internal" && location.pathname === link.to;

  const renderItem = (link: NavLinkItem, event = "nav_primary_click") => (
    <SheetClose asChild key={linkKey(link)}>
      <NavItemLink
        link={link}
        active={isActive(link)}
        locked={showLock(link)}
        onClick={() =>
          trackEvent(event, {
            nav_item: link.label,
            from_path: location.pathname,
            to_path: linkKey(link),
          })
        }
        className={cn(itemBase, isActive(link) ? itemActive : itemIdle)}
      />
    </SheetClose>
  );

  const accountLinks: InternalNavLink[] = [
    { kind: "internal", to: "/account", label: "Account" },
    { kind: "internal", to: "/favorites", label: "Favorites" },
  ];

  return (
    <div className="flex items-center md:hidden">
      <Sheet open={isOpen} onOpenChange={setIsOpen}>
        <SheetTrigger asChild>
          <Button variant="ghost" size="icon" className="h-10 w-10" aria-label="Open menu">
            <Menu className="h-5 w-5" aria-hidden="true" />
          </Button>
        </SheetTrigger>
        <SheetContent side="right" className="w-[min(320px,100vw)] max-w-full p-0">
          <div className="flex h-full flex-col">
            <SheetHeader className="border-b p-4 text-left">
              <div className="flex items-center gap-3">
                <SheetTitle className="flex items-center gap-3">
                  <NavLogo />
                  <span className="text-base font-semibold tracking-tight">Pandects</span>
                </SheetTitle>
                <NavBetaBadge />
              </div>
              <SheetDescription className="sr-only">
                Primary navigation links and account access.
              </SheetDescription>
            </SheetHeader>

            <nav aria-label="Mobile" className="flex-1 overflow-auto px-2 py-3">
              <div className="grid gap-0.5">{renderItem(SEARCH_LINK)}</div>

              {NAV_GROUPS.map((group) => (
                <SheetSection key={group.id} heading={group.label}>
                  {group.links.map((link) => renderItem(link, "nav_data_click"))}
                </SheetSection>
              ))}

              <div className="mt-4 grid gap-0.5">{renderItem(DOCS_LINK)}</div>

              <SheetSection heading="Account">
                {user ? (
                  <>
                    <div className="flex items-center gap-3 py-2 pl-[calc(0.75rem+2px)] pr-3">
                      <AvatarInitial email={user.email} />
                      <span className="min-w-0 truncate text-sm text-foreground">
                        {user.email}
                      </span>
                    </div>
                    {accountLinks.map((link) => renderItem(link, "nav_secondary_click"))}
                    <SheetClose asChild>
                      <button
                        type="button"
                        onClick={() => {
                          logout?.();
                          trackEvent("auth_logout", { from_path: location.pathname });
                        }}
                        className={cn(itemBase, itemIdle, "w-full text-left")}
                      >
                        Sign out
                      </button>
                    </SheetClose>
                  </>
                ) : status === "loading" ? (
                  <div className="ml-[calc(0.75rem+2px)] mr-3 h-9 animate-pulse rounded-md bg-muted/70" aria-hidden="true" />
                ) : (
                  <SheetClose asChild>
                    <Link
                      to="/account"
                      onClick={() => {
                        trackEvent("nav_signin_click", {
                          from_path: location.pathname,
                          to_path: "/account",
                        });
                        trackEvent("auth_open", { from_path: location.pathname, kind: "link" });
                      }}
                      className={cn(itemBase, itemIdle)}
                    >
                      Sign in
                    </Link>
                  </SheetClose>
                )}
              </SheetSection>

              <SheetSection heading="Project">
                {PROJECT_LINKS.map((link) => renderItem(link, "nav_secondary_click"))}
              </SheetSection>
            </nav>

            <div className="border-t border-border p-4">
              <div className="flex items-center justify-between gap-3">
                <span className="text-xs font-medium text-muted-foreground">Theme</span>
                <ThemeToggleInline />
              </div>
              <ul
                role="list"
                aria-label="Legal"
                className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground"
              >
                {LEGAL_LINKS.map((link) => (
                  <li key={link.to}>
                    <SheetClose asChild>
                      <Link
                        to={link.to}
                        className="rounded-sm hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      >
                        {link.label}
                      </Link>
                    </SheetClose>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </div>
  );
}

export default memo(NavigationMobileMenuComponent);
