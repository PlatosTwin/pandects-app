import { Link, useLocation } from "react-router-dom";
import { Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import { trackEvent } from "@/lib/analytics";
import { PROJECT_LINKS } from "@/components/navigation-links";
import { AvatarInitial } from "@/components/NavLinkParts";

export function AuthMenu() {
  const { status, user, logout } = useAuth();
  const location = useLocation();

  if (status === "loading") {
    return (
      <div className="h-9 w-20 animate-pulse rounded-md bg-muted/70" aria-hidden="true" />
    );
  }

  if (!user) {
    return (
      <Button
        asChild
        variant="ghost"
        className="h-9 border border-border bg-transparent px-3 text-sm text-muted-foreground hover:bg-muted/50 hover:text-foreground"
      >
        <Link
          to="/account"
          onClick={() => {
            trackEvent("nav_signin_click", { from_path: location.pathname, to_path: "/account" });
            trackEvent("auth_open", { from_path: location.pathname, kind: "link" });
          }}
        >
          Sign in
        </Link>
      </Button>
    );
  }

  const favoritesActive = location.pathname === "/favorites";

  return (
    <div className="flex items-center gap-1">
      <Button
        asChild
        variant="ghost"
        size="icon"
        className={cn(
          "h-9 w-9 rounded-md",
          favoritesActive ? "text-primary" : "text-muted-foreground hover:text-foreground",
        )}
      >
        <Link
          to="/favorites"
          aria-label="Favorites"
          title="Favorites"
          aria-current={favoritesActive ? "page" : undefined}
        >
          <Star
            className={cn("h-4 w-4", favoritesActive && "fill-current")}
            aria-hidden="true"
          />
        </Link>
      </Button>
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="h-9 w-9 rounded-full"
            aria-label={`Account menu for ${user.email}`}
            title={user.email}
          >
            <AvatarInitial email={user.email} />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-64">
          <DropdownMenuLabel className="flex items-center gap-3 font-normal">
            <AvatarInitial email={user.email} />
            <span className="min-w-0 truncate text-sm font-medium text-foreground">
              {user.email}
            </span>
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem asChild>
            <Link to="/account">Account</Link>
          </DropdownMenuItem>
          <DropdownMenuItem asChild>
            <Link to="/favorites">Favorites</Link>
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          {PROJECT_LINKS.map((link) => (
            <DropdownMenuItem key={link.to} asChild>
              <Link
                to={link.to}
                onClick={() =>
                  trackEvent("nav_secondary_click", {
                    nav_item: link.label,
                    from_path: location.pathname,
                    to_path: link.to,
                  })
                }
              >
                {link.label}
              </Link>
            </DropdownMenuItem>
          ))}
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onSelect={() => {
              logout();
              trackEvent("auth_logout", { from_path: location.pathname });
            }}
          >
            Sign out
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
