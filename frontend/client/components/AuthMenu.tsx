import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { trackEvent } from "@/lib/analytics";

export function AuthMenu() {
  const { status, user, logout } = useAuth();
  const location = useLocation();

  if (status === "loading") {
    return (
      <Button variant="ghost" className="h-9 px-3 text-sm" disabled>
        Loading…
      </Button>
    );
  }

  if (!user) {
    return (
      <Button asChild variant="outline" className="h-9 px-3 text-sm">
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

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" className="h-9 px-3 text-sm">
          {user.email}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuItem asChild>
          <Link to="/account">Account</Link>
        </DropdownMenuItem>
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
  );
}
