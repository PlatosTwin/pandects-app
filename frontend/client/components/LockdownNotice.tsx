import { Link, useLocation } from "react-router-dom";
import { Lock } from "lucide-react";
import { StatusPage } from "@/components/StatusPage";
import { Button } from "@/components/ui/button";

// TEMPORARY (access lockdown): rendered in place of data-driven pages while the API
// is locked down to existing accounts. See lib/lockdown.ts for the gated paths.
export function LockdownNotice() {
  const location = useLocation();
  const next = `${location.pathname}${location.search}`;
  return (
    <StatusPage
      icon={Lock}
      title="Temporarily unavailable"
      description="Access is temporarily limited to existing accounts, and new user creation is paused."
      actions={
        <>
          <Button asChild>
            <Link to={`/login?next=${encodeURIComponent(next)}`}>Sign in</Link>
          </Button>
          <Button asChild variant="outline">
            <Link to="/">Go home</Link>
          </Button>
        </>
      }
    />
  );
}
