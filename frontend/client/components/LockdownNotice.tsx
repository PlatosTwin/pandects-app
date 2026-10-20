import { Link, useLocation } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

// TEMPORARY (access lockdown): rendered in place of data-driven pages while the API
// is locked down to existing accounts. See lib/lockdown.ts for the gated paths.
export function LockdownNotice() {
  const location = useLocation();
  const next = `${location.pathname}${location.search}`;
  return (
    <PageShell title="Temporarily unavailable" size="md">
      <Card className="mx-auto w-full max-w-xl border-border bg-card/95 p-6 shadow-sm sm:p-8">
        <Alert>
          <AlertDescription>
            Access is temporarily limited to existing accounts, and new user creation is
            paused.
          </AlertDescription>
        </Alert>
        <div className="mt-6 text-center text-sm text-muted-foreground">
          Already have an account?{" "}
          <Button asChild variant="link" className="h-auto p-0 text-primary">
            <Link to={`/login?next=${encodeURIComponent(next)}`}>Sign in</Link>
          </Button>
        </div>
      </Card>
    </PageShell>
  );
}
