import { useMemo } from "react";
import { Link, Navigate, useLocation } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card } from "@/components/ui/card";
import { useAuth } from "@/hooks/use-auth";
import { nextPathRequiresDocumentNavigation, safeNextPath } from "@/lib/auth-next";

// TEMPORARY (September reopen): new account creation is disabled. Restore the prior
// signup form (Google + email/password + captcha + legal/verify flow) from
// version control to re-enable registration.
export default function Signup() {
  const { status } = useAuth();
  const location = useLocation();
  const nextPath = useMemo(
    () => safeNextPath(new URLSearchParams(location.search).get("next")),
    [location.search],
  );

  if (status === "authenticated") {
    return <Navigate to={nextPathRequiresDocumentNavigation(nextPath) ? "/account" : nextPath} replace />;
  }

  return (
    <PageShell title="Create account" size="md" className="max-w-3xl">
      <Card className="mx-auto w-full max-w-xl border-border bg-card/95 p-6 shadow-sm sm:p-8">
        <Alert>
          <AlertTitle>Account creation is temporarily disabled</AlertTitle>
          <AlertDescription>
            New user creation is temporarily disabled. Please check back in September.
          </AlertDescription>
        </Alert>
        <div className="mt-6 text-center text-sm text-muted-foreground">
          Already have an account?{" "}
          <Link
            to={`/login?next=${encodeURIComponent(nextPath)}`}
            className="font-medium text-primary hover:underline"
          >
            Sign in
          </Link>
          .
        </div>
      </Card>
    </PageShell>
  );
}
