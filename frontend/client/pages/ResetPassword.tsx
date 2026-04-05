import { FormEvent, useMemo, useState } from "react";
import { Link, Navigate, useLocation } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/hooks/use-auth";
import { requestPasswordReset } from "@/lib/auth-api";
import { safeNextPath } from "@/lib/auth-next";

export default function ResetPassword() {
  const { status } = useAuth();
  const location = useLocation();
  const nextPath = useMemo(
    () => safeNextPath(new URLSearchParams(location.search).get("next")),
    [location.search],
  );
  const [email, setEmail] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (status === "authenticated") {
    return <Navigate to={nextPath} replace />;
  }

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await requestPasswordReset({ email });
      setSubmitted(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <PageShell
      title="Reset password"
      subtitle="Request a Pandects password reset link for your email address."
      size="md"
    >
      <Card className="p-6">
        <div className="grid gap-6">
          {submitted ? (
            <Alert>
              <AlertTitle>Check your email</AlertTitle>
              <AlertDescription>
                If an account exists for <span className="font-medium">{email}</span>, we sent a password reset link.
              </AlertDescription>
            </Alert>
          ) : null}

          {error ? (
            <Alert variant="destructive">
              <AlertTitle>Could not request reset</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}

          <form className="grid gap-4" onSubmit={submit}>
            <div className="grid gap-2">
              <Label htmlFor="reset-email">Email</Label>
              <Input
                id="reset-email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
              />
            </div>
            <Button type="submit" disabled={submitting} className="w-full">
              {submitting ? "Sending reset link…" : "Send reset link"}
            </Button>
          </form>

          <div className="text-sm text-muted-foreground">
            Remembered it?{" "}
            <Link
              to={`/login?next=${encodeURIComponent(nextPath)}`}
              className="text-primary hover:underline"
            >
              Back to sign in
            </Link>
            .
          </div>
        </div>
      </Card>
    </PageShell>
  );
}
