import { FormEvent, useMemo, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { confirmPasswordReset } from "@/lib/auth-api";

export default function ResetPasswordConfirm() {
  const location = useLocation();
  const params = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const userId = params.get("user_id") ?? params.get("userID") ?? "";
  const code = params.get("code") ?? "";
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await confirmPasswordReset({ user_id: userId, code, password });
      setSubmitted(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <PageShell
      title="Choose a new password"
      description="Finish resetting your Pandects password."
      layout="form"
    >
      <Card className="w-full p-6 shadow-sm sm:p-8">
        <div className="grid gap-6">
          {!userId || !code ? (
            <Alert variant="destructive">
              <AlertTitle as="h2">Invalid reset link</AlertTitle>
              <AlertDescription>
                The reset link is missing required information. Request a new password reset email.
              </AlertDescription>
            </Alert>
          ) : null}

          {submitted ? (
            <Alert>
              <AlertTitle as="h2">Password updated</AlertTitle>
              <AlertDescription>
                Your password has been updated. You can now sign in with the new password.
              </AlertDescription>
            </Alert>
          ) : null}

          {error ? (
            <Alert variant="destructive">
              <AlertTitle as="h2">Could not reset password</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}

          {!submitted ? (
            <form className="grid gap-5" onSubmit={submit}>
              <div className="grid gap-2">
                <Label htmlFor="new-password" className="text-sm font-medium text-foreground/90">
                  New password
                </Label>
                <Input
                  id="new-password"
                  type="password"
                  autoComplete="new-password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  disabled={!userId || !code}
                  required
                  className="h-11 border-border bg-background"
                />
              </div>
              <div className="grid gap-2">
                <Label
                  htmlFor="confirm-password"
                  className="text-sm font-medium text-foreground/90"
                >
                  Confirm password
                </Label>
                <Input
                  id="confirm-password"
                  type="password"
                  autoComplete="new-password"
                  value={confirmPassword}
                  onChange={(event) => setConfirmPassword(event.target.value)}
                  disabled={!userId || !code}
                  required
                  className="h-11 border-border bg-background"
                />
              </div>
              <div className="pt-1">
                <Button
                  type="submit"
                  disabled={submitting || !userId || !code}
                  className="min-w-[10rem] px-6"
                >
                  {submitting ? "Updating password…" : "Update password"}
                </Button>
              </div>
            </form>
          ) : null}

          <div className="text-sm text-muted-foreground">
            <Link to="/login" className="text-primary underline underline-offset-4 hover:text-primary/80">
              Back to sign in
            </Link>
          </div>
        </div>
      </Card>
    </PageShell>
  );
}
