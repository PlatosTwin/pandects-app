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
      subtitle="Finish resetting your Pandects password."
      size="md"
    >
      <Card className="p-6">
        <div className="grid gap-6">
          {!userId || !code ? (
            <Alert variant="destructive">
              <AlertTitle>Invalid reset link</AlertTitle>
              <AlertDescription>
                The reset link is missing required information. Request a new password reset email.
              </AlertDescription>
            </Alert>
          ) : null}

          {submitted ? (
            <Alert>
              <AlertTitle>Password updated</AlertTitle>
              <AlertDescription>
                Your password has been updated. You can now sign in with the new password.
              </AlertDescription>
            </Alert>
          ) : null}

          {error ? (
            <Alert variant="destructive">
              <AlertTitle>Could not reset password</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}

          {!submitted ? (
            <form className="grid gap-4" onSubmit={submit}>
              <div className="grid gap-2">
                <Label htmlFor="new-password">New password</Label>
                <Input
                  id="new-password"
                  type="password"
                  autoComplete="new-password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  disabled={!userId || !code}
                  required
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="confirm-password">Confirm password</Label>
                <Input
                  id="confirm-password"
                  type="password"
                  autoComplete="new-password"
                  value={confirmPassword}
                  onChange={(event) => setConfirmPassword(event.target.value)}
                  disabled={!userId || !code}
                  required
                />
              </div>
              <Button type="submit" disabled={submitting || !userId || !code} className="w-full">
                {submitting ? "Updating password…" : "Update password"}
              </Button>
            </form>
          ) : null}

          <div className="text-sm text-muted-foreground">
            <Link to="/login" className="text-primary hover:underline">
              Back to sign in
            </Link>
          </div>
        </div>
      </Card>
    </PageShell>
  );
}
