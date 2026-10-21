import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { TurnstileWidget } from "@/components/TurnstileWidget";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/hooks/use-auth";
import { fetchCaptchaSiteKey, requestPasswordReset } from "@/lib/auth-api";
import { navigateToNextPath, nextPathRequiresDocumentNavigation, safeNextPath } from "@/lib/auth-next";
import { withAuthWakeRetry } from "@/lib/auth-wake";

export default function ResetPassword() {
  const { status } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const nextPath = useMemo(
    () => safeNextPath(new URLSearchParams(location.search).get("next")),
    [location.search],
  );
  const [email, setEmail] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submittedEmail, setSubmittedEmail] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [captchaSiteKey, setCaptchaSiteKey] = useState<string | null>(null);
  const [captchaToken, setCaptchaToken] = useState<string | null>(null);
  const [captchaError, setCaptchaError] = useState<string | null>(null);
  const [captchaResolved, setCaptchaResolved] = useState(false);
  const [captchaResetSignal, setCaptchaResetSignal] = useState(0);

  useEffect(() => {
    let canceled = false;
    void withAuthWakeRetry(() => fetchCaptchaSiteKey())
      .then((result) => {
        if (canceled) return;
        if (result.enabled) setCaptchaSiteKey(result.site_key);
        setCaptchaResolved(true);
      })
      .catch(() => {
        if (canceled) return;
        setCaptchaResolved(true);
      });
    return () => {
      canceled = true;
    };
  }, []);

  useEffect(() => {
    if (status !== "authenticated" || !nextPathRequiresDocumentNavigation(nextPath)) {
      return;
    }
    navigateToNextPath(navigate, nextPath, { replace: true });
  }, [navigate, nextPath, status]);

  if (status === "authenticated") {
    if (nextPathRequiresDocumentNavigation(nextPath)) {
      return null;
    }
    return <Navigate to={nextPath} replace />;
  }

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await requestPasswordReset({ email, captcha_token: captchaToken ?? undefined });
      setSubmittedEmail(email);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      // Turnstile tokens are single-use: drop the consumed token and ask the
      // widget for a fresh one so a retry doesn't fail on a stale token.
      setCaptchaToken(null);
      setCaptchaResetSignal((signal) => signal + 1);
      setSubmitting(false);
    }
  };

  return (
    <PageShell
      title="Reset password"
      description="Request a Pandects password reset link for your email address."
      layout="form"
    >
      <Card className="w-full p-6 shadow-sm sm:p-8">
        <div className="grid gap-6">
          {submittedEmail ? (
            <Alert>
              <AlertTitle as="h2">Check your email</AlertTitle>
              <AlertDescription>
                If an account exists for <span className="font-medium">{submittedEmail}</span>, we sent a password reset link.
              </AlertDescription>
            </Alert>
          ) : null}

          {error ? (
            <Alert variant="destructive">
              <AlertTitle as="h2">Could not request reset</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}

          {captchaError ? (
            <Alert variant="destructive">
              <AlertTitle as="h2">Captcha problem</AlertTitle>
              <AlertDescription>{captchaError}</AlertDescription>
            </Alert>
          ) : null}

          <form className="grid gap-5" onSubmit={submit}>
            <div className="grid gap-2">
              <Label htmlFor="reset-email" className="text-sm font-medium text-foreground/90">
                Email
              </Label>
              <Input
                id="reset-email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
                className="h-11 border-border bg-background"
              />
            </div>
            {captchaSiteKey ? (
              <TurnstileWidget
                siteKey={captchaSiteKey}
                onToken={(token) => {
                  setCaptchaToken(token);
                  if (token) setCaptchaError(null);
                }}
                onError={setCaptchaError}
                resetSignal={captchaResetSignal}
              />
            ) : null}
            <div className="pt-1">
              <Button
                type="submit"
                disabled={
                  submitting ||
                  !captchaResolved ||
                  (captchaSiteKey !== null && !captchaToken)
                }
                className="min-w-[10rem] px-6"
              >
                {submitting ? "Sending reset link…" : "Send reset link"}
              </Button>
            </div>
          </form>

          <div className="text-sm text-muted-foreground">
            Remembered it?{" "}
            <Link
              to={`/login?next=${encodeURIComponent(nextPath)}`}
              className="text-primary underline underline-offset-4 hover:text-primary/80"
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
