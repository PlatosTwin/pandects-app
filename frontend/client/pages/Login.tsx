import { FormEvent, useMemo, useState } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { LegalAcceptancePrompt } from "@/components/auth/LegalAcceptancePrompt";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { LoadingSpinner } from "@/components/ui/loading-spinner";
import { useAuth } from "@/hooks/use-auth";
import {
  finalizeZitadelWebsiteAuth,
  loginWithPassword,
  resendEmailVerification,
  startZitadelGoogleWebsiteAuth,
} from "@/lib/auth-api";
import { safeNextPath } from "@/lib/auth-next";
import { setSessionToken } from "@/lib/auth-session";
import { authSessionTransport } from "@/lib/auth-transport";

type LoginState =
  | { kind: "form" }
  | { kind: "legal"; email: string; nextPath: string }
  | { kind: "verify"; email: string };

export default function Login() {
  const { status, refresh } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const nextPath = useMemo(
    () => safeNextPath(new URLSearchParams(location.search).get("next")),
    [location.search],
  );
  const initialEmail = useMemo(
    () => new URLSearchParams(location.search).get("email") ?? "",
    [location.search],
  );
  const [state, setState] = useState<LoginState>({ kind: "form" });
  const [email, setEmail] = useState(initialEmail);
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [legalAccepted, setLegalAccepted] = useState(false);
  const [legalCheckedAtMs, setLegalCheckedAtMs] = useState<number | null>(null);
  const canResendVerification =
    Boolean(email) &&
    error !== null &&
    error.toLowerCase().includes("verify your email before signing in");

  if (status === "authenticated") {
    return <Navigate to={nextPath} replace />;
  }

  const finishSession = async (payload: {
    next_path: string;
    session_token?: string;
  }) => {
    if (authSessionTransport() === "bearer") {
      if (!payload.session_token) {
        throw new Error("Missing session token.");
      }
      setSessionToken(payload.session_token);
    }
    await refresh();
    navigate(safeNextPath(payload.next_path), { replace: true });
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const result = await loginWithPassword({ email, password, next: nextPath });
      if (result.status === "legal_required") {
        setState({
          kind: "legal",
          email: result.user.email,
          nextPath: safeNextPath(result.next_path),
        });
        return;
      }
      if (result.status === "verification_required") {
        setState({
          kind: "verify",
          email: result.user.email,
        });
        return;
      }
      await finishSession(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const startGoogle = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const { authorize_url } = await startZitadelGoogleWebsiteAuth(nextPath);
      window.location.assign(authorize_url);
    } catch (err) {
      setSubmitting(false);
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const submitLegal = async () => {
    if (state.kind !== "legal" || !legalCheckedAtMs) return;
    setSubmitting(true);
    setError(null);
    try {
      const result = await finalizeZitadelWebsiteAuth({
        checked_at_ms: legalCheckedAtMs,
        docs: ["tos", "privacy", "license"],
      });
      if (result.status === "verification_required") {
        setState({
          kind: "verify",
          email: state.email,
        });
        return;
      }
      await finishSession(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const handleResendVerification = async () => {
    if (!email) return;
    setSubmitting(true);
    setError(null);
    try {
      const result = await resendEmailVerification({ email });
      setState({
        kind: "verify",
        email: result.user.email,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <PageShell
      title="Sign in"
      subtitle="Sign in to Pandects to manage API keys, review usage, and unlock full access."
      size="md"
    >
      <div className="grid gap-6">
        {error ? (
          <Alert variant="destructive">
            <AlertTitle>
              {state.kind === "form" ? "Could not sign in" : "Could not finish sign-in"}
            </AlertTitle>
            <AlertDescription>
              <div className="grid gap-3">
                <div>{error}</div>
                {email && error.toLowerCase().includes("accept the terms") ? (
                  <div>
                    <Link
                      to={`/signup?email=${encodeURIComponent(email)}&next=${encodeURIComponent(nextPath)}`}
                      className="text-primary hover:underline"
                    >
                      Resume account setup
                    </Link>
                  </div>
                ) : null}
                {canResendVerification ? (
                  <div>
                    <Button
                      type="button"
                      variant="link"
                      className="h-auto p-0 text-primary"
                      onClick={() => void handleResendVerification()}
                      disabled={submitting}
                    >
                      {submitting ? "Resending verification email…" : "Resend verification email"}
                    </Button>
                  </div>
                ) : null}
              </div>
            </AlertDescription>
          </Alert>
        ) : null}
        {state.kind === "legal" ? (
          <LegalAcceptancePrompt
            email={state.email}
            checked={legalAccepted}
            disabled={submitting}
            onCheckedChange={(checked) => {
              setLegalAccepted(checked);
              setLegalCheckedAtMs(checked ? Date.now() : null);
            }}
            onSubmit={() => void submitLegal()}
            submitLabel={submitting ? "Finishing sign-in…" : "Continue"}
          />
        ) : state.kind === "verify" ? (
          <Card className="p-6">
            <div className="grid gap-4">
              <div>
                <h2 className="text-base font-medium">Check your email</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  Verify the email for{" "}
                  <span className="font-medium text-foreground">{state.email}</span> to finish
                  activating the account, then sign in.
                </p>
              </div>
              <div className="text-sm text-muted-foreground">
                Need to restart setup?{" "}
                <Link
                  to={`/signup?email=${encodeURIComponent(state.email)}&next=${encodeURIComponent(nextPath)}`}
                  className="text-primary hover:underline"
                >
                  Resume account setup
                </Link>
                . Already verified?{" "}
                <button
                  type="button"
                  className="text-primary hover:underline"
                  onClick={() => {
                    setEmail(state.email);
                    setPassword("");
                    setError(null);
                    setState({ kind: "form" });
                    navigate(
                      `/login?email=${encodeURIComponent(state.email)}&next=${encodeURIComponent(nextPath)}`,
                      { replace: true },
                    );
                  }}
                >
                  Sign in
                </button>
                .
              </div>
            </div>
          </Card>
        ) : (
          <Card className="p-6">
            <div className="grid gap-6">
              <div className="grid gap-3">
                <Button onClick={() => void startGoogle()} disabled={submitting} className="w-full">
                  {submitting ? "Redirecting…" : "Continue with Google"}
                </Button>
                <div className="text-center text-xs uppercase tracking-[0.2em] text-muted-foreground">
                  or sign in with email
                </div>
              </div>
              <form className="grid gap-4" onSubmit={submit}>
                <div className="grid gap-2">
                  <Label htmlFor="login-email">Email</Label>
                  <Input
                    id="login-email"
                    type="email"
                    autoComplete="email"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    required
                  />
                </div>
                <div className="grid gap-2">
                  <div className="flex items-center justify-between gap-4">
                    <Label htmlFor="login-password">Password</Label>
                    <Link
                      to={`/reset-password?next=${encodeURIComponent(nextPath)}`}
                      className="text-sm text-primary hover:underline"
                    >
                      Forgot password?
                    </Link>
                  </div>
                  <Input
                    id="login-password"
                    type="password"
                    autoComplete="current-password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    required
                  />
                </div>
                <Button type="submit" disabled={submitting} className="w-full">
                  {submitting ? (
                    <span className="inline-flex items-center gap-2">
                      <LoadingSpinner size="sm" aria-label="Signing in" />
                      Signing in…
                    </span>
                  ) : (
                    "Sign in"
                  )}
                </Button>
              </form>

              <div className="text-sm text-muted-foreground">
                Need an account?{" "}
                <Link
                  to={`/signup?next=${encodeURIComponent(nextPath)}`}
                  className="text-primary hover:underline"
                >
                  Create one
                </Link>
                .
              </div>
            </div>
          </Card>
        )}
      </div>
    </PageShell>
  );
}
