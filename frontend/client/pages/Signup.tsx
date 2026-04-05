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
  signupWithPassword,
  startZitadelGoogleWebsiteAuth,
} from "@/lib/auth-api";
import { safeNextPath } from "@/lib/auth-next";
import { setSessionToken } from "@/lib/auth-session";
import { authSessionTransport } from "@/lib/auth-transport";

type SignupState =
  | { kind: "form" }
  | { kind: "legal"; email: string; nextPath: string }
  | { kind: "verify"; email: string };

export default function Signup() {
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
  const [state, setState] = useState<SignupState>({ kind: "form" });
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState(initialEmail);
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [legalAccepted, setLegalAccepted] = useState(false);
  const [legalCheckedAtMs, setLegalCheckedAtMs] = useState<number | null>(null);

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
      const result = await signupWithPassword({
        email,
        password,
        first_name: firstName || undefined,
        last_name: lastName || undefined,
        next: nextPath,
      });
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

  return (
    <PageShell
      title="Create account"
      subtitle="Create a Pandects account with email and password, or continue directly with Google."
      size="md"
    >
      <div className="grid gap-6">
        {error ? (
          <Alert variant="destructive">
            <AlertTitle>
              {state.kind === "form" ? "Could not create account" : "Could not finish account setup"}
            </AlertTitle>
            <AlertDescription>{error}</AlertDescription>
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
            submitLabel={submitting ? "Finishing account setup…" : "Continue"}
          />
        ) : state.kind === "verify" ? (
          <Card className="p-6">
            <div className="grid gap-4">
              <div>
                <h2 className="text-base font-medium">Check your email</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  We sent a verification link to{" "}
                  <span className="font-medium text-foreground">{state.email}</span>. Verify your
                  email to finish activating the account, then sign in.
                </p>
              </div>
              <div className="text-sm text-muted-foreground">
                Already verified?{" "}
                <Link
                  to={`/login?next=${encodeURIComponent(nextPath)}`}
                  className="text-primary hover:underline"
                >
                  Sign in
                </Link>
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
                  or create an account with email
                </div>
              </div>
              <form className="grid gap-4" onSubmit={submit}>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="grid gap-2">
                    <Label htmlFor="signup-first-name">First name</Label>
                    <Input
                      id="signup-first-name"
                      autoComplete="given-name"
                      value={firstName}
                      onChange={(event) => setFirstName(event.target.value)}
                    />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="signup-last-name">Last name</Label>
                    <Input
                      id="signup-last-name"
                      autoComplete="family-name"
                      value={lastName}
                      onChange={(event) => setLastName(event.target.value)}
                    />
                  </div>
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="signup-email">Email</Label>
                  <Input
                    id="signup-email"
                    type="email"
                    autoComplete="email"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    required
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="signup-password">Password</Label>
                  <Input
                    id="signup-password"
                    type="password"
                    autoComplete="new-password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    required
                  />
                </div>
                <p className="text-sm text-muted-foreground">
                  By continuing you will create a Pandects account and finish activation after accepting the platform terms.
                </p>
                <Button type="submit" disabled={submitting} className="w-full">
                  {submitting ? (
                    <span className="inline-flex items-center gap-2">
                      <LoadingSpinner size="sm" aria-label="Creating account" />
                      Creating account…
                    </span>
                  ) : (
                    "Create account"
                  )}
                </Button>
              </form>

              <div className="text-sm text-muted-foreground">
                Already have an account?{" "}
                <Link
                  to={`/login?next=${encodeURIComponent(nextPath)}`}
                  className="text-primary hover:underline"
                >
                  Sign in
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
