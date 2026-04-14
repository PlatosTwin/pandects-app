import { FormEvent, useEffect, useMemo, useState } from "react";
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
import { prewarmAuthBackend, withAuthWakeRetry } from "@/lib/auth-wake";

type SignupState =
  | { kind: "form" }
  | { kind: "legal"; email: string; nextPath: string }
  | { kind: "verify"; email: string };

function GoogleMark() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="h-5 w-5">
      <path
        d="M21.805 12.23c0-.68-.061-1.334-.174-1.963H12v3.713h5.498a4.703 4.703 0 0 1-2.04 3.086v2.564h3.3c1.93-1.777 3.047-4.4 3.047-7.4Z"
        fill="#4285F4"
      />
      <path
        d="M12 22c2.76 0 5.076-.915 6.768-2.47l-3.3-2.563c-.916.614-2.09.977-3.468.977-2.656 0-4.906-1.793-5.711-4.205H2.877v2.646A10.22 10.22 0 0 0 12 22Z"
        fill="#34A853"
      />
      <path
        d="M6.289 13.739A6.145 6.145 0 0 1 5.97 11.8c0-.674.116-1.328.319-1.939V7.215H2.877A10.197 10.197 0 0 0 1.8 11.8c0 1.627.389 3.168 1.077 4.585l3.412-2.646Z"
        fill="#FBBC05"
      />
      <path
        d="M12 5.656c1.5 0 2.847.517 3.908 1.534l2.93-2.93C17.07 2.614 14.754 1.6 12 1.6a10.22 10.22 0 0 0-9.123 5.615L6.29 9.86c.805-2.411 3.055-4.204 5.711-4.204Z"
        fill="#EA4335"
      />
    </svg>
  );
}

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

  useEffect(() => {
    void prewarmAuthBackend();
  }, []);

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
      const { authorize_url } = await withAuthWakeRetry(() =>
        startZitadelGoogleWebsiteAuth(nextPath),
      );
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
      className="max-w-3xl"
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
          <Card className="mx-auto w-full max-w-xl border-border/70 bg-card/95 p-6 shadow-sm">
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
          <Card className="mx-auto w-full max-w-2xl border-border/70 bg-card/95 p-6 shadow-sm sm:p-8">
            <div className="grid gap-5">
              <div className="grid gap-6">
                <div className="flex justify-center">
                  <Button
                    onClick={() => void startGoogle()}
                    disabled={submitting}
                    variant="outline"
                    className="h-11 min-w-[17rem] rounded-full border-border/80 bg-white px-5 text-foreground shadow-sm hover:bg-muted/40"
                  >
                    <GoogleMark />
                    {submitting ? "Redirecting…" : "Continue with Google"}
                  </Button>
                </div>
                <div className="flex items-center gap-4 text-[11px] uppercase tracking-[0.24em] text-muted-foreground">
                  <div className="flex-1 border-t border-border" aria-hidden="true" />
                  <span className="text-muted-foreground/95">or create an account with email</span>
                  <div className="flex-1 border-t border-border" aria-hidden="true" />
                </div>
              </div>
              <form className="grid gap-5" onSubmit={submit}>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="grid gap-2">
                    <Label
                      htmlFor="signup-first-name"
                      className="text-sm font-medium text-foreground/90"
                    >
                      First name
                    </Label>
                    <Input
                      id="signup-first-name"
                      autoComplete="given-name"
                      value={firstName}
                      onChange={(event) => setFirstName(event.target.value)}
                      className="h-11 border-border/80 bg-background"
                    />
                  </div>
                  <div className="grid gap-2">
                    <Label
                      htmlFor="signup-last-name"
                      className="text-sm font-medium text-foreground/90"
                    >
                      Last name
                    </Label>
                    <Input
                      id="signup-last-name"
                      autoComplete="family-name"
                      value={lastName}
                      onChange={(event) => setLastName(event.target.value)}
                      className="h-11 border-border/80 bg-background"
                    />
                  </div>
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="signup-email" className="text-sm font-medium text-foreground/90">
                    Email
                  </Label>
                  <Input
                    id="signup-email"
                    type="email"
                    autoComplete="email"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    required
                    className="h-11 border-border/80 bg-background"
                  />
                </div>
                <div className="grid gap-2">
                  <Label
                    htmlFor="signup-password"
                    className="text-sm font-medium text-foreground/90"
                  >
                    Password
                  </Label>
                  <Input
                    id="signup-password"
                    type="password"
                    autoComplete="new-password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    required
                    className="h-11 border-border/80 bg-background"
                  />
                </div>
                <p className="text-sm text-muted-foreground">
                  When you continue, you'll be prompted to accept our terms; then, you can finish
                  activation by verifying your email address.
                </p>
                <div className="flex justify-center pt-1">
                  <Button
                    type="submit"
                    disabled={submitting}
                    className="min-w-[12rem] rounded-full px-6"
                  >
                    {submitting ? (
                      <span className="inline-flex items-center gap-2">
                        <LoadingSpinner size="sm" aria-label="Creating account" />
                        Creating account…
                      </span>
                    ) : (
                      "Create account"
                    )}
                  </Button>
                </div>
              </form>

              <div className="text-center text-sm text-muted-foreground">
                Already have an account?{" "}
                <Link
                  to={`/login?next=${encodeURIComponent(nextPath)}`}
                  className="font-medium text-primary hover:underline"
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
