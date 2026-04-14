import { useEffect, useMemo, useState } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { LegalAcceptancePrompt } from "@/components/auth/LegalAcceptancePrompt";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card } from "@/components/ui/card";
import { useAuth } from "@/hooks/use-auth";
import {
  completeEmailVerification,
  finalizeZitadelWebsiteAuth,
} from "@/lib/auth-api";
import { navigateToNextPath, nextPathRequiresDocumentNavigation, safeNextPath } from "@/lib/auth-next";
import { setSessionToken } from "@/lib/auth-session";
import { authSessionTransport } from "@/lib/auth-transport";

type VerifyState =
  | { kind: "loading" }
  | { kind: "legal"; email: string; nextPath: string }
  | { kind: "done" };

export default function VerifyEmail() {
  const { status, refresh } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const params = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const nextPath = useMemo(
    () => safeNextPath(params.get("next")),
    [params],
  );
  const userId = params.get("user_id") ?? params.get("userID") ?? "";
  const code = params.get("code") ?? "";

  const [state, setState] = useState<VerifyState>({ kind: "loading" });
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [legalAccepted, setLegalAccepted] = useState(false);
  const [legalCheckedAtMs, setLegalCheckedAtMs] = useState<number | null>(null);

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
    navigateToNextPath(navigate, payload.next_path, { replace: true });
  };

  useEffect(() => {
    if (status !== "authenticated" || !nextPathRequiresDocumentNavigation(nextPath)) {
      return;
    }
    navigateToNextPath(navigate, nextPath, { replace: true });
  }, [navigate, nextPath, status]);

  useEffect(() => {
    let active = true;
    const run = async () => {
      if (!userId || !code) {
        setError("The verification link is missing required information.");
        setState({ kind: "done" });
        return;
      }
      try {
        const result = await completeEmailVerification({
          user_id: userId,
          code,
          next: nextPath,
        });
        if (!active) return;
        if (result.status === "legal_required") {
          setState({
            kind: "legal",
            email: result.user.email,
            nextPath: safeNextPath(result.next_path),
          });
          return;
        }
        await finishSession(result);
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : String(err));
        setState({ kind: "done" });
      }
    };
    void run();
    return () => {
      active = false;
    };
  }, [code, nextPath, userId]);

  const submitLegal = async () => {
    if (state.kind !== "legal" || !legalCheckedAtMs) return;
    setSubmitting(true);
    setError(null);
    try {
      const result = await finalizeZitadelWebsiteAuth({
        checked_at_ms: legalCheckedAtMs,
        docs: ["tos", "privacy", "license"],
      });
      await finishSession(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  if (status === "authenticated") {
    if (nextPathRequiresDocumentNavigation(nextPath)) {
      return null;
    }
    return <Navigate to={nextPath} replace />;
  }

  return (
    <PageShell
      title="Verify your email"
      subtitle="Completing your Pandects email verification."
      size="md"
    >
      <div className="grid gap-6">
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
        ) : (
          <Card className="p-6">
            <div className="grid gap-4">
              {!error ? (
                <Alert>
                  <AlertTitle>Verifying your email</AlertTitle>
                  <AlertDescription>
                    Please wait while Pandects completes your email verification.
                  </AlertDescription>
                </Alert>
              ) : (
                <Alert variant="destructive">
                  <AlertTitle>Could not verify email</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
              <div className="text-sm text-muted-foreground">
                <Link to="/login" className="text-primary hover:underline">
                  Back to sign in
                </Link>
              </div>
            </div>
          </Card>
        )}
      </div>
    </PageShell>
  );
}
