import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { LoadingSpinner } from "@/components/ui/loading-spinner";
import { useAuth } from "@/hooks/use-auth";
import {
  completeZitadelWebsiteAuth,
  finalizeZitadelWebsiteAuth,
} from "@/lib/auth-api";
import { setSessionToken } from "@/lib/auth-session";
import { authSessionTransport } from "@/lib/auth-transport";
import { safeNextPath } from "@/lib/auth-next";

type CallbackState =
  | { kind: "working" }
  | { kind: "error"; message: string }
  | { kind: "legal"; email: string; nextPath: string };

export default function AuthZitadelCallback() {
  const navigate = useNavigate();
  const { refresh } = useAuth();
  const [state, setState] = useState<CallbackState>({ kind: "working" });
  const [legalAccepted, setLegalAccepted] = useState(false);
  const [legalCheckedAtMs, setLegalCheckedAtMs] = useState<number | null>(null);
  const [submittingLegal, setSubmittingLegal] = useState(false);

  const search = useMemo(() => window.location.search, []);

  useEffect(() => {
    let cancelled = false;

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
      if (!cancelled) {
        navigate(safeNextPath(payload.next_path), { replace: true });
      }
    };

    const handleCallback = async () => {
      try {
        const params = new URLSearchParams(search);
        const oauthError = params.get("error");
        const oauthDescription = params.get("error_description");
        if (oauthError) {
          throw new Error(oauthDescription ?? oauthError);
        }
        const code = params.get("code");
        const callbackState = params.get("state");
        if (!code || !callbackState) {
          throw new Error("Missing auth provider authorization response.");
        }

        window.history.replaceState(
          null,
          document.title,
          window.location.pathname,
        );

        const result = await completeZitadelWebsiteAuth({ code, state: callbackState });
        if (cancelled) return;
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
        if (cancelled) return;
        const message = err instanceof Error ? err.message : String(err);
        setState({ kind: "error", message });
      }
    };

    void handleCallback();
    return () => {
      cancelled = true;
    };
  }, [navigate, refresh, search]);

  const submitLegal = async () => {
    if (state.kind !== "legal" || !legalCheckedAtMs) return;
    setSubmittingLegal(true);
    try {
      const result = await finalizeZitadelWebsiteAuth({
        checked_at_ms: legalCheckedAtMs,
        docs: ["tos", "privacy", "license"],
      });
      if (authSessionTransport() === "bearer") {
        if (!result.session_token) {
          throw new Error("Missing session token.");
        }
        setSessionToken(result.session_token);
      }
      await refresh();
      navigate(safeNextPath(result.next_path), { replace: true });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setState({ kind: "error", message });
    } finally {
      setSubmittingLegal(false);
    }
  };

  return (
    <PageShell
      title="Signing in…"
      subtitle="Completing your account sign-in."
      size="md"
    >
      <Card className="p-6">
        {state.kind === "working" ? (
          <div className="flex items-center gap-3 text-sm text-muted-foreground" role="status" aria-live="polite">
            <LoadingSpinner size="md" aria-label="Signing in" />
            Finishing sign-in…
          </div>
        ) : state.kind === "error" ? (
          <div className="grid gap-4" role="alert">
            <div>
              <div className="text-sm font-medium">Could not sign in</div>
              <div className="mt-1 text-sm text-muted-foreground">{state.message}</div>
            </div>
            <Button onClick={() => navigate("/account", { replace: true })}>
              Back to account
            </Button>
          </div>
        ) : (
          <div className="grid gap-4">
            <div>
              <h2 className="text-base font-medium">One more step</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                Accept the Pandects terms to finish creating or reactivating the account for{" "}
                <span className="font-medium text-foreground">{state.email}</span>.
              </p>
            </div>
            <div className="flex items-start gap-3 rounded-lg border border-border/60 bg-muted/20 p-4 text-sm">
              <Checkbox
                id="legal-zitadel"
                checked={legalAccepted}
                disabled={submittingLegal}
                onCheckedChange={(next) => {
                  const isChecked = next === true;
                  setLegalAccepted(isChecked);
                  setLegalCheckedAtMs(isChecked ? Date.now() : null);
                }}
              />
              <div className="leading-relaxed">
                <Label htmlFor="legal-zitadel" className="sr-only">
                  Accept legal terms
                </Label>
                I have read and agree to the{" "}
                <Link
                  to="/terms"
                  target="_blank"
                  rel="noreferrer"
                  className="text-primary hover:underline"
                >
                  Terms of Service
                </Link>
                ,{" "}
                <Link
                  to="/privacy-policy"
                  target="_blank"
                  rel="noreferrer"
                  className="text-primary hover:underline"
                >
                  Privacy Policy
                </Link>
                , and{" "}
                <Link
                  to="/license"
                  target="_blank"
                  rel="noreferrer"
                  className="text-primary hover:underline"
                >
                  License
                </Link>
                .
              </div>
            </div>
            <Button
              disabled={submittingLegal || !legalAccepted || !legalCheckedAtMs}
              onClick={() => void submitLegal()}
              className="w-full sm:w-auto"
            >
              {submittingLegal ? "Finishing sign-in…" : "Continue"}
            </Button>
          </div>
        )}
      </Card>
    </PageShell>
  );
}
